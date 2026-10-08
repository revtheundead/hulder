"""The release gate: compare a scorer under reference and candidate serving configs."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import beta, binomtest, rankdata

N_BOOTSTRAP = 1000


class Verdict(Enum):
    PASS = "PASS"
    THRESHOLD_STALE = "THRESHOLD STALE"
    SCORER_DEGRADED = "SCORER DEGRADED"
    INCONCLUSIVE = "INCONCLUSIVE"

    @property
    def exit_code(self) -> int:
        """CLI exit code from the exit codes. Verdicts start at 10 so they never collide with failures."""
        return _EXIT_CODES[self]


_EXIT_CODES = {
    Verdict.PASS: 0,
    Verdict.THRESHOLD_STALE: 10,
    Verdict.SCORER_DEGRADED: 11,
    Verdict.INCONCLUSIVE: 12,
}


@dataclass(frozen=True)
class Report:
    """Plain data: it neither prints nor exits, and compares with ==."""

    verdict: Verdict
    # Share of examples whose decision (score above the threshold or not) differs between
    # reference and candidate. A diagnostic, never an input to the verdict.
    decision_flip_rate: float
    decision_flips: int
    # Mean |candidate - reference| over all examples, in score space.
    mean_abs_shift: float
    # The same two numbers for a second run of the reference against the reference, and
    # whether the candidate moved scores more than that rerun did. None when no rerun was
    # given: the noise floor was not measured.
    noise_floor_decision_flips: int | None = None
    noise_floor_mean_abs_shift: float | None = None
    shift_exceeds_noise_floor: bool | None = None
    warnings: tuple[str, ...] = ()
    # Set only for INCONCLUSIVE: extra benign examples from which on the FPR condition
    # stays decided, if the candidate's observed FPR held. 0 when that condition is
    # already decided, so benign examples alone won't resolve the INCONCLUSIVE. None when
    # it would take more than a million in all.
    more_benign_needed: int | None = None


def check(
    reference: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    target_fpr: float,
    seed: int = 0,
    noise_floor: np.ndarray | None = None,
) -> Report:
    """noise_floor is an optional second run of the reference on the same examples. It feeds
    only the shift diagnostics, never the verdict."""
    benign = labels == 0
    warnings = []
    k, n, ci = _flag_rate(reference[benign], threshold)
    if not ci.low <= target_fpr <= ci.high:
        warnings.append(
            f"The reference's FPR at threshold {threshold:.4g} is {k / n:.2%} "
            f"[95% CI {ci.low:.2%}, {ci.high:.2%}], which excludes the {target_fpr:.2%} target, "
            "so the threshold wasn't calibrated on this snapshot. "
            "Verdicts then mix threshold generalisation with the serving change: "
            "recalibrate on the reference first."
        )
    _, _, candidate_fpr = _flag_rate(candidate[benign], threshold)
    # FNR is the share of attacks not flagged: 1 minus their flag rate, CI bounds swapped.
    k, n, _ = _flag_rate(reference[~benign], threshold)
    reference_fnr = 1 - k / n
    _, _, candidate_flagged = _flag_rate(candidate[~benign], threshold)
    candidate_fnr_low, candidate_fnr_high = 1 - candidate_flagged.high, 1 - candidate_flagged.low
    delta_auroc_low, delta_auroc_high = _paired_bootstrap_delta_auroc(reference, candidate, labels, seed)
    stale = (
        candidate_fpr.low > 2 * target_fpr
        or candidate_fpr.high < target_fpr / 2
        or candidate_fnr_low > reference_fnr + 0.05
    )
    if delta_auroc_high < -0.02:
        verdict = Verdict.SCORER_DEGRADED
    elif stale:
        verdict = Verdict.THRESHOLD_STALE
    elif (
        candidate_fpr.high <= 2 * target_fpr
        and candidate_fnr_high <= reference_fnr + 0.05
        and delta_auroc_low >= -0.02
    ):  # fmt: skip
        verdict = Verdict.PASS
    else:
        verdict = Verdict.INCONCLUSIVE
    more_benign_needed = None
    if verdict is Verdict.INCONCLUSIVE:
        k, n, _ = _flag_rate(candidate[benign], threshold)
        more_benign_needed = _more_needed_to_decide(k / n, n, 2 * target_fpr)
    flipped = (reference > threshold) != (candidate > threshold)
    floor = {}
    if noise_floor is not None:
        candidate_shift, rerun_shift = np.abs(candidate - reference), np.abs(noise_floor - reference)
        low, _ = _paired_bootstrap_mean(candidate_shift - rerun_shift, seed)
        floor = {
            "noise_floor_decision_flips": int(((reference > threshold) != (noise_floor > threshold)).sum()),
            "noise_floor_mean_abs_shift": float(rerun_shift.mean()),
            "shift_exceeds_noise_floor": bool(low > 0),
        }
    return Report(
        verdict=verdict,
        decision_flip_rate=float(flipped.mean()),
        decision_flips=int(flipped.sum()),
        mean_abs_shift=float(np.abs(candidate - reference).mean()),
        warnings=tuple(warnings),
        more_benign_needed=more_benign_needed,
        **floor,
    )


def _flag_rate(scores: np.ndarray, threshold: float):
    """Number flagged (score above threshold), number of scores, and the rate's 95% Clopper-Pearson CI."""
    k, n = int((scores > threshold).sum()), len(scores)
    return k, n, binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")


def _more_needed_to_decide(rate: float, n: int, line: float, max_n: int = 10**6) -> int | None:
    """Fewest extra examples from which on, at the same observed rate, the 95%
    Clopper-Pearson CI stays wholly on one side of line. 0 if it already does from n on;
    None if it doesn't by max_n.

    With the count rate * size treated as continuous the CI bound is monotone in size, so a
    bisection finds where it crosses the line. Real counts are integers and rounding makes
    the decision flicker near that point, so the answer is one past the last size up to
    twice the crossing whose rounded count is undecided.
    """
    if not _decided(rate, np.array([max_n]), line)[0]:
        return None
    lo, hi = n - 1, max_n  # continuous: hi is decided; lo is a sentinel below n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if _decided(rate, np.array([mid]), line)[0] else (mid, hi)
    sizes = np.arange(n, min(2 * hi, max_n) + 1)
    undecided = np.flatnonzero(~_decided(rate, sizes, line, rounded=True))
    return int(sizes[undecided[-1]]) + 1 - n if len(undecided) else 0


def _decided(rate: float, sizes: np.ndarray, line: float, rounded: bool = False) -> np.ndarray:
    """Whether the 95% Clopper-Pearson CI of rate * size successes lies on one side of line."""
    k = np.round(rate * sizes) if rounded else rate * sizes
    with np.errstate(all="ignore"):
        low = np.where(k > 0, beta.ppf(0.025, k, sizes - k + 1), 0.0)
        high = np.where(k < sizes, beta.ppf(0.975, k + 1, sizes - k), 1.0)
    return (high <= line) | (low > line)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Reference implementation (Mann-Whitney): the chance an attack outscores a benign
    example, ties counting half."""
    ranks = rankdata(scores)
    attack = labels == 1
    n_attack, n_benign = int(attack.sum()), int((~attack).sum())
    return float((ranks[attack].sum() - n_attack * (n_attack + 1) / 2) / (n_attack * n_benign))


def _paired_bootstrap_delta_auroc(
    reference: np.ndarray, candidate: np.ndarray, labels: np.ndarray, seed: int
) -> tuple[float, float]:
    """95% percentile CI of AUROC(candidate) - AUROC(reference).

    Resampling is stratified by class, and each resample uses the same examples for both
    configs, so the comparison stays paired.
    """
    rng = np.random.default_rng(seed)
    n_benign, n_attack = int((labels == 0).sum()), int((labels == 1).sum())
    w_benign = _resample_counts(rng, n_benign)
    w_attack = _resample_counts(rng, n_attack)
    delta = _weighted_auroc(candidate, labels, w_benign, w_attack) - _weighted_auroc(
        reference, labels, w_benign, w_attack
    )
    low, high = np.quantile(delta, [0.025, 0.975])
    return float(low), float(high)


def _paired_bootstrap_mean(differences: np.ndarray, seed: int) -> tuple[float, float]:
    """95% percentile CI of the mean of per-example differences, resampling examples."""
    # A stream apart from the ΔAUROC bootstrap's, so adding this doesn't change that CI.
    rng = np.random.default_rng((seed, 1))
    means = _resample_counts(rng, len(differences)) @ differences / len(differences)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _resample_counts(rng: np.random.Generator, n: int) -> np.ndarray:
    """(N_BOOTSTRAP, n) multiplicities of n draws with replacement from n examples."""
    draws = rng.integers(0, n, (N_BOOTSTRAP, n)) + n * np.arange(N_BOOTSTRAP)[:, None]
    return np.bincount(draws.ravel(), minlength=N_BOOTSTRAP * n).reshape(N_BOOTSTRAP, n)


def _weighted_auroc(scores: np.ndarray, labels: np.ndarray, w_benign: np.ndarray, w_attack: np.ndarray) -> np.ndarray:
    """_auroc of each bootstrap resample, given as rows of per-example multiplicities.

    Equivalent to copying each example w times and calling _auroc, but costs a cumulative
    sum per resample instead of a sort: for each attack, sum the weight of benign examples
    below it plus half of those tied with it.
    """
    benign_scores = scores[labels == 0]
    order = np.argsort(benign_scores, kind="stable")
    sorted_benign = benign_scores[order]
    cum = np.zeros((len(w_benign), len(order) + 1), dtype=w_benign.dtype)
    cum[:, 1:] = np.cumsum(w_benign[:, order], axis=1)
    attack_scores = scores[labels == 1]
    below = cum[:, np.searchsorted(sorted_benign, attack_scores, side="left")]
    tied = cum[:, np.searchsorted(sorted_benign, attack_scores, side="right")] - below
    return (w_attack * (below + tied / 2)).sum(axis=1) / (len(order) * w_attack.shape[1])
