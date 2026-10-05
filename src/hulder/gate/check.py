"""The release gate: compare a scorer under reference and candidate serving configs."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import binomtest


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
    warnings: tuple[str, ...] = ()


def check(
    reference: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    target_fpr: float,
) -> Report:
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
    candidate_fnr_low = 1 - candidate_flagged.high
    stale = (
        candidate_fpr.low > 2 * target_fpr
        or candidate_fpr.high < target_fpr / 2
        or candidate_fnr_low > reference_fnr + 0.05
    )
    verdict = Verdict.THRESHOLD_STALE if stale else Verdict.PASS
    return Report(verdict=verdict, warnings=tuple(warnings))


def _flag_rate(scores: np.ndarray, threshold: float):
    """Number flagged (score above threshold), number of scores, and the rate's 95% Clopper-Pearson CI."""
    k, n = int((scores > threshold).sum()), len(scores)
    return k, n, binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
