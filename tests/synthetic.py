"""Synthetic paired scores with known ground truth, for testing the gate (README §7).

Scores are sigmoid(logit). Benign logits are N(-separation, 1) and attack logits N(0, 1),
so the scorer's AUROC is Phi(separation / sqrt(2)). The candidate sees the same examples
as the reference, plus run-to-run noise in the logit and optionally a shift.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import norm

import hulder

# Sizes of the S-Labs calibration set (docs/results/phase1/probe_splits.json).
N_BENIGN = 1191
N_ATTACK = 910

# Run-to-run noise of two fresh fp16 engines, measured with the
# main probe on the calibration split, fp16 vs fp16_run1: 44% of scores are bit-identical
# and the logit differences have sd 3.65e-3. Zero-inflated N(0, 4.9e-3) matches both, and
# predicts mean |delta logit| 2.2e-3 and p99 1.16e-2 against a measured 2.17e-3 and 1.15e-2.
FRESH_ENGINE_P_ZERO = 0.44
FRESH_ENGINE_NOISE = 4.9e-3


@dataclass
class Paired:
    reference: np.ndarray
    candidate: np.ndarray
    labels: np.ndarray
    threshold: float


def unchanged_scorer(
    seed: int,
    target_fpr: float = 0.01,
    n_benign: int = N_BENIGN,
    n_attack: int = N_ATTACK,
    separation: float = 3.5,
    noise: float = 1e-3,
    p_zero: float = 0.0,
    candidate_fpr: float | None = None,
    attack_shift: float = 0.0,
    missed_attack_shift: float = 0.0,
) -> Paired:
    """Paired scores. With the defaults it is a null case: the candidate differs by noise only.

    The threshold is fitted on the reference's own benign scores, as in the real workflow
    (experiments/phase1/probe/train.py), so the reference's FPR is at most target_fpr by
    construction. With the defaults the FNR is about 12% and the AUROC about 0.993.

    A fraction p_zero of candidate scores is left bit-identical to the reference. If
    candidate_fpr is given, every candidate logit is shifted by the same amount so that the
    candidate's population FPR at the fitted threshold is candidate_fpr. attack_shift moves
    only the attack examples' candidate logits, which changes FNR and AUROC but not FPR.
    missed_attack_shift (<= 0) moves only attacks already below the threshold further
    down, which lowers AUROC but changes neither FNR nor FPR.
    """
    rng = np.random.default_rng(seed)
    labels = np.r_[np.zeros(n_benign, int), np.ones(n_attack, int)]
    logits = np.r_[rng.normal(-separation, 1, n_benign), rng.normal(0, 1, n_attack)]
    noisy = logits + rng.normal(0, noise, logits.shape) * (rng.random(logits.shape) >= p_zero)
    reference = expit(logits)
    threshold = float(np.quantile(reference[labels == 0], 1 - target_fpr, method="higher"))
    if candidate_fpr is not None:
        noisy += logit(threshold) + separation - norm.ppf(1 - candidate_fpr)
    noisy[labels == 1] += attack_shift
    noisy[(labels == 1) & (noisy <= logit(threshold))] += missed_attack_shift
    return Paired(reference, expit(noisy), labels, threshold)


def missed_attack_delta_auroc(shift: float, separation: float = 3.5, target_fpr: float = 0.01) -> float:
    """Population AUROC change from missed_attack_shift.

    Attacks A ~ N(0, 1) at or below the threshold's logit T = Phi^-1(1 - target_fpr) - separation
    move to A + shift, and each such attack now outscores a benign B ~ N(-separation, 1) with
    chance Phi(A + shift + separation) instead of Phi(A + separation). So
    delta = E[1{A <= T} (Phi(A + shift + separation) - Phi(A + separation))].
    """
    t = norm.ppf(1 - target_fpr) - separation
    gain = lambda a: norm.pdf(a) * (norm.cdf(a + shift + separation) - norm.cdf(a + separation))
    return quad(gain, -np.inf, t)[0]


def missed_attack_shift_for(delta_auroc: float, separation: float = 3.5, target_fpr: float = 0.01) -> float:
    """The missed_attack_shift whose population AUROC change is delta_auroc. Raises ValueError
    beyond the floor, where every missed attack already sits below every benign example."""
    return brentq(lambda s: missed_attack_delta_auroc(s, separation, target_fpr) - delta_auroc, -50, 0)


def blur_for(delta_auroc: float, separation: float = 3.5) -> float:
    """The candidate logit noise sd whose population AUROC change is delta_auroc.

    With noise sd s on every logit, the classes are N(-separation, 1 + s^2) and N(0, 1 + s^2),
    so the AUROC is Phi(separation / sqrt(2 (1 + s^2))). Unlike missed_attack_shift this
    also raises FPR at the reference's threshold.
    """
    z = norm.ppf(norm.cdf(separation / np.sqrt(2)) + delta_auroc)
    return float(np.sqrt((separation / z) ** 2 / 2 - 1))


def reports(make: Callable[[int], Paired], seeds: range, target_fpr: float = 0.01) -> list[hulder.Report]:
    """One check per seed."""
    out = []
    for seed in seeds:
        d = make(seed)
        out.append(hulder.check(d.reference, d.candidate, d.labels, threshold=d.threshold, target_fpr=target_fpr))
    return out


def verdict_rates(make: Callable[[int], Paired], seeds: range, target_fpr: float = 0.01) -> dict:
    """Share of seeds giving each verdict, keyed by the verdict's value."""
    counts = Counter(r.verdict for r in reports(make, seeds, target_fpr))
    return {v.value: counts[v] / len(seeds) for v in hulder.Verdict}
