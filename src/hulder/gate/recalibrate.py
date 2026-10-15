"""Recalibration: a new threshold for the same target FPR, fitted on the candidate's benign scores
(docs/design.md §6). The scorer is unchanged; only the threshold moves."""

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, Context

import numpy as np
from scipy.stats import binom

# The recalibrated threshold keeps the FPR at or below the target with this probability over
# calibration sets: the same 95% as the verdict rule's bounds.
CONFIDENCE = 0.95


@dataclass(frozen=True)
class Fit:
    """A binomial-tail threshold, or why there is none. Exactly one of threshold and
    benign_needed is None."""

    # Flag a score strictly above it. None when the benign count can't certify the target.
    threshold: float | None
    # Benign calibration scores above the threshold.
    benign_above: int | None
    # The fewest benign examples in all that could certify the target, when these can't.
    benign_needed: int | None


def binomial_tail_threshold(
    benign_scores: np.ndarray,
    target_fpr: float,
    confidence: float = CONFIDENCE,
    *,
    attack_scores: np.ndarray | None = None,
) -> Fit:
    """The most permissive threshold whose FPR is at most target_fpr with the given confidence.

    With j benign scores allowed above it, the threshold's true FPR exceeds the target only if at
    most j of n benign draws land above the target's quantile, which has probability
    P(Bin(n, target_fpr) <= j). So j is the largest count with that probability at most
    1 - confidence. Ties can only leave fewer than j above it, which is conservative.

    The threshold is then rounded up to a short decimal that flags the same calibration
    examples, benign and, when attack_scores are given, attack, so the candidate's FNR at it
    is what the order statistic itself would give."""
    scores = np.sort(np.asarray(benign_scores, dtype=np.float64))
    n = len(scores)
    _validate(n, target_fpr)
    delta = 1 - confidence
    allowed = int(binom.ppf(delta, n, target_fpr))
    # ppf gives the smallest j with cdf(j) >= delta; step back to the largest j with cdf(j) <= delta.
    while allowed >= 0 and binom.cdf(allowed, n, target_fpr) > delta:
        allowed -= 1
    if allowed < 0:
        # (1 - target)**m <= delta is P(Bin(m, target) <= 0) <= delta.
        needed = math.ceil(math.log(delta) / math.log1p(-target_fpr))
        return Fit(threshold=None, benign_above=None, benign_needed=needed)
    last_unflagged = scores[n - allowed - 1]
    every = scores if attack_scores is None else np.r_[scores, np.asarray(attack_scores, dtype=np.float64)]
    higher = every[every > last_unflagged]
    threshold = short_threshold(last_unflagged, float(higher.min()) if len(higher) else math.inf)
    return Fit(threshold=threshold, benign_above=int((scores > threshold).sum()), benign_needed=None)


def conformal_threshold(benign_scores: np.ndarray, target_fpr: float) -> float | None:
    """Split-conformal threshold: the ceil((n + 1)(1 - target_fpr))-th smallest benign score, whose
    FPR is at most target_fpr in expectation over calibration sets, not with high probability.
    None when that rank passes n. Reference implementation for the recalibration study only."""
    scores = np.sort(np.asarray(benign_scores, dtype=np.float64))
    n = len(scores)
    _validate(n, target_fpr)
    rank = math.ceil((n + 1) * (1 - target_fpr))
    return None if rank > n else float(scores[rank - 1])


def short_threshold(lo: float, hi: float) -> float:
    """The value in [lo, hi) with the fewest significant digits, rounding lo towards +inf.

    Every value there flags the same calibration scores, and none flags more of any other
    distribution than lo does, so the FPR guarantee holds. A short value prints exactly, so
    passing the printed threshold back to check reproduces the fit."""
    for digits in range(1, 18):
        value = float(Context(prec=digits, rounding=ROUND_CEILING).create_decimal(repr(float(lo))))
        if lo <= value < hi:
            return value
    return float(lo)


def _validate(n: int, target_fpr: float) -> None:
    if n == 0:
        raise ValueError("no benign scores to fit a threshold on")
    if not 0 < target_fpr <= 0.5:  # check.MAX_TARGET_FPR, not imported: check imports this module
        raise ValueError(f"target_fpr is {target_fpr}; it must be a rate in (0, 0.5]")
