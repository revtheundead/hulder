"""Candidate 95% intervals for a paired difference of proportions, for the paired FPR and FNR
conditions (docs/design.md §6). The coverage run (paired_coverage.py) picks one of them.

Every function takes the four cells of a paired 2×2 table over the same n examples:
- a, flagged by both runs;
- b, flagged by the candidate only (newly flagged);
- c, flagged by the reference only (newly unflagged);
- n, all examples, so the fourth cell is n − a − b − c.

and returns (low, high) for the candidate's rate minus the reference's, (b − c) / n. For FNR,
"flagged" reads as "missed". Bounds are the ends of a two-sided interval at `alpha`, as the gate's
Clopper-Pearson bounds are.

Tango's score interval and Newcombe's square-and-add interval follow the code of the R package
contingencytables 3.1.1 (Fagerland, Lydersen and Laake), functions
Tango_asymptotic_score_CI_paired_2x2 and Newcombe_square_and_add_CI_paired_2x2, with its
n[1, 2] = b and n[2, 1] = c. The continuity-corrected variant swaps in Wilson intervals with
continuity correction, as Newcombe's paper describes for its paired method 11. That reading of the
paper is unchecked against its text, and the Wilson intervals are checked against scipy's.
"""

import math

from scipy.optimize import brentq
from scipy.stats import beta, norm

ALPHA = 0.05


def exact_conditional(a: int, b: int, c: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """McNemar-style: a Clopper-Pearson interval for b's share of the d = b + c discordant examples,
    mapped to (2π − 1) d / n with d held fixed. At d = 0 it is the single point 0."""
    d = b + c
    if d == 0:
        return 0.0, 0.0
    low = beta.ppf(alpha / 2, b, d - b + 1) if b > 0 else 0.0
    high = beta.ppf(1 - alpha / 2, b + 1, d - b) if b < d else 1.0
    return float((2 * low - 1) * d / n), float((2 * high - 1) * d / n)


def _tango_statistic(delta0: float, b: int, c: int, n: int) -> float:
    big_a = 2 * n
    big_b = -b - c + (2 * n - b + c) * delta0
    big_c = -c * delta0 * (1 - delta0)
    p21 = (math.sqrt(big_b**2 - 4 * big_a * big_c) - big_b) / (2 * big_a)
    variance = n * (2 * p21 + delta0 * (1 - delta0))
    numerator = b - c - n * delta0
    if variance <= 0:
        # Only at delta0 = 0 with b = c = 0, where the numerator is 0 too and the statistic's limit is 0.
        return 0.0
    return numerator / math.sqrt(variance)


def tango(a: int, b: int, c: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Tango's asymptotic score interval (Tango 1998). The bounds are where the score statistic,
    with the cell probabilities fitted under each candidate difference, equals ∓z."""
    z = norm.ppf(1 - alpha / 2)
    estimate = (b - c) / n
    tol = 1e-7
    lo_end, hi_end = -1 + tol, 1 - tol
    low = -1.0 if estimate == -1 else brentq(lambda d: _tango_statistic(d, b, c, n) - z, lo_end, hi_end, xtol=1e-12)
    high = 1.0 if estimate == 1 else brentq(lambda d: _tango_statistic(d, b, c, n) + z, lo_end, hi_end, xtol=1e-12)
    return float(low), float(high)


def wilson(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """The Wilson score interval for k of n, without continuity correction."""
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson_cc(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """The Wilson score interval for k of n with continuity correction."""
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    if k == 0:
        low = 0.0
    else:
        low = (2 * n * p + z * z - 1 - z * math.sqrt(z * z - 2 - 1 / n + 4 * p * (n * (1 - p) + 1))) / (2 * (n + z * z))
    if k == n:
        high = 1.0
    else:
        high = (2 * n * p + z * z + 1 + z * math.sqrt(z * z + 2 - 1 / n + 4 * p * (n * (1 - p) - 1))) / (
            2 * (n + z * z)
        )
    return max(0.0, low), min(1.0, high)


def newcombe(a: int, b: int, c: int, n: int, alpha: float = ALPHA, score=None) -> tuple[float, float]:
    """Newcombe's square-and-add interval for paired data (Newcombe 1998), from the two runs' Wilson
    intervals and a correction for their correlation."""
    score = score or wilson
    d = n - a - b - c
    candidate, reference = a + b, a + c
    p1, p2 = candidate / n, reference / n
    l1, u1 = score(candidate, n, alpha)
    l2, u2 = score(reference, n, alpha)
    if 0 in (candidate, n - candidate, reference, n - reference):
        psi = 0.0
    else:
        product = candidate * (n - candidate) * reference * (n - reference)
        cross = a * d - b * c
        if cross > n / 2:
            psi = (cross - n / 2) / math.sqrt(product)
        elif cross >= 0:
            psi = 0.0
        else:
            psi = cross / math.sqrt(product)
    estimate = p1 - p2
    low = estimate - math.sqrt(max(0.0, (p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * psi * (p1 - l1) * (u2 - p2)))
    high = estimate + math.sqrt(max(0.0, (p2 - l2) ** 2 + (u1 - p1) ** 2 - 2 * psi * (p2 - l2) * (u1 - p1)))
    return float(low), float(high)


def newcombe_cc(a: int, b: int, c: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Newcombe's square-and-add interval built on continuity-corrected Wilson intervals, with the same
    correlation correction. Added for the second coverage run, after the first run's cells were seen."""
    return newcombe(a, b, c, n, alpha, score=wilson_cc)


# The first coverage run's candidates.
INTERVALS = {"exact_conditional": exact_conditional, "tango": tango, "newcombe": newcombe}
# Every candidate, including the second run's.
CANDIDATES = {**INTERVALS, "newcombe_cc": newcombe_cc}
