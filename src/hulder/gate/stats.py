"""Small statistical helpers shared by the gate and its users."""

import math

import numpy as np
from scipy.stats import beta, norm


def clopper_pearson(k, n, alpha: float = 0.05):
    """The exact (Clopper-Pearson) 1 - alpha interval for a binomial rate of k successes in n
    trials, as (low, high). k and n may be arrays, which broadcast. k may be fractional, as when a
    rate is projected onto another sample size, which the beta quantiles allow. Taking alpha rather
    than a confidence level keeps the default quantiles exactly 0.025 and 0.975."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    if np.any(n <= 0) or np.any(k < 0) or np.any(k > n):
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")
    with np.errstate(all="ignore"):
        low = np.where(k > 0, beta.ppf(alpha / 2, k, n - k + 1), 0.0)
        high = np.where(k < n, beta.ppf(1 - alpha / 2, k + 1, n - k), 1.0)
    if low.ndim == 0:
        return float(low), float(high)
    return low, high


def wilson_cc(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """The Wilson score interval for k of n, with continuity correction."""
    z = float(norm.ppf(1 - alpha / 2))
    p = k / n
    low = 0.0
    if k > 0:
        low = (2 * n * p + z * z - 1 - z * math.sqrt(z * z - 2 - 1 / n + 4 * p * (n * (1 - p) + 1))) / (2 * (n + z * z))
    high = 1.0
    if k < n:
        high = (2 * n * p + z * z + 1 + z * math.sqrt(z * z + 2 - 1 / n + 4 * p * (n * (1 - p) - 1))) / (
            2 * (n + z * z)
        )
    return max(0.0, low), min(1.0, high)


def paired_change_interval(both: int, up: int, down: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """A 1 - alpha interval for the change in a rate between two runs over the same n examples,
    (up - down) / n. `both` examples count for both runs, `up` for the candidate only and `down` for
    the reference only. It is Newcombe's square-and-add interval on continuity-corrected Wilson
    intervals. Informational only: in its coverage check it missed the bar fixed beforehand in 2 of
    1,536 cells, at most 3.68% per end against a nominal 2.5% (docs/results/phase2/paired_coverage_cc.md),
    so nothing in the verdict reads it."""
    if n <= 0 or min(both, up, down) < 0 or both + up + down > n:
        raise ValueError(f"need non-negative counts summing to at most n > 0, got {both}, {up}, {down} of {n}")
    neither = n - both - up - down
    candidate, reference = both + up, both + down
    p1, p2 = candidate / n, reference / n
    l1, u1 = wilson_cc(candidate, n, alpha)
    l2, u2 = wilson_cc(reference, n, alpha)
    psi = 0.0
    if 0 not in (candidate, n - candidate, reference, n - reference):
        cross = both * neither - up * down
        scale = math.sqrt(candidate * (n - candidate) * reference * (n - reference))
        if cross > n / 2:
            psi = (cross - n / 2) / scale
        elif cross < 0:
            psi = cross / scale
    change = p1 - p2
    low = change - math.sqrt(max(0.0, (p1 - l1) ** 2 + (u2 - p2) ** 2 - 2 * psi * (p1 - l1) * (u2 - p2)))
    high = change + math.sqrt(max(0.0, (p2 - l2) ** 2 + (u1 - p1) ** 2 - 2 * psi * (p2 - l2) * (u1 - p1)))
    return float(low), float(high)
