"""Small statistical helpers shared by the gate and its users."""

import numpy as np
from scipy.stats import beta


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
