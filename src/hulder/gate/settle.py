"""How many more prompts would decide a condition an INCONCLUSIVE report left undecided.

Each estimate assumes the observed rates hold and asks for the extra data that decides the condition
with a given chance (80% by default), so it is a planning number, not a guarantee. A condition is
decided when its 95% bound has cleared its line on one side, as in the verdict rule
(docs/design.md §6). The estimates are checked against fresh synthetic data in
docs/results/phase2/settle_expected.json.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.stats import binom, norm

from hulder.gate.stats import clopper_pearson

if TYPE_CHECKING:
    from hulder.gate.check import Estimate

CHANCE = 0.8
MAX_N = 10**6  # examples per class in all, past which a rate's estimate gives up
MAX_SCALE = 1000.0  # times the current data, past which ΔAUROC's estimate gives up
_Z = float(norm.ppf(0.975))


def _rate_decided(k: float, n: float, line: float) -> bool:
    low, high = clopper_pearson(k, n)
    return high <= line or low > line


def _last_true(predicate, lo: int, hi: int) -> int:
    """The largest x in [lo, hi] with predicate(x), for a predicate true then false. lo - 1 if none."""
    if not predicate(lo):
        return lo - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if predicate(mid) else (lo, mid - 1)
    return lo


def rate_settle_chance(k: int, n: int, extra: int, line: float) -> float:
    """The chance that `extra` more examples decide a rate of k in n against `line`, if the observed
    rate k / n holds. The extra flagged count is binomial, and the decided counts form two tails:
    few enough that the upper bound is at most the line, or enough that the lower bound is above it."""
    if extra == 0:
        return 1.0 if _rate_decided(k, n, line) else 0.0
    total, p = n + extra, k / n
    pass_side = _last_true(lambda x: clopper_pearson(k + x, total)[1] <= line, 0, extra)
    not_stale = _last_true(lambda x: clopper_pearson(k + x, total)[0] <= line, 0, extra)
    chance = binom.cdf(pass_side, extra, p) if pass_side >= 0 else 0.0
    chance += binom.sf(not_stale, extra, p)
    return float(min(chance, 1.0))


def extra_to_settle(k: int, n: int, line: float, chance: float = CHANCE, max_n: int = MAX_N) -> int | None:
    """Roughly the fewest extra examples that decide a rate of k in n with `chance`, if it holds.
    0 when it's already decided, None when max_n examples in all wouldn't. The chance isn't strictly
    monotone in the extra count, since counts are whole numbers, so this is a planning figure."""
    if _rate_decided(k, n, line):
        return 0
    limit = max_n - n
    hi = 1
    while rate_settle_chance(k, n, hi, line) < chance:
        if hi >= limit:
            return None
        hi = min(2 * hi, limit)
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if rate_settle_chance(k, n, mid, line) >= chance else (mid, hi)
    return hi


def auroc_settle_chance(delta: Estimate, line: float, scale: float) -> float:
    """The chance that data grown to `scale` times its size, both classes alike, decides ΔAUROC
    against `line`, if the observed value holds. The bootstrap interval is read as normal, and its
    half-width shrinks as 1 / √scale. ΔAUROC is decided at or above the line (the PASS side) or
    wholly below it (SCORER DEGRADED)."""
    if scale <= 1.0:
        return 1.0 if delta.low >= line or delta.high < line else 0.0
    se = (delta.high - delta.low) / (2 * _Z)
    if se == 0:
        return 1.0 if delta.value != line else 0.0
    half = _Z * se / scale**0.5
    spread = se * (scale - 1) ** 0.5 / scale  # sd of the final estimate around the observed one
    return float(norm.cdf((delta.value - half - line) / spread) + norm.cdf((line - delta.value - half) / spread))


def scale_to_settle_auroc(
    delta: Estimate, line: float, chance: float = CHANCE, max_scale: float = MAX_SCALE
) -> float | None:
    """Roughly how many times the current data, both classes alike, decides ΔAUROC with `chance`.
    1.0 when it's already decided, None when max_scale wouldn't."""
    if auroc_settle_chance(delta, line, 1.0) == 1.0:
        return 1.0
    if auroc_settle_chance(delta, line, max_scale) < chance:
        return None
    lo, hi = 1.0, max_scale
    while hi / lo > 1.001:
        mid = (lo * hi) ** 0.5
        lo, hi = (lo, mid) if auroc_settle_chance(delta, line, mid) >= chance else (mid, hi)
    return hi
