"""How many more prompts would settle a condition an INCONCLUSIVE report left undecided.

A condition is settled when its 95% bound clears its line on one side, as in the verdict rule
(docs/design.md §6). The chance that more data settles it is averaged over the true values the
current data still allows, not computed at the observed value. So a value close to its line, where
the truth could sit on it, gets a lower chance and asks for more data.

Each estimate looks for the smallest extension, up to GROWTH times the current data, that gives at
least an 80% chance. If none does, it reports the chance at that limit. ΔAUROC also stops at what
the bootstrap fits in memory. The estimates are checked against fresh synthetic data in
docs/results/phase2/settle_v2_expected.json.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import betabinom, norm

from hulder.gate.stats import clopper_pearson

if TYPE_CHECKING:
    from hulder.gate.check import Estimate

CHANCE = 0.8
GROWTH = 10  # the most data, as a multiple of the current, that counts as a practical extension
# The ΔAUROC bootstrap holds 1,000 resample counts per example, so past this many examples in all
# a check needs more memory than a typical machine has.
AUROC_MAX_EXAMPLES = 40_000
PRIOR = 0.5  # Jeffreys, for rates and the paired attack outcomes
DRAWS = 4000  # posterior draws for the FNR's paired outcomes
_Z = float(norm.ppf(0.975))


@dataclass(frozen=True)
class Settle:
    """An extension and its chance to settle the condition. `reached` when the chance is at least
    CHANCE, otherwise the extension is the most considered: GROWTH times the data, or, when `capped`,
    what ΔAUROC's bootstrap fits in memory."""

    more_benign: int
    more_attack: int
    chance: float
    reached: bool
    capped: bool = False


def minimum_for_pass(line: float) -> int:
    """The fewest examples whose 95% upper bound, with no errors at all, is at most `line`.
    Below it, PASS is out of reach whatever the scorer does."""
    n = max(1, math.ceil(math.log(0.025) / math.log1p(-line)) - 1)
    while clopper_pearson(0, n)[1] > line:
        n += 1
    return n


def _last_true(predicate, lo: int, hi: int) -> int:
    """The largest x in [lo, hi] with predicate(x), for a predicate true then false. lo - 1 if none."""
    if not predicate(lo):
        return lo - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if predicate(mid) else (lo, mid - 1)
    return lo


def _rate_decided(k, n, line) -> bool:
    low, high = clopper_pearson(k, n)
    return high <= line or low > line


def fpr_chance(k: int, n: int, extra: int, line: float) -> float:
    """The chance that `extra` more benign examples settle an FPR of k in n against a fixed line. The
    true rate has a Jeffreys posterior, so the extra flagged count is beta-binomial, and the settled
    counts form two tails: few enough that the upper bound is at most the line, or enough that the
    lower bound is above it."""
    if extra == 0:
        return 1.0 if _rate_decided(k, n, line) else 0.0
    total, a, b = n + extra, k + PRIOR, n - k + PRIOR
    pass_side = _last_true(lambda x: clopper_pearson(k + x, total)[1] <= line, 0, extra)
    not_stale = _last_true(lambda x: clopper_pearson(k + x, total)[0] <= line, 0, extra)
    chance = betabinom.cdf(pass_side, extra, a, b) if pass_side >= 0 else 0.0
    chance += betabinom.sf(not_stale, extra, a, b)
    return float(min(chance, 1.0))


def fnr_chance(cells: tuple[int, int, int, int], extra: int, tolerance: float, seed: int = 0) -> float:
    """The chance that `extra` more attacks settle the FNR condition. `cells` counts the attacks both
    runs miss, only the reference misses, only the candidate misses, and neither misses. The line is
    the reference's FNR + tolerance, re-read on the extended data, so both runs' outcomes are drawn
    together: the four shares from their Dirichlet posterior, then the extra attacks from them."""
    both, ref_only, cand_only, neither = cells
    n = both + ref_only + cand_only + neither
    if extra == 0:
        low, high = clopper_pearson(both + cand_only, n)
        line = (both + ref_only) / n + tolerance
        return 1.0 if high <= line or low > line else 0.0
    rng = np.random.default_rng(seed)
    shares = rng.dirichlet(np.asarray(cells, dtype=float) + PRIOR, DRAWS)
    new = rng.multinomial(extra, shares)
    total = n + extra
    reference = both + ref_only + new[:, 0] + new[:, 1]
    candidate = both + cand_only + new[:, 0] + new[:, 2]
    low, high = clopper_pearson(candidate, total)
    line = reference / total + tolerance
    return float(np.mean((high <= line) | (low > line)))


def auroc_chance(delta: Estimate, line: float, scale: float) -> float:
    """The chance that data grown to `scale` times its size, both classes alike, settles ΔAUROC. The
    bootstrap interval is read as a normal posterior for the true value, the new data's estimate is
    normal around it with variance shrinking in proportion to its size, and the final interval's
    half-width shrinks as 1 / √scale. Settled at or above the line (the PASS side) or wholly below it."""
    if scale <= 1.0:
        return 1.0 if delta.low >= line or delta.high < line else 0.0
    se = (delta.high - delta.low) / (2 * _Z)
    if se == 0:
        return 1.0 if delta.value != line else 0.0
    half = _Z * se / math.sqrt(scale)
    spread = se * math.sqrt((scale - 1) / scale)  # sd of the final estimate given the current data
    return float(norm.cdf((delta.value - half - line) / spread) + norm.cdf((line - delta.value - half) / spread))


def _smallest_integer(chance, limit: int) -> tuple[int, float, bool]:
    """The smallest extra in [0, limit] whose chance reaches CHANCE, or limit if none does. The chance
    isn't strictly monotone in a whole-number extension, so this is a planning figure."""
    at_limit = chance(limit)
    if at_limit < CHANCE:
        return limit, at_limit, False
    if chance(0) >= CHANCE:
        return 0, chance(0), True
    hi = 1
    while hi < limit and chance(hi) < CHANCE:
        hi = min(2 * hi, limit)
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if chance(mid) >= CHANCE else (mid, hi)
    return hi, chance(hi), True


def settle_fpr(k: int, n: int, line: float) -> Settle:
    extra, chance, reached = _smallest_integer(lambda e: fpr_chance(k, n, e, line), (GROWTH - 1) * n)
    return Settle(extra, 0, chance, reached)


def settle_fnr(cells: tuple[int, int, int, int], tolerance: float) -> Settle:
    n = sum(cells)
    extra, chance, reached = _smallest_integer(lambda e: fnr_chance(cells, e, tolerance), (GROWTH - 1) * n)
    return Settle(0, extra, chance, reached)


def settle_auroc(delta: Estimate, line: float, n_benign: int, n_attack: int) -> Settle:
    """Both classes grow by the same factor. The factor stops at GROWTH, or earlier at what the
    bootstrap fits in memory, which sets `capped`."""
    n = n_benign + n_attack
    memory = AUROC_MAX_EXAMPLES / n
    limit, capped = (GROWTH, False) if memory >= GROWTH else (max(memory, 1.0), True)

    def extension(scale: float) -> tuple[int, int]:
        return math.floor((scale - 1) * n_benign), math.floor((scale - 1) * n_attack)

    def at(scale: float) -> float:
        b, a = extension(scale)
        return auroc_chance(delta, line, (n + a + b) / n)

    if at(1.0) >= CHANCE:
        return Settle(0, 0, at(1.0), True)
    if at(limit) < CHANCE:
        return Settle(*extension(limit), at(limit), False, capped)
    lo, hi = 1.0, float(limit)
    while hi / lo > 1.0005:
        mid = math.sqrt(lo * hi)
        lo, hi = (lo, mid) if at(mid) >= CHANCE else (mid, hi)
    return Settle(*extension(hi), at(hi), True)
