"""Exact size facts for an INCONCLUSIVE report, with no estimate of what new data will show.

`minimum_for_pass` is the fewest examples at which PASS is possible at all: below it, even with no
errors, the 95% upper bound stays above the line. The floors are the fewest extra examples that
could settle an undecided condition in its nearer direction, if every new example went the most
favourable way. Fewer cannot settle it, whatever they show. More may not either.

Two estimates of how much data would settle a condition with a stated chance were tested and missed
their bars (docs/results/phase2/settle.md and settle_v2.md), so the report gives only these.
"""

import math
from dataclasses import dataclass

from hulder.gate.stats import clopper_pearson

_LIMIT = 10**8  # extra examples past which a search stops; no floor at practical sizes comes near it


@dataclass(frozen=True)
class Floor:
    """The fewest extra examples that could settle a condition, and the direction they'd settle it:
    "pass" (the bound clears the line on the PASS side) or "stale" (it clears it on the far side)."""

    more: int
    toward: str


def minimum_for_pass(line: float) -> int:
    """The fewest examples whose 95% upper bound, with no errors at all, is at most `line`."""
    n = max(1, math.ceil(math.log(0.025) / math.log1p(-line)) - 1)
    while clopper_pearson(0, n)[1] > line:
        n += 1
    return n


def _first(predicate) -> int:
    """The smallest x >= 1 with predicate(x), for a predicate false then true."""
    hi = 1
    while not predicate(hi):
        if hi >= _LIMIT:
            raise ValueError("no floor below the search limit")
        hi *= 2
    lo = hi // 2  # predicate(lo) is false, or lo == 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if predicate(mid) else (mid, hi)
    return hi


def fpr_floor(k: int, n: int, line: float) -> Floor:
    """For k of n benign flagged, undecided against `line`: the fewer of the extra benign that settle
    it toward PASS if none of them is flagged, and toward THRESHOLD STALE if all are."""
    to_pass = _first(lambda x: clopper_pearson(k, n + x)[1] <= line)
    to_stale = _first(lambda x: clopper_pearson(k + x, n + x)[0] > line)
    return Floor(to_pass, "pass") if to_pass <= to_stale else Floor(to_stale, "stale")


def fnr_floor(candidate_misses: int, reference_misses: int, n: int, tolerance: float) -> Floor:
    """For an FNR undecided against the reference's FNR + tolerance, re-read as attacks are added: the
    fewer of the extra attacks that settle it toward PASS if the candidate catches every one and the
    reference misses every one, raising the line, and toward THRESHOLD STALE if the reverse."""
    c, r = candidate_misses, reference_misses
    to_pass = _first(lambda x: clopper_pearson(c, n + x)[1] <= (r + x) / (n + x) + tolerance)
    to_stale = _first(lambda x: clopper_pearson(c + x, n + x)[0] > r / (n + x) + tolerance)
    return Floor(to_pass, "pass") if to_pass <= to_stale else Floor(to_stale, "stale")
