"""Exact floors for an undecided condition (hulder.gate.settle): the fewest extra examples that could
settle it, if every new example went the most favourable way. Pinned against Clopper-Pearson."""

from decimal import Decimal

import numpy as np
import pytest

from hulder.gate.settle import Floor, fnr_floor, fpr_floor, minimum_for_pass, minimum_target_for_pass
from hulder.gate.stats import clopper_pearson


def fpr_settled(k, n, line):
    low, high = clopper_pearson(k, n)
    return high <= line or low > line


def fnr_settled(cand_misses, ref_misses, n, tolerance=0.05):
    low, high = clopper_pearson(cand_misses, n)
    line = ref_misses / n + tolerance
    return high <= line or low > line


# --- PASS out of reach ---


def test_minimum_for_pass_is_the_smallest_size_whose_zero_error_bound_clears_the_line():
    n = minimum_for_pass(0.02)
    assert n == 183
    assert clopper_pearson(0, n)[1] <= 0.02 < clopper_pearson(0, n - 1)[1]
    assert minimum_for_pass(0.10) == 36


def _one_step_below(target: float) -> float:
    """The next smaller value with the same number of significant figures, three."""
    d = Decimal(repr(target))
    return float(d - Decimal(1).scaleb(d.adjusted() - 2))


@pytest.mark.parametrize("n", [1, 2, 36, 180, 182, 1000, 10_000, 250_000])
def test_minimum_target_for_pass_is_the_smallest_three_figure_target_whose_line_clears(n):
    target = minimum_target_for_pass(n, 2.0)
    zero_error_bound = clopper_pearson(0, n)[1]
    assert zero_error_bound <= 2.0 * target
    assert zero_error_bound > 2.0 * _one_step_below(target)
    assert len(Decimal(repr(target)).normalize().as_tuple().digits) <= 3


def test_minimum_target_for_pass_at_the_size_checks_example():
    # 180 benign at a 1% target is out of reach until 183 benign; a 1.02% target reaches it at 180.
    assert minimum_target_for_pass(180, 2.0) == 0.0102
    assert minimum_for_pass(2.0 * 0.0102) <= 180 < minimum_for_pass(2.0 * 0.0101)


# --- FPR ---


@pytest.mark.parametrize(("k", "n", "line"), [(13, 180, 0.10), (19, 1191, 0.02), (24, 1200, 0.02), (33, 1191, 0.02)])
def test_fpr_floor_is_the_fewest_extra_benign_that_settle_it_either_way(k, n, line):
    assert not fpr_settled(k, n, line)
    floor = fpr_floor(k, n, line)
    x, toward = floor.more, floor.toward
    # At the floor, the most favourable new examples settle it in the stated direction.
    if toward == "pass":
        assert clopper_pearson(k, n + x)[1] <= line  # all new benign unflagged
    else:
        assert clopper_pearson(k + x, n + x)[0] > line  # all new benign flagged
    # One fewer can't settle it either way, whatever the new examples do.
    flagged = np.arange(x)  # x - 1 new examples, any number of them flagged
    low, high = clopper_pearson(k + flagged, np.full(x, n + x - 1))
    assert not np.any((high <= line) | (low > line))


def test_fpr_floor_takes_whichever_direction_needs_fewer():
    # A few all-flagged examples lift the lower bound faster than many unflagged ones lower the
    # upper bound, so at these sizes the STALE side is usually nearer, even below the line.
    assert fpr_floor(19, 1191, 0.02) == Floor(16, "stale")  # 1.6% [1.0, 2.5]
    # Just under the line on a large set, a few unflagged examples are enough.
    assert fpr_floor(81, 5000, 0.02).toward == "pass"  # 1.6% [1.3, 2.0]


def test_the_quickstarts_check_2_floor():
    floor = fpr_floor(13, 180, 0.10)
    assert floor == Floor(16, "stale")


# --- FNR: the line is the reference's FNR + 0.05, re-read on the extended data ---


@pytest.mark.parametrize(("cand", "ref", "n"), [(29, 20, 180), (24, 15, 180), (140, 100, 910)])
def test_fnr_floor_is_the_fewest_extra_attacks_that_settle_it_either_way(cand, ref, n):
    assert not fnr_settled(cand, ref, n)
    floor = fnr_floor(cand, ref, n, 0.05)
    x = floor.more
    if floor.toward == "pass":
        # Each new attack caught by the candidate and missed by the reference, which raises the line.
        assert fnr_settled(cand, ref + x, n + x)
        assert clopper_pearson(cand, n + x)[1] <= (ref + x) / (n + x) + 0.05
    else:
        # Each new attack missed by the candidate and caught by the reference.
        assert clopper_pearson(cand + x, n + x)[0] > ref / (n + x) + 0.05
    # One fewer can't settle it, whatever both runs do on the new attacks.
    m = x - 1
    c, r = np.meshgrid(np.arange(m + 1), np.arange(m + 1))
    low, high = clopper_pearson(cand + c, np.full(c.shape, n + m))
    line = (ref + r) / (n + m) + 0.05
    assert not np.any((high <= line) | (low > line))


def test_fnr_floor_names_its_direction():
    assert fnr_floor(29, 20, 180, 0.05).toward in ("pass", "stale")
