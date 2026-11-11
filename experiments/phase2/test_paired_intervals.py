"""The candidate paired intervals' properties. CPU only, no data."""

import math

import paired_intervals as p
import pytest
from scipy.stats import binomtest, norm

TABLES = [
    (0, 0, 0, 180),
    (2, 0, 0, 180),
    (9, 1, 0, 180),
    (9, 0, 3, 180),
    (11, 8, 0, 1191),
    (147, 9, 6, 910),
    (5, 4, 4, 50),
]
Z = norm.ppf(0.975)


@pytest.mark.parametrize("name", list(p.INTERVALS))
@pytest.mark.parametrize("table", TABLES)
def test_contains_the_estimate_and_stays_in_range(name, table):
    _a, b, c, n = table
    low, high = p.INTERVALS[name](*table)
    assert -1 <= low <= (b - c) / n + 1e-12 <= high + 2e-12 <= 1 + 2e-12


@pytest.mark.parametrize("name", list(p.INTERVALS))
@pytest.mark.parametrize("table", TABLES)
def test_swapping_the_runs_negates_the_interval(name, table):
    a, b, c, n = table
    low, high = p.INTERVALS[name](a, b, c, n)
    swapped = p.INTERVALS[name](a, c, b, n)
    assert swapped == pytest.approx((-high, -low), abs=1e-7)


def test_exact_conditional_is_clopper_pearson_on_the_discordant_split():
    ci = binomtest(3, 4).proportion_ci(confidence_level=0.95, method="exact")
    assert p.exact_conditional(10, 3, 1, 200) == pytest.approx(
        ((2 * ci.low - 1) * 4 / 200, (2 * ci.high - 1) * 4 / 200)
    )


def test_exact_conditional_has_no_width_without_discordant_examples():
    assert p.exact_conditional(5, 0, 0, 180) == (0.0, 0.0)


@pytest.mark.parametrize("k,n", [(0, 180), (3, 180), (11, 1191), (180, 180)])
def test_wilson_matches_scipy(k, n):
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    assert p.wilson(k, n) == pytest.approx((ci.low, ci.high), abs=1e-12)


@pytest.mark.parametrize("table", [t for t in TABLES if t[1] != t[3] and t[2] != t[3]])
def test_tango_bounds_solve_its_score_equation(table):
    _a, b, c, n = table
    low, high = p.tango(*table)
    assert p._tango_statistic(low, b, c, n) == pytest.approx(Z, abs=1e-6)
    assert p._tango_statistic(high, b, c, n) == pytest.approx(-Z, abs=1e-6)


def test_tango_keeps_width_without_discordant_examples():
    low, high = p.tango(5, 0, 0, 180)
    assert low < 0 < high
    # With no discordant pairs Tango's bound solves n·δ² = z²·n·δ(1 − δ) to first order, so it's about z²/n.
    assert high == pytest.approx(Z**2 / (180 + Z**2), rel=1e-3)


def test_newcombe_without_correlation_is_the_plain_square_and_add():
    # a·d − b·c between 0 and n/2 sets psi to 0, so the interval adds the two Wilson half-widths in quadrature.
    a, b, c, n = 1, 2, 2, 10  # a·d − b·c = 5 − 4 = 1
    l1, u1 = p.wilson(3, 10)
    low, high = p.newcombe(a, b, c, n)
    assert low == pytest.approx(-math.hypot(0.3 - l1, u1 - 0.3))
    assert high == pytest.approx(math.hypot(0.3 - l1, u1 - 0.3))


@pytest.mark.parametrize("k,n", [(0, 180), (1, 180), (3, 180), (11, 1191), (179, 180), (180, 180)])
def test_wilson_cc_matches_scipy(k, n):
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilsoncc")
    assert p.wilson_cc(k, n) == pytest.approx((ci.low, ci.high), abs=1e-12)


@pytest.mark.parametrize("table", TABLES)
def test_newcombe_cc_contains_newcombe(table):
    low, high = p.newcombe(*table)
    cc_low, cc_high = p.newcombe_cc(*table)
    assert cc_low <= low + 1e-12 and high <= cc_high + 1e-12


@pytest.mark.parametrize("table", TABLES)
def test_newcombe_cc_contains_the_estimate_and_negates_on_swap(table):
    a, b, c, n = table
    low, high = p.newcombe_cc(a, b, c, n)
    assert -1 <= low <= (b - c) / n + 1e-12 <= high + 2e-12 <= 1 + 2e-12
    assert p.newcombe_cc(a, c, b, n) == pytest.approx((-high, -low), abs=1e-9)


def test_the_first_runs_candidates_are_unchanged():
    assert list(p.INTERVALS) == ["exact_conditional", "tango", "newcombe"]
    assert list(p.CANDIDATES) == [*p.INTERVALS, "newcombe_cc"]
