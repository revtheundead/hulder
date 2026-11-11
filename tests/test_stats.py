import numpy as np
import pytest
from scipy.stats import binomtest

import hulder
from hulder.gate.stats import paired_change_interval, wilson_cc


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        # R's binom.test(1, 4)$conf.int
        (1, 4, 0.006309463, 0.805879550),
        # none of 10: the upper bound is 1 - 0.025 ** (1 / 10)
        (0, 10, 0.0, 0.308497108),
        # all of 10: the mirror image
        (10, 10, 0.691502892, 1.0),
    ],
)
def test_clopper_pearson_gives_the_exact_95_percent_interval(k, n, low, high):
    assert hulder.clopper_pearson(k, n) == pytest.approx((low, high), abs=1e-9)


def test_clopper_pearson_matches_scipys_exact_interval_on_integer_counts():
    for k, n in [(0, 1), (3, 7), (12, 1191), (2041, 2042)]:
        ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
        assert hulder.clopper_pearson(k, n) == pytest.approx((ci.low, ci.high), rel=1e-12, abs=1e-15)


def test_clopper_pearson_takes_another_alpha():
    ci = binomtest(3, 40).proportion_ci(confidence_level=0.99, method="exact")
    assert hulder.clopper_pearson(3, 40, alpha=0.01) == pytest.approx((ci.low, ci.high), rel=1e-12)


def test_clopper_pearson_broadcasts_over_arrays():
    low, high = hulder.clopper_pearson(np.array([0, 1, 4]), np.array([4, 4, 4]))
    assert low == pytest.approx([0.0, 0.006309463, 0.397635364], abs=1e-9)
    assert high == pytest.approx([0.602364636, 0.805879550, 1.0], abs=1e-9)


@pytest.mark.parametrize(("k", "n"), [(-1, 4), (5, 4), (0, 0)])
def test_clopper_pearson_rejects_impossible_counts(k, n):
    with pytest.raises(ValueError):
        hulder.clopper_pearson(k, n)


# Values from experiments/phase2/paired_intervals.py's newcombe_cc, the interval its coverage run
# measured (docs/results/phase2/paired_coverage_cc.md). Tables are (both, candidate only, reference only, n).
NEWCOMBE_CC = [
    ((9, 0, 2, 1191), (-0.006577571796410213, 0.00254754463472112)),
    ((138, 6, 9, 910), (-0.012619925867500667, 0.005911063750544749)),
    ((0, 0, 0, 180), (-0.026044898336393053, 0.026044898336393053)),
    ((5, 3, 0, 180), (-0.011632712816538961, 0.05105019246230902)),
    ((20, 1, 1, 180), (-0.02601389767110045, 0.02601389767110045)),
    ((0, 180, 0, 180), (0.9631669515420445, 1.0)),
    ((170, 0, 10, 180), (-0.10267589192181004, -0.017985479684738133)),
]


@pytest.mark.parametrize("table,expected", NEWCOMBE_CC)
def test_paired_change_interval_matches_the_measured_interval(table, expected):
    assert paired_change_interval(*table) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("table,_", NEWCOMBE_CC)
def test_paired_change_interval_contains_the_change_and_negates_when_runs_swap(table, _):
    both, up, down, n = table
    low, high = paired_change_interval(both, up, down, n)
    assert low <= (up - down) / n <= high
    assert paired_change_interval(both, down, up, n) == pytest.approx((-high, -low), abs=1e-12)


@pytest.mark.parametrize("k,n", [(0, 180), (1, 180), (11, 1191), (179, 180), (180, 180)])
def test_wilson_cc_matches_scipy(k, n):
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilsoncc")
    assert wilson_cc(k, n) == pytest.approx((ci.low, ci.high), abs=1e-12)


@pytest.mark.parametrize("table", [(-1, 0, 0, 10), (0, 0, 0, 0), (5, 4, 4, 10)])
def test_paired_change_interval_rejects_impossible_tables(table):
    with pytest.raises(ValueError):
        paired_change_interval(*table)
