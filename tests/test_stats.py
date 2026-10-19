import numpy as np
import pytest
from scipy.stats import binomtest

import hulder


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
