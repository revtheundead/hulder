"""Recalibrated thresholds from the candidate's benign scores (docs/design.md §6)."""

import numpy as np
import pytest
from scipy.stats import binom

from hulder.gate.recalibrate import binomial_tail_threshold, conformal_threshold, short_threshold


@pytest.mark.parametrize(
    ("n", "target_fpr", "allowed"),
    [(180, 0.05, 3), (360, 0.05, 10), (1191, 0.01, 6), (299, 0.01, 0)],
)
def test_binomial_tail_allows_the_most_benign_above_that_a_binomial_tail_permits(n, target_fpr, allowed):
    scores = np.random.default_rng(0).normal(size=n)
    fit = binomial_tail_threshold(scores, target_fpr)
    assert fit.benign_above == allowed == int((scores > fit.threshold).sum())
    assert binom.cdf(allowed, n, target_fpr) <= 0.05 < binom.cdf(allowed + 1, n, target_fpr)
    assert fit.benign_needed is None


def test_binomial_tail_refuses_below_the_benign_count_that_can_certify_the_target():
    # (1 - 0.01)**299 = 0.0496 <= 0.05 < 0.99**298, so 299 benign is the fewest that certify 1%.
    fit = binomial_tail_threshold(np.arange(180.0), 0.01)
    assert fit.threshold is None and fit.benign_above is None
    assert fit.benign_needed == 299


def test_binomial_tail_threshold_sits_between_the_last_unflagged_and_first_flagged_benign():
    scores = np.random.default_rng(1).uniform(size=180)
    ordered = np.sort(scores)
    fit = binomial_tail_threshold(scores, 0.05)
    # 3 allowed above: the threshold lies in [4th highest, 3rd highest), so any value there flags the same three.
    assert ordered[-4] <= fit.threshold < ordered[-3]


def test_ties_at_the_top_make_the_threshold_flag_fewer_benign_never_more():
    scores = np.r_[np.zeros(170), np.ones(10)]
    fit = binomial_tail_threshold(scores, 0.05)
    assert fit.threshold == 1.0 and fit.benign_above == 0


def test_the_threshold_is_the_shortest_decimal_that_flags_the_same_benign():
    scores = np.r_[np.linspace(0.0, 0.43, 176), [0.441032871, 0.4523, 0.6, 0.7]]
    fit = binomial_tail_threshold(scores, 0.05)
    assert fit.threshold == 0.45  # in [0.441032871, 0.4523), and printed exactly as passed back


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (0.441032871, 0.4523, 0.45),
        (0.3952117, 0.3952118, 0.3952117),
        (-1.234, -1.2, -1.23),  # rounds towards +inf, staying above lo
        (0.0, 0.5, 0.0),
        (2.5, np.inf, 3.0),
    ],
)
def test_short_threshold_rounds_up_to_the_fewest_digits_inside_the_gap(lo, hi, expected):
    t = short_threshold(lo, hi)
    assert t == expected and lo <= t < hi


def test_short_threshold_falls_back_to_lo_when_no_shorter_decimal_fits():
    lo = 0.1 + 2**-55
    hi = np.nextafter(lo, 1.0)
    assert short_threshold(lo, hi) == lo


def test_conformal_threshold_is_the_ceil_n_plus_one_times_one_minus_target_order_statistic():
    scores = np.random.default_rng(2).normal(size=180)
    t = conformal_threshold(scores, 0.05)
    # ceil(181 * 0.95) = 172, so the 172nd smallest: 8 benign above it.
    assert t == np.sort(scores)[171] and int((scores > t).sum()) == 8


def test_conformal_threshold_is_none_when_too_few_benign_for_the_target():
    assert conformal_threshold(np.arange(10.0), 0.05) is None  # ceil(11 * 0.95) = 11 > 10


@pytest.mark.parametrize("target_fpr", [0.0, 0.6, float("nan")])
def test_a_target_outside_the_gate_range_is_refused(target_fpr):
    with pytest.raises(ValueError, match="target_fpr"):
        binomial_tail_threshold(np.arange(100.0), target_fpr)


def test_no_benign_scores_is_refused():
    with pytest.raises(ValueError, match="benign"):
        binomial_tail_threshold(np.array([]), 0.05)
