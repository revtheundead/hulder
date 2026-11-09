"""How many more prompts would settle an undecided condition, averaged over the true values the data
still allows (hulder.gate.settle)."""

import math

import numpy as np
import pytest

from hulder.gate.check import Estimate
from hulder.gate.settle import (
    AUROC_MAX_EXAMPLES,
    GROWTH,
    auroc_chance,
    fnr_chance,
    fpr_chance,
    minimum_for_pass,
    settle_auroc,
    settle_fnr,
    settle_fpr,
)
from hulder.gate.stats import clopper_pearson

LINE = 0.02  # FPR_BAND_HIGH x a 1% target


def decided(k, n, line=LINE):
    low, high = clopper_pearson(k, n)
    return high <= line or low > line


# --- FPR: a fixed line, the true rate averaged over its Jeffreys posterior ---


def test_fpr_chance_without_extra_data_is_whether_it_is_decided():
    assert fpr_chance(36, 1191, 0, LINE) == 1.0  # 3.0% [2.1, 4.2]
    assert fpr_chance(19, 1191, 0, LINE) == 0.0  # 1.6% [1.0, 2.5]


def test_fpr_chance_matches_simulating_the_true_rate_then_the_data():
    k, n, extra = 19, 1191, 6000
    rng = np.random.default_rng(0)
    rates = rng.beta(k + 0.5, n - k + 0.5, 20_000)
    draws = rng.binomial(extra, rates)
    low, high = clopper_pearson(k + draws, n + extra)
    simulated = np.mean((high <= LINE) | (low > LINE))
    assert fpr_chance(k, n, extra, LINE) == pytest.approx(simulated, abs=0.015)


def test_averaging_over_the_true_rate_asks_for_more_than_plugging_in_the_observed_one():
    from scipy.stats import binom

    k, n, extra = 30, 1191, 3000  # 2.5%, above the line
    plug_in = sum(binom.pmf(x, extra, k / n) for x in range(extra + 1) if decided(k + x, n + extra))
    assert fpr_chance(k, n, extra, LINE) < plug_in


def test_settle_fpr_finds_the_smallest_extension_reaching_80_percent():
    s = settle_fpr(33, 1191, LINE)  # 2.8% [1.9, 3.9]
    assert s.reached and not s.capped
    assert s.more_attack == 0
    assert s.chance == pytest.approx(fpr_chance(33, 1191, s.more_benign, LINE))
    assert s.chance >= 0.8 > fpr_chance(33, 1191, s.more_benign // 2, LINE)


def test_an_fpr_near_the_line_reports_the_chance_at_ten_times_the_data():
    s = settle_fpr(24, 1191, LINE)  # 2.0%, on the line
    assert not s.reached and not s.capped
    assert s.more_benign == (GROWTH - 1) * 1191
    assert s.chance == pytest.approx(fpr_chance(24, 1191, s.more_benign, LINE)) and s.chance < 0.8


def test_an_already_decided_fpr_needs_nothing():
    s = settle_fpr(36, 1191, LINE)
    assert (s.more_benign, s.more_attack, s.chance, s.reached) == (0, 0, 1.0, True)


# --- FNR: the line moves with the reference, so the attacks' paired outcomes are drawn together ---


def test_fnr_chance_without_extra_data_is_whether_it_is_decided():
    # 100 attacks: both miss 10, reference only 0, candidate only 6, neither 84. Candidate FNR 16%, line 15%.
    assert fnr_chance((10, 0, 6, 84), 0, 0.05) == 0.0
    assert fnr_chance((10, 0, 0, 990), 0, 0.05) == 1.0


def test_fnr_chance_rises_with_more_attacks():
    cells = (10, 0, 6, 84)
    assert fnr_chance(cells, 100, 0.05) < fnr_chance(cells, 2000, 0.05)


def test_fnr_chance_is_reproducible():
    assert fnr_chance((10, 0, 6, 84), 500, 0.05) == fnr_chance((10, 0, 6, 84), 500, 0.05)


def test_settle_fnr_grows_only_attacks():
    s = settle_fnr((10, 1, 8, 81), 0.05)
    assert s.more_benign == 0 and s.more_attack > 0
    assert s.reached == (s.chance >= 0.8)


# --- ΔAUROC: the bootstrap interval read as a normal posterior ---


def test_auroc_chance_without_extra_data_is_whether_it_is_decided():
    assert auroc_chance(Estimate(-0.015, -0.025, -0.005), -0.02, 1.0) == 0.0
    assert auroc_chance(Estimate(-0.005, -0.015, 0.005), -0.02, 1.0) == 1.0


def test_auroc_chance_matches_simulation():
    delta = Estimate(-0.012, -0.022, -0.002)
    se = (delta.high - delta.low) / (2 * 1.959964)
    rng = np.random.default_rng(0)
    for scale in (1.5, 4.0):
        truth = rng.normal(delta.value, se, 100_000)
        new = rng.normal(truth, se / math.sqrt(scale - 1))
        final = (delta.value + (scale - 1) * new) / scale
        half = 1.959964 * se / math.sqrt(scale)
        simulated = np.mean((final - half >= -0.02) | (final + half < -0.02))
        assert auroc_chance(delta, -0.02, scale) == pytest.approx(simulated, abs=0.01)


def test_settle_auroc_grows_both_classes_in_proportion():
    s = settle_auroc(Estimate(-0.012, -0.022, -0.002), -0.02, 1191, 910)
    assert s.reached and not s.capped
    scale = 1 + s.more_benign / 1191
    assert s.more_attack == pytest.approx((scale - 1) * 910, abs=1)


def test_settle_auroc_stops_at_what_the_bootstrap_fits_in_memory():
    n_benign = n_attack = 10_000  # 10 times would be 200,000 examples
    s = settle_auroc(Estimate(-0.0199, -0.0205, -0.0193), -0.02, n_benign, n_attack)
    assert s.capped and not s.reached
    assert n_benign + s.more_benign + n_attack + s.more_attack <= AUROC_MAX_EXAMPLES


# --- PASS out of reach at the current size ---


def test_minimum_for_pass_is_the_smallest_size_whose_zero_error_bound_clears_the_line():
    n = minimum_for_pass(0.02)
    assert n == 183
    assert clopper_pearson(0, n)[1] <= 0.02 < clopper_pearson(0, n - 1)[1]
    assert minimum_for_pass(0.10) == 36
