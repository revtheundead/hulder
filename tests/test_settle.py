"""How many more prompts would decide an undecided condition (hulder.gate.settle)."""

import numpy as np
import pytest

from hulder.gate.check import Estimate
from hulder.gate.settle import auroc_settle_chance, extra_to_settle, rate_settle_chance, scale_to_settle_auroc
from hulder.gate.stats import clopper_pearson

LINE = 0.02  # FPR_BAND_HIGH x a 1% target


def decided(k, n, line=LINE):
    low, high = clopper_pearson(k, n)
    return high <= line or low > line


def test_an_already_decided_rate_needs_nothing():
    assert decided(36, 1191)  # 3.0% [2.1, 4.2]
    assert extra_to_settle(36, 1191, LINE) == 0
    assert rate_settle_chance(36, 1191, 0, LINE) == 1.0


def test_an_undecided_rate_with_no_extra_has_no_chance():
    assert not decided(19, 1191)  # 1.6% [1.0, 2.5]
    assert rate_settle_chance(19, 1191, 0, LINE) == 0.0


def test_the_estimate_is_the_smallest_extra_reaching_the_chance():
    k, n = 30, 1191  # 2.5% [1.7, 3.6], above the line
    extra = extra_to_settle(k, n, LINE)
    assert extra is not None and extra > 0
    assert rate_settle_chance(k, n, extra, LINE) >= 0.8
    assert rate_settle_chance(k, n, extra // 2, LINE) < 0.8


def test_the_chance_matches_simulation():
    k, n, extra = 19, 1191, 4000
    rng = np.random.default_rng(0)
    draws = rng.binomial(extra, k / n, 20_000)
    simulated = np.mean([decided(k + x, n + extra) for x in draws])
    assert rate_settle_chance(k, n, extra, LINE) == pytest.approx(simulated, abs=0.02)


def test_a_rate_on_the_line_is_beyond_the_limit():
    assert extra_to_settle(24, 1200, LINE, max_n=200_000) is None  # exactly 2%


def test_a_zero_rate_decides_toward_the_pass_side():
    assert not decided(0, 100)  # 0% [0, 3.6]
    extra = extra_to_settle(0, 100, LINE)
    assert extra is not None and decided(0, 100 + extra)


def test_the_auroc_chance_is_zero_or_one_without_extra_data():
    undecided = Estimate(-0.015, -0.025, -0.005)
    assert auroc_settle_chance(undecided, -0.02, 1.0) == 0.0
    assert auroc_settle_chance(Estimate(-0.005, -0.015, 0.005), -0.02, 1.0) == 1.0


def test_the_auroc_scale_reaches_the_chance_and_matches_simulation():
    delta = Estimate(-0.012, -0.022, -0.002)
    scale = scale_to_settle_auroc(delta, -0.02)
    assert scale is not None and scale > 1
    assert auroc_settle_chance(delta, -0.02, scale) >= 0.8 - 1e-9
    # Simulate: the extension's estimate is normal around the observed value, sd shrinking with its size.
    se = (delta.high - delta.low) / (2 * 1.959964)
    rng = np.random.default_rng(0)
    for s in (1.5, 3.0):
        new = rng.normal(delta.value, se / np.sqrt(s - 1), 50_000)
        final = (delta.value + (s - 1) * new) / s
        half = 1.959964 * se / np.sqrt(s)
        simulated = np.mean((final - half >= -0.02) | (final + half < -0.02))
        assert auroc_settle_chance(delta, -0.02, s) == pytest.approx(simulated, abs=0.01)


def test_an_auroc_already_decided_needs_no_scale():
    assert scale_to_settle_auroc(Estimate(-0.005, -0.015, 0.005), -0.02) == 1.0


def test_an_auroc_on_the_line_is_beyond_the_limit():
    assert scale_to_settle_auroc(Estimate(-0.02, -0.03, -0.01), -0.02) is None
