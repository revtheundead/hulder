"""The recalibration study's ground truth, and its bounds, fixed before running (docs/design.md §6).

Measured results go in docs/results/phase2/recalibration.md, from
experiments/phase2/recalibration.py at 10,000 seeds per cell. These tests use fewer.
"""

import numpy as np
import pytest
from recalibration_study import (
    BOUNDS,
    CASES,
    POPULATIONS,
    PROMOTIONS,
    SIZES,
    TARGETS,
    bounds_hold,
    promotion_round_trip,
    study_cell,
)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("threshold", [0.03, 0.2371, 0.25, 0.5])
def test_exact_rates_match_a_large_sample(case, threshold):
    population = POPULATIONS[case]
    benign, attack = population.sample(np.random.default_rng(0), 2_000_000, 2_000_000)
    fpr, fnr = population.rates(threshold)
    for exact, observed in ((fpr, (benign > threshold).mean()), (fnr, (attack <= threshold).mean())):
        se = np.sqrt(max(exact * (1 - exact), 1e-12) / 2_000_000)
        assert abs(observed - exact) <= 4 * se + 1e-9


def test_rounded_scores_are_heavily_tied_and_float16_scores_lightly():
    benign, _ = POPULATIONS["rounded_2dp"].sample(np.random.default_rng(0), 1191, 0)
    assert len(np.unique(benign)) < 40
    benign, _ = POPULATIONS["float16"].sample(np.random.default_rng(0), 1191, 0)
    assert len(np.unique(benign)) > 1000


def test_every_bound_is_checked():
    cells = {(c, n, t): study_cell(c, n, t, range(20)) for c in CASES[:1] for n in SIZES[:1] for t in TARGETS}
    assert set(bounds_hold(cells, {})) == set(BOUNDS)


@pytest.mark.slow
def test_study_bounds_hold():
    cells = {(c, n, t): study_cell(c, n, t, range(2000)) for c in CASES for n in SIZES for t in TARGETS}
    promotions = {p: promotion_round_trip(*p, range(400)) for p in PROMOTIONS}
    held = bounds_hold(cells, promotions)
    assert held == dict.fromkeys(BOUNDS, True)
