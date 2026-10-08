"""The gate's own false-alarm and detection rates on synthetic data (README §7).

Bounds are fixed before running. Measured rates go in docs/results/phase1/gate_calibration.md,
from experiments/phase1/gate/calibration.py, and in the verdict rule.
"""

from functools import partial

import pytest
from synthetic import FRESH_ENGINE_NOISE, FRESH_ENGINE_P_ZERO, missed_attack_shift_for, unchanged_scorer, verdict_rates

SEEDS = range(400)

pytestmark = pytest.mark.slow


def test_null_rarely_fails_to_pass():
    rates = verdict_rates(unchanged_scorer, SEEDS)
    assert 1 - rates["PASS"] <= 0.10


def test_fresh_engine_noise_rarely_fails_to_pass():
    rates = verdict_rates(partial(unchanged_scorer, noise=FRESH_ENGINE_NOISE, p_zero=FRESH_ENGINE_P_ZERO), SEEDS)
    assert 1 - rates["PASS"] <= 0.10


def test_candidate_fpr_of_five_times_target_is_mostly_threshold_stale():
    rates = verdict_rates(partial(unchanged_scorer, candidate_fpr=0.05), SEEDS)
    assert rates["THRESHOLD STALE"] >= 0.90


@pytest.mark.parametrize("candidate_fpr", [0.03, 0.05])
def test_candidate_fpr_of_three_times_target_or_more_almost_never_passes(candidate_fpr):
    # Pre-registered (2026-10-05): a gate that waves a tripled escalation volume through
    # is worse than one that can't decide.
    rates = verdict_rates(partial(unchanged_scorer, candidate_fpr=candidate_fpr), SEEDS)
    assert rates["PASS"] <= 0.01


# Pre-registered (2026-10-05, author). missed_attack_shift lowers AUROC without changing
# FPR or FNR, so these measure SCORER DEGRADED on its own. Sizes are population ΔAUROC.


@pytest.mark.parametrize("delta_auroc", [-0.05, -0.1])
def test_auroc_drop_of_0_05_or_more_is_mostly_scorer_degraded_and_almost_never_passes(delta_auroc):
    rates = verdict_rates(partial(unchanged_scorer, missed_attack_shift=missed_attack_shift_for(delta_auroc)), SEEDS)
    assert rates["SCORER DEGRADED"] >= 0.80
    assert rates["PASS"] <= 0.01


def test_auroc_drop_inside_the_tolerance_is_rarely_scorer_degraded():
    rates = verdict_rates(partial(unchanged_scorer, missed_attack_shift=missed_attack_shift_for(-0.01)), SEEDS)
    assert rates["SCORER DEGRADED"] <= 0.05
