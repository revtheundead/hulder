"""The gate's own false-alarm and detection rates on synthetic data (README §7).

Bounds are fixed before running. Measured rates go in docs/results/phase1/gate_calibration.md,
from experiments/phase1/gate/calibration.py, and in the verdict rule.
"""

from functools import partial

import pytest
from synthetic import FRESH_ENGINE_NOISE, FRESH_ENGINE_P_ZERO, unchanged_scorer, verdict_rates

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
