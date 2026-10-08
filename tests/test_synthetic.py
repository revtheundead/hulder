"""The synthetic generator's degradation sizes match a large-n run of the generator itself."""

import pytest
from synthetic import blur_for, missed_attack_shift_for, unchanged_scorer

from hulder.gate.check import _auroc

N_LARGE = 1_000_000


def large_n_delta_auroc(**kwargs) -> float:
    d = unchanged_scorer(seed=0, n_benign=N_LARGE, n_attack=N_LARGE, **kwargs)
    return _auroc(d.candidate, d.labels) - _auroc(d.reference, d.labels)


@pytest.mark.parametrize("delta", [-0.01, -0.02, -0.05, -0.1])
def test_missed_attack_shift_gives_the_requested_population_delta_auroc(delta):
    assert large_n_delta_auroc(missed_attack_shift=missed_attack_shift_for(delta)) == pytest.approx(delta, abs=1e-3)


@pytest.mark.parametrize("delta", [-0.01, -0.02, -0.05, -0.1])
def test_blur_gives_the_requested_population_delta_auroc(delta):
    assert large_n_delta_auroc(noise=blur_for(delta)) == pytest.approx(delta, abs=1e-3)


def test_missed_attack_shift_cannot_reach_below_its_floor():
    # Only attacks under the threshold (about 12%) move, so AUROC can fall by at most ~0.1145.
    with pytest.raises(ValueError):
        missed_attack_shift_for(-0.12)
