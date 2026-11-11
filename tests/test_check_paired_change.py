"""The paired change: an informational interval on the FPR and FNR change over the same examples.
It must never affect the verdict, exit code, floors or out-of-reach checks (docs/design.md §10)."""

import dataclasses

import numpy as np
import pytest
from synthetic import unchanged_scorer

import hulder
from hulder.gate import check as check_module
from hulder.gate.check import PairedChange
from hulder.gate.stats import paired_change_interval


def scores(reference_flags, candidate_flags):
    """Scores at threshold 0.5: 0.9 is flagged, 0.1 isn't."""
    return np.where(reference_flags, 0.9, 0.1), np.where(candidate_flags, 0.9, 0.1)


def test_counts_each_way_for_fpr_and_fnr():
    labels = np.array([0] * 6 + [1] * 5)
    # Benign: both flag 1, candidate only 2, reference only 1, neither 2.
    # Attacks, flagged means caught: both catch 2, candidate only catches 1, reference only 1, neither 1.
    ref_flags = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0], bool)
    cand_flags = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0], bool)
    reference, candidate = scores(ref_flags, cand_flags)
    report = hulder.check(reference, candidate, labels, threshold=0.5, target_fpr=0.05)
    fpr, fnr = report.paired_changes
    assert (fpr.condition, fpr.both, fpr.increase, fpr.decrease, fpr.n) == ("fpr", 1, 2, 1, 6)
    # For FNR the counted event is a miss: both miss 1, newly missed 1, newly caught 1.
    assert (fnr.condition, fnr.both, fnr.increase, fnr.decrease, fnr.n) == ("fnr", 1, 1, 1, 5)
    assert fpr.estimate.value == pytest.approx(1 / 6)
    assert fpr.estimate.value == pytest.approx(report.candidate_fpr.value - report.reference_fpr.value)
    assert fnr.estimate.value == pytest.approx(report.candidate_fnr.value - report.reference_fnr.value)
    assert (fpr.estimate.low, fpr.estimate.high) == paired_change_interval(1, 2, 1, 6)
    assert (fnr.estimate.low, fnr.estimate.high) == paired_change_interval(1, 1, 1, 5)


def test_flagged_means_strictly_above_the_threshold():
    labels = np.array([0, 0, 1, 1])
    reference = np.array([0.5, 0.1, 0.9, 0.5])
    candidate = np.array([0.6, 0.1, 0.9, 0.6])
    fpr, fnr = hulder.check(reference, candidate, labels, threshold=0.5, target_fpr=0.05).paired_changes
    assert (fpr.increase, fpr.decrease) == (1, 0)
    assert (fnr.increase, fnr.decrease) == (0, 1)


def test_every_verdict_carries_both_changes():
    for kwargs in ({}, {"candidate_fpr": 0.05}, {"candidate_fpr": 0.015}):
        d = unchanged_scorer(0, **kwargs)
        report = hulder.check(d.reference, d.candidate, d.labels, threshold=d.threshold, target_fpr=0.01)
        assert [c.condition for c in report.paired_changes] == ["fpr", "fnr"]
        assert all(isinstance(c, PairedChange) for c in report.paired_changes)


CASES = [
    {},
    {"candidate_fpr": 0.015},
    {"candidate_fpr": 0.05},
    {"attack_shift": -1.0},
    {"attack_shift": -0.3, "n_benign": 180, "n_attack": 180, "target_fpr": 0.05},
]


@pytest.mark.parametrize("kwargs", CASES)
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("fake", [(-1.0, 1.0), (0.0, 0.0), (0.5, 0.9)])
def test_the_interval_never_reaches_the_verdict(monkeypatch, kwargs, seed, fake):
    """Replacing the interval with anything at all changes only paired_changes."""
    d = unchanged_scorer(seed, **kwargs)
    target = kwargs.get("target_fpr", 0.01)

    def run():
        return hulder.check(d.reference, d.candidate, d.labels, threshold=d.threshold, target_fpr=target)

    real = run()
    monkeypatch.setattr(check_module, "paired_change_interval", lambda *a, **k: fake)
    faked = run()
    assert faked.paired_changes != real.paired_changes
    assert dataclasses.replace(faked, paired_changes=()) == dataclasses.replace(real, paired_changes=())
