import pytest

import hulder
from synthetic import unchanged_scorer


def test_unchanged_scorer_at_realistic_size_passes():
    # About 1,150 benign at a 1% target, threshold fitted on the reference: an unchanged
    # scorer's FPR CI is roughly [0.5%, 1.7%], which the one-sided PASS band
    # (upper bound <= 2%) must accept.
    data = unchanged_scorer(seed=0)
    report = hulder.check(
        data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01
    )
    assert report.verdict is hulder.Verdict.PASS


@pytest.mark.parametrize("fitted_fpr", [0.03, 0.002])
def test_threshold_not_calibrated_on_the_reference_snapshot_warns(fitted_fpr):
    # Fitted for 3% (or 0.2%) FPR on 1,191 benign, the reference's FPR CI at the threshold
    # excludes the 1% target, so the threshold can't have come from this snapshot.
    data = unchanged_scorer(seed=0, target_fpr=fitted_fpr)
    report = hulder.check(
        data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01
    )
    [warning] = report.warnings
    assert "wasn't calibrated on this snapshot" in warning
    assert "recalibrate on the reference first" in warning


def test_threshold_calibrated_on_the_reference_snapshot_gives_no_warning():
    data = unchanged_scorer(seed=0)
    report = hulder.check(
        data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01
    )
    assert report.warnings == ()
