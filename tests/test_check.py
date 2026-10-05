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


def test_candidate_scores_drifting_up_to_five_times_target_fpr_is_threshold_stale():
    # Seed 0 at a true candidate FPR of 5% gives 50 false positives of 1,191,
    # CI [3.1%, 5.5%], whose lower bound is above 2 x target.
    data = unchanged_scorer(seed=0, candidate_fpr=0.05)
    report = hulder.check(
        data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01
    )
    assert report.verdict is hulder.Verdict.THRESHOLD_STALE


def test_candidate_scores_drifting_down_to_near_zero_fpr_is_threshold_stale():
    # Too few false positives also means a stale threshold: attacks slip under it.
    # At 1,191 benign the FPR upper bound is below target / 2 only with at most 1 false
    # positive (CI upper bound 0.47%), so this drifts to a true FPR of 0.01%.
    data = unchanged_scorer(seed=0, candidate_fpr=0.0001)
    report = hulder.check(
        data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01
    )
    assert report.verdict is hulder.Verdict.THRESHOLD_STALE


def test_attacks_slipping_under_the_threshold_is_threshold_stale():
    # Attack logits down by 0.5 raise the FNR from about 12% to about 25% while AUROC
    # falls only about 0.01, so the scorer still separates and the threshold is what failed.
    data = unchanged_scorer(seed=0, attack_shift=-0.5)
    report = hulder.check(
        data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01
    )
    assert report.verdict is hulder.Verdict.THRESHOLD_STALE


def test_scorer_that_no_longer_separates_is_scorer_degraded_even_though_fnr_also_rose():
    # Attack logits down by 2 drop AUROC from about 0.993 to about 0.856. FNR rises too,
    # which alone would be THRESHOLD STALE, but no threshold fixes a scorer that no longer
    # separates, so SCORER DEGRADED is checked first.
    data = unchanged_scorer(seed=0, attack_shift=-2.0)
    report = hulder.check(
        data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01
    )
    assert report.verdict is hulder.Verdict.SCORER_DEGRADED
