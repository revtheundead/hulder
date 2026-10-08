import numpy as np
import pytest
from synthetic import unchanged_scorer

import hulder


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


def test_candidate_fpr_whose_ci_straddles_the_band_is_inconclusive():
    # Seed 0 at a true candidate FPR of 2.5% gives 22 false positives of 1,191,
    # CI [1.2%, 2.8%]: neither below 2 x target (PASS) nor above it (STALE).
    data = unchanged_scorer(seed=0, candidate_fpr=0.025)
    report = hulder.check(
        data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01
    )
    assert report.verdict is hulder.Verdict.INCONCLUSIVE


def test_candidate_fnr_whose_ci_straddles_the_band_is_inconclusive():
    # Seed 0, attack logits down by 0.3: reference FNR 13.3%, candidate 19.9%,
    # CI [17.3%, 22.6%] around the 18.3% line. ΔAUROC CI is about [-0.006, -0.004] and
    # the FPR is unchanged, so only the FNR condition is undecided.
    data = unchanged_scorer(seed=0, attack_shift=-0.3)
    report = hulder.check(
        data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01
    )
    assert report.verdict is hulder.Verdict.INCONCLUSIVE


def test_auroc_drop_whose_ci_straddles_the_band_is_inconclusive():
    # Seed 0, attacks already below the threshold pushed down by 1 logit: FPR and FNR are
    # unchanged, and the ΔAUROC CI, about [-0.027, -0.018], straddles -0.02.
    data = unchanged_scorer(seed=0, missed_attack_shift=-1.0)
    report = hulder.check(
        data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01
    )
    assert report.verdict is hulder.Verdict.INCONCLUSIVE
    # The FPR condition is decided, so more benign examples wouldn't help.
    assert report.more_benign_needed == 0


def _benign_then_attack(n_benign, k_flagged, n_attack=910, k_missed=120):
    """Scores with exactly k_flagged benign above 0.5 and k_missed attacks below it."""
    benign = np.r_[np.full(k_flagged, 0.6), np.linspace(0.01, 0.4, n_benign - k_flagged)]
    attack = np.r_[np.full(k_missed, 0.45), np.linspace(0.7, 0.99, n_attack - k_missed)]
    return np.r_[benign, attack], np.r_[np.zeros(n_benign, int), np.ones(n_attack, int)]


def test_inconclusive_reports_how_many_more_benign_examples_would_resolve_it():
    # 22 of 1,191 benign flagged (1.85%, CI [1.2%, 2.8%]) against a 2% line.
    reference, labels = _benign_then_attack(1191, 11)
    candidate, _ = _benign_then_attack(1191, 22)
    report = hulder.check(reference, candidate, labels, threshold=0.5, target_fpr=0.01)
    assert report.verdict is hulder.Verdict.INCONCLUSIVE
    more = report.more_benign_needed
    assert more > 0

    # With that many more benign at the same observed FPRs, the verdict is decided.
    n = 1191 + more
    reference, labels = _benign_then_attack(n, round(11 / 1191 * n))
    candidate, _ = _benign_then_attack(n, round(22 / 1191 * n))
    report = hulder.check(reference, candidate, labels, threshold=0.5, target_fpr=0.01)
    assert report.verdict is not hulder.Verdict.INCONCLUSIVE


def test_report_gives_the_share_of_examples_whose_decision_flipped():
    # Reference flags benign 0-10 and the candidate benign 0-21, attacks unchanged:
    # benign 11-21 flip, 11 of 2,101 examples.
    reference, labels = _benign_then_attack(1191, 11)
    candidate, _ = _benign_then_attack(1191, 22)
    report = hulder.check(reference, candidate, labels, threshold=0.5, target_fpr=0.01)
    assert report.decision_flip_rate == pytest.approx(11 / 2101)


def test_without_a_noise_floor_the_shift_is_reported_and_the_floor_is_not_measured():
    reference, labels = _benign_then_attack(1191, 11)
    candidate = reference + 0.01  # no score crosses 0.5
    report = hulder.check(reference, candidate, labels, threshold=0.5, target_fpr=0.01)
    assert report.mean_abs_shift == pytest.approx(0.01)
    assert report.decision_flips == 0
    assert report.noise_floor_mean_abs_shift is None
    assert report.noise_floor_decision_flips is None
    assert report.shift_exceeds_noise_floor is None


def test_noise_floor_rerun_is_reported_side_by_side_and_never_changes_the_verdict():
    reference, labels = _benign_then_attack(1191, 11)
    # Benign 11 sits at 0.01. The candidate lifts benign 11-47 to 0.6 (37 flips) and
    # the rerun lifts benign 11 alone (1 flip, mean |shift| 0.59 / 2,101).
    candidate, rerun = reference.copy(), reference.copy()
    candidate[11:48] = 0.6
    rerun[11] = 0.6
    without = hulder.check(reference, candidate, labels, threshold=0.5, target_fpr=0.01)
    report = hulder.check(reference, candidate, labels, threshold=0.5, target_fpr=0.01, noise_floor=rerun)
    assert (report.decision_flips, report.noise_floor_decision_flips) == (37, 1)
    assert report.noise_floor_mean_abs_shift == pytest.approx(0.59 / 2101)
    assert report.shift_exceeds_noise_floor is True
    assert report.verdict is without.verdict
