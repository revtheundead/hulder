import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from synthetic import FRESH_ENGINE_NOISE, FRESH_ENGINE_P_ZERO, unchanged_scorer

import hulder


def test_unchanged_scorer_at_realistic_size_passes():
    # About 1,150 benign at a 1% target, threshold fitted on the reference: an unchanged
    # scorer's FPR CI is roughly [0.5%, 1.7%], which the one-sided PASS band
    # (upper bound <= 2%) must accept.
    data = unchanged_scorer(seed=0)
    report = hulder.check(data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01)
    assert report.verdict is hulder.Verdict.PASS


@pytest.mark.parametrize("fitted_fpr", [0.03, 0.002])
def test_threshold_not_calibrated_on_the_reference_snapshot_warns(fitted_fpr):
    # Fitted for 3% (or 0.2%) FPR on 1,191 benign, the reference's FPR CI at the threshold
    # excludes the 1% target, so the threshold can't have come from this snapshot.
    data = unchanged_scorer(seed=0, target_fpr=fitted_fpr)
    report = hulder.check(data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01)
    [warning] = report.warnings
    assert "wasn't calibrated on this snapshot" in warning
    assert "recalibrate on the reference first" in warning


def test_threshold_calibrated_on_the_reference_snapshot_gives_no_warning():
    data = unchanged_scorer(seed=0)
    report = hulder.check(data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01)
    assert report.warnings == ()


def test_candidate_scores_drifting_up_to_five_times_target_fpr_is_threshold_stale():
    # Seed 0 at a true candidate FPR of 5% gives 50 false positives of 1,191,
    # CI [3.1%, 5.5%], whose lower bound is above 2 x target.
    data = unchanged_scorer(seed=0, candidate_fpr=0.05)
    report = hulder.check(data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01)
    assert report.verdict is hulder.Verdict.THRESHOLD_STALE


def test_candidate_scores_drifting_down_to_near_zero_fpr_is_threshold_stale():
    # Too few false positives also means a stale threshold: attacks slip under it.
    # At 1,191 benign the FPR upper bound is below target / 2 only with at most 1 false
    # positive (CI upper bound 0.47%), so this drifts to a true FPR of 0.01%.
    data = unchanged_scorer(seed=0, candidate_fpr=0.0001)
    report = hulder.check(data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01)
    assert report.verdict is hulder.Verdict.THRESHOLD_STALE


def test_attacks_slipping_under_the_threshold_is_threshold_stale():
    # Attack logits down by 0.5 raise the FNR from about 12% to about 25% while AUROC
    # falls only about 0.01, so the scorer still separates and the threshold is what failed.
    data = unchanged_scorer(seed=0, attack_shift=-0.5)
    report = hulder.check(data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01)
    assert report.verdict is hulder.Verdict.THRESHOLD_STALE


def test_scorer_that_no_longer_separates_is_scorer_degraded_even_though_fnr_also_rose():
    # Attack logits down by 2 drop AUROC from about 0.993 to about 0.856. FNR rises too,
    # which alone would be THRESHOLD STALE, but no threshold fixes a scorer that no longer
    # separates, so SCORER DEGRADED is checked first.
    data = unchanged_scorer(seed=0, attack_shift=-2.0)
    report = hulder.check(data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01)
    assert report.verdict is hulder.Verdict.SCORER_DEGRADED


def test_candidate_fpr_whose_ci_straddles_the_band_is_inconclusive():
    # Seed 0 at a true candidate FPR of 2.5% gives 22 false positives of 1,191,
    # CI [1.2%, 2.8%]: neither below 2 x target (PASS) nor above it (STALE).
    data = unchanged_scorer(seed=0, candidate_fpr=0.025)
    report = hulder.check(data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01)
    assert report.verdict is hulder.Verdict.INCONCLUSIVE


def test_candidate_fnr_whose_ci_straddles_the_band_is_inconclusive():
    # Seed 0, attack logits down by 0.3: reference FNR 13.3%, candidate 19.9%,
    # CI [17.3%, 22.6%] around the 18.3% line. ΔAUROC CI is about [-0.006, -0.004] and
    # the FPR is unchanged, so only the FNR condition is undecided.
    data = unchanged_scorer(seed=0, attack_shift=-0.3)
    report = hulder.check(data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01)
    assert report.verdict is hulder.Verdict.INCONCLUSIVE


def test_auroc_drop_whose_ci_straddles_the_band_is_inconclusive():
    # Seed 0, attacks already below the threshold pushed down by 1 logit: FPR and FNR are
    # unchanged, and the ΔAUROC CI, about [-0.027, -0.018], straddles -0.02.
    data = unchanged_scorer(seed=0, missed_attack_shift=-1.0)
    report = hulder.check(data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01)
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


def test_a_candidate_no_noisier_than_the_rerun_does_not_exceed_the_noise_floor():
    # Two independent reruns at the fresh-engine noise level: neither is a change.
    data = unchanged_scorer(seed=0, noise=FRESH_ENGINE_NOISE, p_zero=FRESH_ENGINE_P_ZERO)
    rerun = unchanged_scorer(seed=0, noise=FRESH_ENGINE_NOISE, p_zero=FRESH_ENGINE_P_ZERO, noise_seed=1).candidate
    report = hulder.check(
        data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01, noise_floor=rerun
    )
    assert report.shift_exceeds_noise_floor is False


def test_a_rerun_identical_to_the_reference_leaves_the_floor_unmeasured_and_warns():
    # A rerun bit-identical to the reference measures nothing: as a zero floor, any noise
    # in the candidate would "exceed" it.
    reference, labels = _benign_then_attack(1191, 11)
    candidate = reference + 0.01
    report = hulder.check(reference, candidate, labels, threshold=0.5, target_fpr=0.01, noise_floor=reference.copy())
    assert report.noise_floor_mean_abs_shift is None
    assert report.noise_floor_decision_flips is None
    assert report.shift_exceeds_noise_floor is None
    [warning] = report.warnings
    assert "bit-identical to the reference" in warning
    assert "noise floor was not measured" in warning


def _determinism_case():
    # Both bootstraps run: the ΔAUROC one and, with a rerun, the noise floor's. Report
    # carries no CI bounds yet, only the verdict and shift_exceeds_noise_floor that they
    # decide, so an unseeded bootstrap passes these tests unless a draw tips a decision.
    data = unchanged_scorer(seed=3, noise=FRESH_ENGINE_NOISE, p_zero=FRESH_ENGINE_P_ZERO, candidate_fpr=0.015)
    rerun = unchanged_scorer(seed=3, noise=FRESH_ENGINE_NOISE, p_zero=FRESH_ENGINE_P_ZERO, noise_seed=1).candidate
    return hulder.check(
        data.reference,
        data.candidate,
        data.labels,
        threshold=data.threshold,
        target_fpr=0.01,
        seed=7,
        noise_floor=rerun,
    )


def test_same_inputs_and_seed_give_an_identical_report():
    first, second = _determinism_case(), _determinism_case()
    assert first.noise_floor_mean_abs_shift is not None
    assert first == second


def test_same_inputs_and_seed_give_an_identical_report_across_processes():
    # repr round-trips floats exactly, so equal reprs mean equal Reports. Different hash
    # seeds catch any dependence on set or dict ordering.
    code = (
        f"import sys; sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "import test_check\n"
        "print(repr(test_check._determinism_case()))"
    )  # fmt: skip
    reprs = [
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        ).stdout
        for hash_seed in ("1", "2")
    ]
    assert reprs[0] == reprs[1] == repr(_determinism_case()) + "\n"


_META = {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "revision": "989aa79",
    "quantization": None,
    "system_prompt": "You are a helpful assistant.",
    "layer": 20,
    "position": -1,
    "prompt_ids_sha256": "aaa",
}
_AWQ = {**_META, "model": "Qwen/Qwen2.5-1.5B-Instruct-AWQ", "revision": "3ecffa0", "quantization": "awq"}


def _check_with_meta(reference_meta, candidate_meta, **kwargs):
    data = unchanged_scorer(seed=0)
    return hulder.check(
        data.reference,
        data.candidate,
        data.labels,
        threshold=data.threshold,
        target_fpr=0.01,
        reference_meta=reference_meta,
        candidate_meta=candidate_meta,
        **kwargs,
    )


def test_changes_under_test_are_listed_and_not_warned_about():
    # fp16 vs the official AWQ checkpoint: a different repo and revision of the same base model.
    report = _check_with_meta(_META, _AWQ)
    assert report.warnings == ()
    assert report.changes_under_test == (
        ("model", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct-AWQ"),
        ("quantization", None, "awq"),
        ("revision", "989aa79", "3ecffa0"),
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [("base_model", "Qwen/Qwen2.5-7B-Instruct"), ("system_prompt", ""), ("layer", 16), ("team", "x")],
)
def test_a_difference_outside_the_changes_under_test_warns(key, value):
    # Unknown keys like "team" must match too: nobody declared them as under test.
    report = _check_with_meta({**_META, "team": "a"}, {**_AWQ, "team": "a", key: value})
    [warning] = report.warnings
    assert f"{key!r} differs" in warning
    assert key not in [k for k, _, _ in report.changes_under_test]


def test_a_key_declared_as_under_test_is_listed_instead_of_warned_about():
    report = _check_with_meta({**_META, "team": "a"}, {**_AWQ, "team": "b"}, changes_under_test=("team",))
    assert report.warnings == ()
    assert ("team", "a", "b") in report.changes_under_test


def test_a_key_on_one_side_only_counts_as_a_difference():
    report = _check_with_meta({**_META, "chat_template": "t"}, _AWQ)
    [warning] = report.warnings
    assert "'chat_template' differs" in warning


def test_a_missing_base_model_warns_that_it_could_not_be_determined():
    without = {k: v for k, v in _AWQ.items() if k != "base_model"}
    report = _check_with_meta(_META, without)
    [warning] = report.warnings
    assert "base model" in warning
    assert "couldn't be determined" in warning


def test_metadata_on_one_side_only_warns_that_it_was_not_compared():
    report = _check_with_meta(_META, None)
    [warning] = report.warnings
    assert "metadata" in warning
    assert "not compared" in warning


def test_without_metadata_nothing_is_compared_or_warned_about():
    report = _check_with_meta(None, None)
    assert report.warnings == ()
    assert report.changes_under_test == ()


def test_different_prompt_ids_raise_a_pairing_error():
    with pytest.raises(hulder.PairingError, match="prompt_ids_sha256"):
        _check_with_meta(_META, {**_AWQ, "prompt_ids_sha256": "bbb"})
