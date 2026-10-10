import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binomtest
from synthetic import FRESH_ENGINE_NOISE, FRESH_ENGINE_P_ZERO, unchanged_scorer

import hulder
import hulder.gate.check as gate_check


def _paired(labels):
    """Example ids for both runs, so tests about other warnings don't also get the pairing one."""
    ids = [f"slabs/{i}" for i in range(len(labels))]
    return {"reference_ids": ids, "candidate_ids": ids}


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
    report = hulder.check(
        data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01, **_paired(data.labels)
    )
    [warning] = report.warnings
    assert "wasn't calibrated on this snapshot" in warning
    assert "recalibrate on the reference first" in warning


def test_threshold_calibrated_on_the_reference_snapshot_gives_no_warning():
    data = unchanged_scorer(seed=0)
    report = hulder.check(
        data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01, **_paired(data.labels)
    )
    assert report.warnings == ()


def test_candidate_scores_shifting_up_to_five_times_target_fpr_is_threshold_stale():
    # Seed 0 at a true candidate FPR of 5% gives 50 false positives of 1,191,
    # CI [3.1%, 5.5%], whose lower bound is above 2 x target.
    data = unchanged_scorer(seed=0, candidate_fpr=0.05)
    report = hulder.check(data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01)
    assert report.verdict is hulder.Verdict.THRESHOLD_STALE


def test_candidate_scores_shifting_down_to_near_zero_fpr_is_threshold_stale():
    # Too few false positives also means a stale threshold: attacks slip under it.
    # At 1,191 benign the FPR upper bound is below target / 2 only with at most 1 false
    # positive (CI upper bound 0.47%), so this shifts to a true FPR of 0.01%.
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
    report = hulder.check(
        reference,
        candidate,
        labels,
        threshold=0.5,
        target_fpr=0.01,
        noise_floor=reference.copy(),
        noise_floor_ids=_paired(labels)["reference_ids"],
        **_paired(labels),
    )
    assert report.noise_floor_mean_abs_shift is None
    assert report.noise_floor_decision_flips is None
    assert report.shift_exceeds_noise_floor is None
    [warning] = report.warnings
    assert "bit-identical to the reference" in warning
    assert "noise floor was not measured" in warning


def test_a_candidate_identical_to_the_reference_warns():
    # The same file passed twice, say by a mistyped CI path, would otherwise PASS silently.
    reference, labels = _benign_then_attack(1191, 11)
    report = hulder.check(reference, reference.copy(), labels, threshold=0.5, target_fpr=0.01, **_paired(labels))
    assert report.verdict is hulder.Verdict.PASS
    [warning] = report.warnings
    assert "candidate is bit-identical to the reference" in warning


def _determinism_case():
    # Both bootstraps run: the ΔAUROC one and, with a rerun, the noise floor's. Report
    # carries both bootstraps' bounds (delta_auroc and shift_over_noise_floor), so
    # unseeding either fails these tests (checked 2026-10-10).
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
    metadata, pairing = sorted(report.warnings, key=lambda w: w.startswith("Pairing"))
    assert "metadata" in metadata
    assert "not compared" in metadata
    assert pairing.startswith("Pairing not verified")


def test_without_metadata_nothing_is_compared_and_only_pairing_is_warned_about():
    report = _check_with_meta(None, None)
    [warning] = report.warnings
    assert warning.startswith("Pairing not verified")
    assert report.changes_under_test == ()


def test_different_prompt_ids_raise_a_pairing_error():
    with pytest.raises(hulder.PairingError, match="prompt_ids_sha256"):
        _check_with_meta(_META, {**_AWQ, "prompt_ids_sha256": "bbb"})


def _check_with_ids(reference_ids, candidate_ids, reference_meta=None, candidate_meta=None):
    data = unchanged_scorer(seed=0)
    return hulder.check(
        data.reference,
        data.candidate,
        data.labels,
        threshold=data.threshold,
        target_fpr=0.01,
        reference_meta=reference_meta,
        candidate_meta=candidate_meta,
        reference_ids=reference_ids,
        candidate_ids=candidate_ids,
    )


def test_example_ids_in_the_same_order_verify_the_pairing():
    ids = [f"slabs/{i}" for i in range(len(unchanged_scorer(seed=0).labels))]
    assert _check_with_ids(ids, list(ids)).warnings == ()


def test_example_ids_in_a_different_order_raise_a_pairing_error():
    ids = [f"slabs/{i}" for i in range(len(unchanged_scorer(seed=0).labels))]
    with pytest.raises(hulder.PairingError, match="not run on the same prompts in the same order"):
        _check_with_ids(ids, ids[1:] + ids[:1])


def test_ids_on_one_side_are_checked_against_the_hash_in_the_others_metadata():
    ids = [f"slabs/{i}" for i in range(len(unchanged_scorer(seed=0).labels))]
    meta = {"base_model": "m", "prompt_ids_sha256": hulder.prompt_ids_sha256(ids)}
    assert _check_with_ids(None, ids, reference_meta=meta, candidate_meta={"base_model": "m"}).warnings == ()
    with pytest.raises(hulder.PairingError):
        _check_with_ids(None, ids[::-1], reference_meta=meta, candidate_meta={"base_model": "m"})


def test_without_ids_or_hashes_the_report_warns_that_pairing_was_not_verified():
    [warning] = _check_with_ids(None, None).warnings
    assert "pairing not verified; scores assumed to be in the same order" in warning.lower()


def test_prompt_ids_hash_depends_on_order_and_not_on_id_type_boundaries():
    assert hulder.prompt_ids_sha256(["a", "b"]) != hulder.prompt_ids_sha256(["b", "a"])
    assert hulder.prompt_ids_sha256(["ab", "c"]) != hulder.prompt_ids_sha256(["a", "bc"])
    assert hulder.prompt_ids_sha256(np.array([1, 2])) == hulder.prompt_ids_sha256(["1", "2"])


def _identity_probe():
    # One feature, no standardisation: the logit is the activation itself.
    return hulder.Probe(weights=np.array([1.0]), bias=0.0, mean=np.array([0.0]), std=np.array([1.0]))


def _as_activations(scores):
    return np.log(scores / (1 - scores)).astype(np.float32)[:, None]


def test_activations_with_a_probe_give_the_same_report_as_the_probes_scores():
    data = unchanged_scorer(seed=0, candidate_fpr=0.03)
    probe, reference, candidate = _identity_probe(), _as_activations(data.reference), _as_activations(data.candidate)
    kwargs = {"threshold": data.threshold, "target_fpr": 0.01, **_paired(data.labels)}
    from_scores = hulder.check(
        hulder.apply_probe(probe, reference), hulder.apply_probe(probe, candidate), data.labels, **kwargs
    )
    from_activations = hulder.check(reference, candidate, data.labels, probe=probe, **kwargs)
    assert from_activations.verdict is from_scores.verdict
    assert from_activations.mean_abs_shift == from_scores.mean_abs_shift
    assert from_activations.decision_flips == from_scores.decision_flips


def test_with_a_probe_the_logit_shift_is_reported_beside_the_score_shift():
    # Logit 20 saturates the float32 sigmoid: a +2 logit shift there moves the score by
    # nothing, so only the logit shift shows the probe's view moved.
    reference = np.array([[-3.0], [-2.0], [2.0], [20.0]], dtype=np.float32)
    candidate = reference + np.array([[0.0], [0.0], [0.0], [2.0]], dtype=np.float32)
    rerun = reference + np.array([[0.4], [0.0], [0.0], [0.0]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    report = hulder.check(
        reference, candidate, labels, threshold=0.5, target_fpr=0.01, probe=_identity_probe(), noise_floor=rerun
    )
    assert report.mean_abs_shift == 0.0
    assert report.mean_abs_logit_shift == pytest.approx(2.0 / 4)
    assert report.noise_floor_mean_abs_logit_shift == pytest.approx(0.4 / 4)


def test_bring_your_own_scores_have_no_logit_shift():
    data = unchanged_scorer(seed=0)
    report = hulder.check(data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01)
    assert report.mean_abs_logit_shift is None
    assert report.noise_floor_mean_abs_logit_shift is None


def test_two_dimensional_input_without_a_probe_is_rejected():
    with pytest.raises(ValueError, match="probe="):
        hulder.check(np.zeros((4, 2)), np.zeros((4, 2)), np.array([0, 0, 1, 1]), threshold=0.5, target_fpr=0.01)


def _check_with_rerun(noise_floor_ids):
    data = unchanged_scorer(seed=0, noise=FRESH_ENGINE_NOISE, p_zero=FRESH_ENGINE_P_ZERO)
    rerun = unchanged_scorer(seed=0, noise=FRESH_ENGINE_NOISE, p_zero=FRESH_ENGINE_P_ZERO, noise_seed=1).candidate
    return hulder.check(
        data.reference,
        data.candidate,
        data.labels,
        threshold=data.threshold,
        target_fpr=0.01,
        noise_floor=rerun,
        noise_floor_ids=noise_floor_ids,
        **_paired(data.labels),
    )


def test_rerun_ids_in_the_same_order_verify_the_rerun_pairing():
    ids = _paired(unchanged_scorer(seed=0).labels)["reference_ids"]
    assert _check_with_rerun(list(ids)).warnings == ()


def test_rerun_ids_in_a_different_order_raise_a_pairing_error():
    ids = _paired(unchanged_scorer(seed=0).labels)["reference_ids"]
    with pytest.raises(hulder.PairingError, match="noise-floor rerun"):
        _check_with_rerun(ids[::-1])


def test_a_rerun_without_ids_warns_that_its_pairing_was_not_verified():
    [warning] = _check_with_rerun(None).warnings
    assert warning.startswith("Rerun pairing not verified")


def test_ids_that_contradict_the_hash_in_their_own_metadata_raise_a_pairing_error():
    # The reviewer's case: both metadata hashes agree, but the candidate's ids are reversed.
    ids = [f"slabs/{i}" for i in range(len(unchanged_scorer(seed=0).labels))]
    meta = {"base_model": "m", "prompt_ids_sha256": hulder.prompt_ids_sha256(ids)}
    with pytest.raises(hulder.PairingError, match="candidate's ids don't match"):
        _check_with_ids(ids, ids[::-1], reference_meta=meta, candidate_meta=dict(meta))


@pytest.mark.parametrize("run", ["reference", "candidate", "noise_floor"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_scores_are_rejected(run, bad):
    # An all-NaN candidate used to come out SCORER DEGRADED: NaN is never above the threshold.
    data = unchanged_scorer(seed=0)
    runs = {
        "reference": data.reference.copy(),
        "candidate": data.candidate.copy(),
        "noise_floor": data.candidate.copy(),
    }
    runs[run][5] = bad
    with pytest.raises(ValueError, match=f"{run}.*not finite"):
        hulder.check(**runs, labels=data.labels, threshold=data.threshold, target_fpr=0.01)


def test_non_finite_activations_are_rejected_after_the_probe_scores_them():
    reference = np.array([[-3.0], [-2.0], [2.0], [3.0]], dtype=np.float32)
    candidate = reference.copy()
    candidate[0] = np.nan
    with pytest.raises(ValueError, match="candidate.*not finite"):
        hulder.check(
            reference, candidate, np.array([0, 0, 1, 1]), threshold=0.5, target_fpr=0.01, probe=_identity_probe()
        )


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf])
def test_a_non_finite_threshold_is_rejected(threshold):
    data = unchanged_scorer(seed=0)
    with pytest.raises(ValueError, match="threshold"):
        hulder.check(data.reference, data.candidate, data.labels, threshold=threshold, target_fpr=0.01)


@pytest.mark.parametrize(
    ("run", "make_bad"),
    [
        ("noise_floor", lambda s: s[:, None]),  # (n, 1) used to broadcast against (n,) to (n, n)
        ("noise_floor", lambda s: s[:1]),  # length 1 used to broadcast too
        ("candidate", lambda s: s[:-1]),
        ("labels", lambda s: s[:-1]),
    ],
)
def test_runs_and_labels_must_be_one_dimensional_and_the_same_length(run, make_bad):
    data = unchanged_scorer(seed=0)
    args = {
        "reference": data.reference,
        "candidate": data.candidate,
        "noise_floor": data.candidate,
        "labels": data.labels,
    }
    args[run] = make_bad(args[run])
    with pytest.raises(ValueError, match=run):
        hulder.check(**args, threshold=data.threshold, target_fpr=0.01)


@pytest.mark.parametrize("which", ["reference_ids", "candidate_ids", "noise_floor_ids"])
@pytest.mark.parametrize(
    ("make_bad", "problem"),
    [(lambda ids: ids[:5], "5 ids for 2101"), (lambda ids: ["x"] * len(ids), "repeat")],
)
def test_ids_must_match_the_number_of_scores_and_be_unique(which, make_bad, problem):
    data = unchanged_scorer(seed=0)
    ids = {**_paired(data.labels), "noise_floor_ids": _paired(data.labels)["reference_ids"]}
    ids[which] = make_bad(ids[which])
    with pytest.raises(ValueError, match=f"{which}.*{problem}"):
        hulder.check(
            data.reference,
            data.candidate,
            data.labels,
            threshold=data.threshold,
            target_fpr=0.01,
            noise_floor=data.candidate,
            **ids,
        )


@pytest.mark.parametrize(
    ("labels", "problem"),
    [
        ([0, 0, 2, 1], "found 2"),  # a 2 used to count as an attack for FNR and vanish from AUROC
        ([-1, -1, 1, 1], "found -1"),  # used to raise a bare ZeroDivisionError
        ([0, 0, 0, 0], "no attack"),
        ([1, 1, 1, 1], "no benign"),
    ],
)
def test_labels_must_be_zero_or_one_with_both_classes_present(labels, problem):
    scores = np.array([0.1, 0.2, 0.7, 0.8])
    with pytest.raises(ValueError, match=problem):
        hulder.check(scores, scores, np.array(labels), threshold=0.5, target_fpr=0.01)


def test_a_single_example_is_rejected_for_its_missing_class():
    with pytest.raises(ValueError, match="no attack"):
        hulder.check(np.array([0.2]), np.array([0.2]), np.array([0]), threshold=0.5, target_fpr=0.01)


def test_noise_floor_ids_without_a_noise_floor_are_rejected():
    # They used to be ignored, so reversed ids gave PASS with no warning.
    data = unchanged_scorer(seed=0)
    ids = _paired(data.labels)["reference_ids"]
    with pytest.raises(ValueError, match="noise_floor_ids.*no noise_floor"):
        hulder.check(
            data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01, noise_floor_ids=ids
        )


def test_changes_under_test_as_a_bare_string_is_rejected():
    # set("team") would declare the keys t, e, a and m.
    with pytest.raises(TypeError, match="changes_under_test"):
        _check_with_meta(_META, _AWQ, changes_under_test="team")


@pytest.mark.parametrize("target_fpr", [0.0, -0.01, 0.51, 1.0])
def test_target_fpr_must_be_a_rate_above_zero_and_at_most_a_half(target_fpr):
    # 1.0 meant as 1% used to give a confident THRESHOLD STALE.
    data = unchanged_scorer(seed=0)
    with pytest.raises(ValueError, match="target_fpr"):
        hulder.check(data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=target_fpr)


def test_everything_after_target_fpr_is_keyword_only():
    # The library-first decision's positional call check(r, c, l, t, f, rerun) used to bind the rerun to seed.
    data = unchanged_scorer(seed=0)
    with pytest.raises(TypeError, match="positional"):
        hulder.check(data.reference, data.candidate, data.labels, data.threshold, 0.01, data.candidate)


def test_report_says_explicitly_whether_the_noise_floor_was_measured():
    reference, labels = _benign_then_attack(1191, 11)
    candidate, rerun = reference + 0.01, reference.copy()
    rerun[11] = 0.6
    kwargs = {"threshold": 0.5, "target_fpr": 0.01}
    no_rerun = hulder.check(reference, candidate, labels, **kwargs)
    identical = hulder.check(reference, candidate, labels, noise_floor=reference.copy(), **kwargs)
    measured = hulder.check(reference, candidate, labels, noise_floor=rerun, **kwargs)
    assert (no_rerun.noise_floor_measured, no_rerun.rerun_identical_to_reference) == (False, False)
    assert (identical.noise_floor_measured, identical.rerun_identical_to_reference) == (False, True)
    assert (measured.noise_floor_measured, measured.rerun_identical_to_reference) == (True, False)


def test_report_says_explicitly_when_more_benign_examples_would_not_settle_it():
    # 24 of 1,191 benign flagged (2.02%) against the 2% line: undecided even at a million.
    reference, labels = _benign_then_attack(1191, 11)
    candidate, _ = _benign_then_attack(1191, 24)
    report = hulder.check(reference, candidate, labels, threshold=0.5, target_fpr=0.01)
    assert report.verdict is hulder.Verdict.INCONCLUSIVE
    assert report.more_benign_beyond_limit is True
    assert report.more_benign_needed is None


def test_a_decided_verdict_has_no_more_benign_needed_and_is_not_beyond_the_limit():
    data = unchanged_scorer(seed=0)
    report = hulder.check(data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01)
    assert report.verdict is hulder.Verdict.PASS
    assert (report.more_benign_needed, report.more_benign_beyond_limit) == (None, False)


@pytest.mark.parametrize(
    ("verdict", "code"),
    [("PASS", 0), ("THRESHOLD_STALE", 10), ("SCORER_DEGRADED", 11), ("INCONCLUSIVE", 12)],
)
def test_verdict_exit_codes_follow_adr_0003(verdict, code):
    assert hulder.Verdict[verdict].exit_code == code


def test_weighted_auroc_matches_the_reference_auroc_on_expanded_copies():
    # Ties within and across classes, which the half-credit rule must handle the same way.
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(40, int), np.ones(30, int)]
    scores = rng.integers(0, 12, len(labels)) / 12
    w_benign = gate_check._resample_counts(rng, 40)[:20]
    w_attack = gate_check._resample_counts(rng, 30)[:20]
    fast = gate_check._weighted_auroc(scores, labels, w_benign, w_attack)
    for row in range(20):
        weights = np.r_[w_benign[row], w_attack[row]]
        expected = gate_check._auroc(np.repeat(scores, weights), np.repeat(labels, weights))
        assert fast[row] == pytest.approx(expected, abs=1e-12)


def test_identical_runs_give_a_delta_auroc_interval_of_exactly_zero():
    # Each resample uses the same examples for both runs, so their AUROCs are equal.
    data = unchanged_scorer(seed=0)
    assert gate_check._paired_bootstrap_delta_auroc(data.reference, data.reference, data.labels, 0) == (0.0, 0.0)


def _fpr_case(k_flagged):
    # Reference and candidate identical, so FNR and ΔAUROC sit inside their bands.
    scores, labels = _benign_then_attack(1191, k_flagged)
    ci = binomtest(k_flagged, 1191).proportion_ci(confidence_level=0.95, method="exact")
    return scores, labels, ci


def _verdict_at(scores, candidate, labels, target_fpr):
    return hulder.check(scores, candidate, labels, threshold=0.5, target_fpr=target_fpr).verdict


def test_fpr_upper_bound_exactly_at_twice_target_passes():
    # PASS needs upper bound <= 2 x target: equality passes, one ulp less doesn't.
    scores, labels, ci = _fpr_case(22)
    assert _verdict_at(scores, scores, labels, ci.high / 2) is hulder.Verdict.PASS
    assert _verdict_at(scores, scores, labels, np.nextafter(ci.high / 2, 0)) is hulder.Verdict.INCONCLUSIVE


def test_fpr_lower_bound_exactly_at_twice_target_is_not_stale():
    # STALE needs lower bound > 2 x target.
    scores, labels, ci = _fpr_case(22)
    assert _verdict_at(scores, scores, labels, ci.low / 2) is hulder.Verdict.INCONCLUSIVE
    assert _verdict_at(scores, scores, labels, np.nextafter(ci.low / 2, 0)) is hulder.Verdict.THRESHOLD_STALE


def test_fpr_upper_bound_exactly_at_half_target_is_not_stale():
    # STALE needs upper bound < target / 2.
    scores, labels, ci = _fpr_case(1)
    assert _verdict_at(scores, scores, labels, 2 * ci.high) is hulder.Verdict.PASS
    assert _verdict_at(scores, scores, labels, np.nextafter(2 * ci.high, 1)) is hulder.Verdict.THRESHOLD_STALE


def _tolerance_reaching(base, bound):
    """The tolerance t with base + t == bound exactly, and the largest with base + t < bound."""
    t = bound - base
    while base + t < bound:
        t = np.nextafter(t, np.inf)
    while base + t > bound:
        t = np.nextafter(t, -np.inf)
    assert base + t == bound
    below = t
    while base + below >= bound:
        below = np.nextafter(below, -np.inf)
    return float(t), float(below)


def _fnr_case():
    # The candidate misses 200 attacks where the reference missed 120; every attack still
    # outscores every benign example, so ΔAUROC is 0, and FPR is unchanged.
    reference, labels = _benign_then_attack(1191, 11, k_missed=120)
    candidate, _ = _benign_then_attack(1191, 11, k_missed=200)
    flagged = binomtest(910 - 200, 910).proportion_ci(confidence_level=0.95, method="exact")
    # check computes the FNR as 1 - flagged / n, which can differ from missed / n in the last bit.
    return reference, candidate, labels, 1 - (910 - 120) / 910, 1 - flagged.high, 1 - flagged.low


def test_fnr_upper_bound_exactly_at_the_tolerance_passes(monkeypatch):
    reference, candidate, labels, reference_fnr, _, fnr_high = _fnr_case()
    at, below = _tolerance_reaching(reference_fnr, fnr_high)
    monkeypatch.setattr(gate_check, "FNR_TOLERANCE", at)
    assert _verdict_at(reference, candidate, labels, 0.01) is hulder.Verdict.PASS
    monkeypatch.setattr(gate_check, "FNR_TOLERANCE", below)
    assert _verdict_at(reference, candidate, labels, 0.01) is hulder.Verdict.INCONCLUSIVE


def test_fnr_lower_bound_exactly_at_the_tolerance_is_not_stale(monkeypatch):
    reference, candidate, labels, reference_fnr, fnr_low, _ = _fnr_case()
    at, below = _tolerance_reaching(reference_fnr, fnr_low)
    monkeypatch.setattr(gate_check, "FNR_TOLERANCE", at)
    assert _verdict_at(reference, candidate, labels, 0.01) is hulder.Verdict.INCONCLUSIVE
    monkeypatch.setattr(gate_check, "FNR_TOLERANCE", below)
    assert _verdict_at(reference, candidate, labels, 0.01) is hulder.Verdict.THRESHOLD_STALE


def _auroc_case():
    # Attacks below the threshold pushed further down: only ΔAUROC moves, CI about [-0.027, -0.018].
    data = unchanged_scorer(seed=0, missed_attack_shift=-1.0)
    low, high = gate_check._paired_bootstrap_delta_auroc(data.reference, data.candidate, data.labels, 0)
    return data, low, high


def _auroc_verdict(data):
    return hulder.check(data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01).verdict


def test_delta_auroc_lower_bound_exactly_at_the_tolerance_passes(monkeypatch):
    # PASS needs lower bound >= -tolerance.
    data, low, _ = _auroc_case()
    monkeypatch.setattr(gate_check, "AUROC_TOLERANCE", -low)
    assert _auroc_verdict(data) is hulder.Verdict.PASS
    monkeypatch.setattr(gate_check, "AUROC_TOLERANCE", np.nextafter(-low, 0))
    assert _auroc_verdict(data) is hulder.Verdict.INCONCLUSIVE


def test_delta_auroc_upper_bound_exactly_at_the_tolerance_is_not_degraded(monkeypatch):
    # SCORER DEGRADED needs upper bound < -tolerance.
    data, _, high = _auroc_case()
    monkeypatch.setattr(gate_check, "AUROC_TOLERANCE", -high)
    assert _auroc_verdict(data) is hulder.Verdict.INCONCLUSIVE
    monkeypatch.setattr(gate_check, "AUROC_TOLERANCE", np.nextafter(-high, 0))
    assert _auroc_verdict(data) is hulder.Verdict.SCORER_DEGRADED


def test_report_carries_fpr_and_fnr_with_clopper_pearson_intervals():
    # 22 of 1,191 benign flagged by the candidate, 11 by the reference; 120 of 910 attacks missed by both.
    reference, labels = _benign_then_attack(1191, 11)
    candidate, _ = _benign_then_attack(1191, 22)
    report = hulder.check(reference, candidate, labels, threshold=0.5, target_fpr=0.01)
    assert (report.n_benign, report.n_attack) == (1191, 910)
    for estimate, k in ((report.reference_fpr, 11), (report.candidate_fpr, 22)):
        ci = binomtest(k, 1191).proportion_ci(confidence_level=0.95, method="exact")
        assert estimate == hulder.Estimate(k / 1191, ci.low, ci.high)
    flagged = binomtest(790, 910).proportion_ci(confidence_level=0.95, method="exact")
    for estimate in (report.reference_fnr, report.candidate_fnr):
        assert estimate == hulder.Estimate(1 - 790 / 910, 1 - flagged.high, 1 - flagged.low)


def test_report_carries_auroc_for_each_run_and_the_paired_delta_with_bootstrap_intervals():
    data = unchanged_scorer(seed=0, missed_attack_shift=-1.0)
    report = hulder.check(data.reference, data.candidate, data.labels, threshold=data.threshold, target_fpr=0.01)
    reference_auroc = gate_check._auroc(data.reference, data.labels)
    candidate_auroc = gate_check._auroc(data.candidate, data.labels)
    assert report.reference_auroc.value == reference_auroc
    assert report.candidate_auroc.value == candidate_auroc
    assert report.delta_auroc.value == candidate_auroc - reference_auroc
    low, high = gate_check._paired_bootstrap_delta_auroc(data.reference, data.candidate, data.labels, 0)
    assert (report.delta_auroc.low, report.delta_auroc.high) == (low, high)
    for estimate in (report.reference_auroc, report.candidate_auroc, report.delta_auroc):
        assert estimate.low <= estimate.value <= estimate.high


def test_report_carries_the_interval_the_noise_floor_decision_reads():
    # shift_over_noise_floor is mean(|candidate - reference| - |rerun - reference|) with its
    # 95% paired bootstrap CI; the shift exceeds the floor when the lower bound is above 0.
    report = _determinism_case()
    excess = report.shift_over_noise_floor
    assert excess.value == pytest.approx(report.mean_abs_shift - report.noise_floor_mean_abs_shift)
    assert excess.low <= excess.value <= excess.high
    assert report.shift_exceeds_noise_floor is (excess.low > 0)


def test_the_noise_floor_interval_depends_on_the_seed_alone():
    # So the determinism tests, which compare whole Reports, catch an unseeded floor bootstrap.
    data = unchanged_scorer(seed=0, noise=FRESH_ENGINE_NOISE, p_zero=FRESH_ENGINE_P_ZERO)
    rerun = unchanged_scorer(seed=0, noise=FRESH_ENGINE_NOISE, p_zero=FRESH_ENGINE_P_ZERO, noise_seed=1).candidate

    def excess(seed):
        return hulder.check(
            data.reference,
            data.candidate,
            data.labels,
            threshold=data.threshold,
            target_fpr=0.01,
            seed=seed,
            noise_floor=rerun,
        ).shift_over_noise_floor

    assert excess(7) == excess(7)
    assert excess(7) != excess(8)


def test_without_a_measured_floor_there_is_no_floor_interval():
    reference, labels = _benign_then_attack(1191, 11)
    report = hulder.check(reference, reference + 0.01, labels, threshold=0.5, target_fpr=0.01)
    assert report.shift_over_noise_floor is None
