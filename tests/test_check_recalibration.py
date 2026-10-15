"""The recalibrated threshold in Report, the printed report and --json (docs/design.md §6)."""

import json

import numpy as np
import pytest
from scipy.special import expit, logit
from synthetic import missed_attack_shift_for, unchanged_scorer

import hulder
from hulder import cli
from hulder.gate.recalibrate import binomial_tail_threshold

NOT_CALIBRATED = "wasn't calibrated on this snapshot"
INPUTS = {"reference": "ref.npz", "candidate": "cand.npz", "rerun": None, "probe": None}


def _check(data, target_fpr=0.01, threshold=None):
    return hulder.check(
        data.reference,
        data.candidate,
        data.labels,
        threshold=data.threshold if threshold is None else threshold,
        target_fpr=target_fpr,
    )


def _stale():
    report = _check(data := unchanged_scorer(seed=0, candidate_fpr=0.05))
    assert report.verdict is hulder.Verdict.THRESHOLD_STALE
    return data, report


def test_threshold_stale_carries_the_binomial_tail_threshold_of_the_candidate():
    data, report = _stale()
    benign = data.labels == 0
    fit = binomial_tail_threshold(data.candidate[benign], 0.01, attack_scores=data.candidate[~benign])
    r = report.recalibration
    assert (r.method, r.confidence) == ("binomial-tail", 0.95)
    assert r.threshold == fit.threshold and r.benign_above == fit.benign_above
    assert r.benign_needed is None
    missed = int((data.candidate[~benign] <= r.threshold).sum())
    assert r.candidate_fnr.value == pytest.approx(missed / int((~benign).sum()))
    assert r.candidate_fnr.low <= r.candidate_fnr.value <= r.candidate_fnr.high


def test_inconclusive_carries_one_too():
    report = _check(unchanged_scorer(seed=0, candidate_fpr=0.015))
    assert report.verdict is hulder.Verdict.INCONCLUSIVE
    assert report.recalibration is not None and report.recalibration.threshold is not None


@pytest.mark.parametrize(
    "data",
    [unchanged_scorer(seed=0), unchanged_scorer(seed=0, missed_attack_shift=missed_attack_shift_for(-0.1))],
    ids=["pass", "scorer_degraded"],
)
def test_pass_and_scorer_degraded_carry_none(data):
    report = _check(data)
    assert report.verdict in (hulder.Verdict.PASS, hulder.Verdict.SCORER_DEGRADED)
    assert report.recalibration is None


def test_too_few_benign_to_certify_the_target_gives_the_count_needed():
    report = _check(unchanged_scorer(seed=0, n_benign=180, n_attack=180, candidate_fpr=0.05))
    assert report.verdict in (hulder.Verdict.THRESHOLD_STALE, hulder.Verdict.INCONCLUSIVE)
    r = report.recalibration
    assert r.threshold is None and r.benign_above is None and r.candidate_fnr is None
    assert r.benign_needed == 299


def test_a_probe_recalibrates_in_score_space():
    data, report = _stale()
    probe = hulder.Probe(weights=np.array([1.0]), bias=0.0, mean=np.array([0.0]), std=np.array([1.0]))
    as_logits = lambda scores: logit(scores)[:, None]
    from_activations = hulder.check(
        as_logits(data.reference),
        as_logits(data.candidate),
        data.labels,
        threshold=data.threshold,
        target_fpr=0.01,
        probe=probe,
    )
    benign = data.labels == 0
    assert from_activations.recalibration.benign_above == int(
        (expit(logit(data.candidate[benign])) > from_activations.recalibration.threshold).sum()
    )
    assert from_activations.recalibration.threshold == pytest.approx(report.recalibration.threshold, rel=1e-6)


def test_a_promoted_binomial_tail_threshold_is_accepted_as_calibrated_on_its_snapshot():
    # 3 of 180 benign above it: the reference's FPR CI is [0.3%, 4.8%], which excludes the 5% target.
    data = unchanged_scorer(seed=0, target_fpr=0.05, n_benign=180, n_attack=180)
    benign = data.labels == 0
    fit = binomial_tail_threshold(data.reference[benign], 0.05)
    report = _check(data, target_fpr=0.05, threshold=fit.threshold)
    assert report.reference_fpr.high < 0.05
    assert not any(NOT_CALIBRATED in w for w in report.warnings)


def test_a_threshold_flagging_fewer_benign_than_binomial_tail_allows_still_warns():
    data = unchanged_scorer(seed=0, target_fpr=0.05, n_benign=180, n_attack=180)
    too_high = float(data.reference[data.labels == 0].max())
    report = _check(data, target_fpr=0.05, threshold=too_high)
    assert any(NOT_CALIBRATED in w for w in report.warnings)


def test_report_is_unhashable_by_design():
    _, report = _stale()
    assert hulder.Report.__hash__ is None
    with pytest.raises(TypeError):
        hash(report)


def test_the_printed_report_gives_the_threshold_its_guarantee_and_that_it_needs_confirming():
    _, report = _stale()
    text = cli.render(report, INPUTS)
    r = report.recalibration
    assert f"Recalibrated threshold  {r.threshold!r} for a 1.0% target FPR (binomial-tail, 95% confidence)" in text
    assert f"{r.benign_above} of {report.n_benign:,} benign" in text
    assert "fitted on this candidate's calibration scores" in text
    assert "confirm it on a fresh candidate snapshot" in text
    assert text.index("Recalibrated threshold") < text.index("Performance at threshold")


def test_inconclusive_labels_it_optional_and_pass_prints_none():
    inconclusive = cli.render(_check(unchanged_scorer(seed=0, candidate_fpr=0.015)), INPUTS)
    assert "Recalibrated threshold (optional)" in inconclusive
    assert "Recalibrated threshold" not in cli.render(_check(unchanged_scorer(seed=0)), INPUTS)


def test_the_printed_report_says_how_many_benign_could_certify_the_target():
    report = _check(unchanged_scorer(seed=0, n_benign=180, n_attack=180, candidate_fpr=0.05))
    text = cli.render(report, INPUTS)
    title = "Recalibrated threshold" + (" (optional)" if report.verdict is hulder.Verdict.INCONCLUSIVE else "")
    assert f"{title}: none. 180 benign can't certify a 1.0% FPR at 95% confidence; 299 benign in all could." in text


def test_json_carries_the_recalibration():
    _, report = _stale()
    out = json.loads(cli.to_json(report, INPUTS))["report"]["recalibration"]
    assert out["threshold"] == report.recalibration.threshold
    assert set(out) == {"method", "confidence", "threshold", "benign_above", "candidate_fnr", "benign_needed"}
