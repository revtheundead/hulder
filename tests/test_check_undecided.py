"""INCONCLUSIVE reports name each undecided condition, what would settle it, and when PASS is out of
reach at the current size."""

import numpy as np
import pytest
from synthetic import missed_attack_shift_for, unchanged_scorer

import hulder
from hulder.gate.check import AUROC_TOLERANCE, FNR_TOLERANCE, FPR_BAND_HIGH
from hulder.gate.settle import fnr_floor, fpr_floor, minimum_for_pass

TARGET = 0.01


def report_for(seed, target=TARGET, **kwargs):
    d = unchanged_scorer(seed, target_fpr=target, **kwargs)
    return d, hulder.check(d.reference, d.candidate, d.labels, threshold=d.threshold, target_fpr=target)


def first_with(condition, seeds=range(200), target=TARGET, **kwargs):
    for seed in seeds:
        d, report = report_for(seed, target=target, **kwargs)
        found = [u for u in report.undecided if u.condition == condition]
        if found:
            return d, report, found[0]
    pytest.fail(f"no seed left {condition} undecided")


def straddles(u):
    if u.condition == "delta_auroc":
        return u.estimate.low < u.line <= u.estimate.high
    return u.estimate.low <= u.line < u.estimate.high


def test_an_undecided_fpr_carries_its_floor():
    _, report, u = first_with("fpr", candidate_fpr=0.015)
    assert report.verdict is hulder.Verdict.INCONCLUSIVE
    assert u.estimate == report.candidate_fpr and u.line == FPR_BAND_HIGH * TARGET
    k = round(report.candidate_fpr.value * report.n_benign)
    assert u.floor == fpr_floor(k, report.n_benign, u.line)


def test_an_undecided_fnr_carries_its_floor_against_the_moving_line():
    d, report, u = first_with("fnr", n_benign=180, n_attack=180, attack_shift=-0.25, target=0.05)
    attack = d.labels == 1
    cand_misses = int((d.candidate[attack] <= d.threshold).sum())
    ref_misses = int((d.reference[attack] <= d.threshold).sum())
    assert u.estimate == report.candidate_fnr
    assert u.line == pytest.approx(report.reference_fnr.value + FNR_TOLERANCE)
    assert u.floor == fnr_floor(cand_misses, ref_misses, report.n_attack, FNR_TOLERANCE)


def test_an_undecided_delta_auroc_has_no_floor():
    _, report, u = first_with("delta_auroc", missed_attack_shift=missed_attack_shift_for(-0.02))
    assert u.estimate == report.delta_auroc and u.line == -AUROC_TOLERANCE
    assert u.floor is None


def test_every_inconclusive_report_names_only_and_at_least_one_undecided_condition():
    for seed in range(40):
        _, report = report_for(seed, candidate_fpr=0.02)
        if report.verdict is hulder.Verdict.INCONCLUSIVE:
            assert report.undecided
            assert all(straddles(u) for u in report.undecided)
        else:
            assert report.undecided == ()


def test_pass_out_of_reach_at_180_benign_and_a_1_percent_target():
    _, report = report_for(0, n_benign=180, n_attack=180)
    assert report.verdict is hulder.Verdict.INCONCLUSIVE
    [fpr] = [r for r in report.out_of_reach if r.condition == "fpr"]
    assert (fpr.n, fpr.minimum) == (180, minimum_for_pass(0.02)) == (180, 183)


def test_pass_within_reach_at_the_calibration_size():
    _, report = report_for(0, candidate_fpr=0.02)
    assert report.out_of_reach == ()


def test_fnr_out_of_reach_when_even_no_misses_cant_clear_its_line():
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(400, int), np.ones(40, int)]
    reference = np.where(labels == 1, rng.uniform(0.6, 1.0, 440), rng.uniform(0.0, 0.4, 440))
    reference[:20] = 0.6  # 5% of benign flagged, inside the FPR band
    candidate = reference.copy()
    candidate[400:402] = 0.1  # two attacks newly missed
    report = hulder.check(reference, candidate, labels, threshold=0.5, target_fpr=0.05)
    assert report.verdict is hulder.Verdict.INCONCLUSIVE
    [fnr] = [r for r in report.out_of_reach if r.condition == "fnr"]
    assert (fnr.n, fnr.minimum) == (40, minimum_for_pass(0.05))


def test_decided_verdicts_have_neither():
    _, report = report_for(0)
    assert report.verdict is hulder.Verdict.PASS
    assert (report.undecided, report.out_of_reach) == ((), ())
