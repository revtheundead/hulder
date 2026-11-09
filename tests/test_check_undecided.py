"""INCONCLUSIVE reports name each undecided condition and the extra prompts that would settle it."""

import math

import pytest
from synthetic import missed_attack_shift_for, unchanged_scorer

import hulder
from hulder.gate.check import AUROC_TOLERANCE, FNR_TOLERANCE, FPR_BAND_HIGH
from hulder.gate.settle import extra_to_settle, scale_to_settle_auroc

TARGET = 0.01


def report_for(seed, **kwargs):
    d = unchanged_scorer(seed, **kwargs)
    return hulder.check(d.reference, d.candidate, d.labels, threshold=d.threshold, target_fpr=TARGET)


def first_with(condition, seeds=range(200), **kwargs):
    for seed in seeds:
        report = report_for(seed, **kwargs)
        found = [u for u in report.undecided if u.condition == condition]
        if found:
            return report, found[0]
    pytest.fail(f"no seed left {condition} undecided")


def straddles(u):
    if u.condition == "delta_auroc":
        return u.estimate.low < u.line <= u.estimate.high
    return u.estimate.low <= u.line < u.estimate.high


def test_an_undecided_fpr_names_its_interval_line_and_benign_needed():
    report, u = first_with("fpr", candidate_fpr=0.015)
    assert report.verdict is hulder.Verdict.INCONCLUSIVE
    assert u.estimate == report.candidate_fpr and u.line == FPR_BAND_HIGH * TARGET
    k = round(report.candidate_fpr.value * report.n_benign)
    assert u.more_benign == extra_to_settle(k, report.n_benign, u.line)
    assert u.more_attack == (None if u.more_benign is None else 0)


def test_an_undecided_fnr_names_attacks_needed_against_the_reference_line():
    report, u = first_with("fnr", n_benign=180, n_attack=180, attack_shift=-0.25)
    assert u.estimate == report.candidate_fnr
    assert u.line == pytest.approx(report.reference_fnr.value + FNR_TOLERANCE)
    misses = round(report.candidate_fnr.value * report.n_attack)
    assert u.more_attack == extra_to_settle(misses, report.n_attack, u.line)
    assert u.more_benign == (None if u.more_attack is None else 0)


def test_an_undecided_delta_auroc_grows_both_classes():
    report, u = first_with("delta_auroc", missed_attack_shift=missed_attack_shift_for(-0.02))
    assert u.estimate == report.delta_auroc and u.line == -AUROC_TOLERANCE
    scale = scale_to_settle_auroc(report.delta_auroc, -AUROC_TOLERANCE)
    if scale is None:
        assert (u.more_benign, u.more_attack) == (None, None)
    else:
        assert u.more_benign == math.ceil((scale - 1) * report.n_benign)
        assert u.more_attack == math.ceil((scale - 1) * report.n_attack)


def test_every_inconclusive_report_names_only_and_at_least_one_undecided_condition():
    for seed in range(40):
        report = report_for(seed, candidate_fpr=0.02)
        if report.verdict is hulder.Verdict.INCONCLUSIVE:
            assert report.undecided
            assert all(straddles(u) for u in report.undecided)
        else:
            assert report.undecided == ()
