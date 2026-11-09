"""The settle calibration's fixed parts: its cases, its bars and how it reads them."""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm
from synthetic import unchanged_scorer

import hulder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/phase2"))

import settle_validation as v


def test_fnr_rise_shift_raises_the_population_fnr_by_the_rise():
    for target in (0.01, 0.05):
        t = norm.ppf(1 - target) - v.SEPARATION
        shift = v.fnr_rise_shift(0.04, target)
        assert np.isclose(norm.cdf(t - shift) - norm.cdf(t), 0.04)


def test_the_bar_is_fixed_on_fresh_seeds_at_two_sizes():
    doc = v.expected_document()
    assert doc["seeds"] == [2000, 3000]
    assert doc["sizes"]["small"] == {"n_benign": 180, "n_attack": 180, "target_fpr": 0.05}
    assert doc["sizes"]["calibration-sized"]["target_fpr"] == 0.01
    assert doc["bins"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert (doc["min_seeds"], doc["calibration_tolerance"], doc["wrong_way_max"]) == (50, 0.1, 0.01)


def test_cases_include_every_line_and_scale_with_the_target():
    small = v.cases(0.05)
    assert small["FPR at 2x the target, on the line"]["kwargs"] == {"candidate_fpr": 0.1}
    on_line = [name for name, c in small.items() if None in c["side"].values()]
    assert len(on_line) == 3


def test_a_condition_the_case_does_not_move_sits_inside_its_tolerance():
    case = v.cases(0.01)["FNR rise of 0.06"]
    assert v.population_side(case, "fnr") == "fail"
    assert v.population_side(case, "fpr") == "pass"


def test_bins_compare_the_realised_rate_with_the_mean_stated_chance():
    entries = [{"chance": 0.8, "settled": "pass" if i < 70 else None} for i in range(100)]
    entries += [{"chance": 0.3, "settled": None} for _ in range(10)]
    [low, high] = v.bins_of(entries)
    assert low["bar_applies"] is False and low["meets_bar"] is None
    assert (high["seeds"], high["realised"], high["meets_bar"]) == (100, 0.7, True)


def test_a_bin_outside_ten_points_misses():
    entries = [{"chance": 0.85, "settled": "pass" if i < 70 else None} for i in range(100)]
    assert v.bins_of(entries)[0]["meets_bar"] is False


def test_wrong_way_is_a_share_of_settled_seeds():
    row = {"case": "FPR at 2.5x the target", "size": "calibration-sized"}
    entries = [{"condition": "fpr", "settled": "fail"} for _ in range(99)] + [{"condition": "fpr", "settled": "pass"}]
    [w] = v.wrong_way(row, entries)
    assert (w["wrong_way"], w["share"], w["meets_bar"]) == (1, 0.01, True)


def test_side_matches_a_full_check():
    for seed in range(5):
        d = unchanged_scorer(seed, n_benign=300, n_attack=300, candidate_fpr=0.02, attack_shift=-0.2)
        report = hulder.check(d.reference, d.candidate, d.labels, threshold=d.threshold, target_fpr=0.01)
        fpr = report.candidate_fpr
        line = report.reference_fnr.value + 0.05
        fnr = report.candidate_fnr
        expected = {
            "fpr": "pass" if fpr.high <= 0.02 else "fail" if fpr.low > 0.02 else None,
            "fnr": "pass" if fnr.high <= line else "fail" if fnr.low > line else None,
        }
        for condition, side in expected.items():
            assert v.side(d, condition, 0.01) == side
