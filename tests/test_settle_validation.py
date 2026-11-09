"""The settle-estimate validation's fixed parts: its cases and its bar."""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/phase2"))

import settle_validation as v


def test_fnr_rise_shift_raises_the_population_fnr_by_the_rise():
    t = norm.ppf(1 - v.TARGET_FPR) - v.SEPARATION
    shift = v.fnr_rise_shift(0.04)
    # attacks move by shift, so FNR = Φ(t - shift)
    assert np.isclose(norm.cdf(t - shift) - norm.cdf(t), 0.04)
    assert shift < 0


def test_cases_straddle_the_lines():
    names = v.cases()
    assert names["FPR at 2x the target, on the line"] == {"candidate_fpr": 0.02}
    assert set(names) >= {"null", "FPR at 1.5x the target", "FNR rise of 0.04", "FNR rise of 0.06"}


def test_the_bar_is_a_band_around_the_promise():
    doc = v.expected_document()
    assert doc["chance"] == 0.8
    assert doc["band"] == [0.7, 0.9]
    assert doc["min_seeds"] == 50
    assert doc["seeds"] == [0, 1000]
    assert set(doc["sizes"]) == {"calibration-sized", "quickstart-sized"}
    assert doc["sizes"]["quickstart-sized"] == {"n_benign": 180, "n_attack": 180}


def test_only_conditions_that_can_be_undecided_are_validated():
    assert v.expected_document()["conditions"] == ["fpr_high", "fnr", "delta_auroc"]


def test_population_sides():
    assert v.population_side("FPR at 2.5x the target", "fpr") == "fail"
    assert v.population_side("FPR at 2x the target, on the line", "fpr") is None
    assert v.population_side("FNR rise of 0.06", "fpr") == "pass"
    assert v.population_side("FNR rise of 0.04", "fnr") == "pass"


def test_summarise_counts_settled_seeds_against_the_band():
    seeds = [
        {
            "verdict": "INCONCLUSIVE",
            "undecided": [
                {"condition": "fpr", "more_benign": 10, "more_attack": 0, "settled": "pass" if i < 40 else None}
            ],
        }
        for i in range(50)
    ]
    seeds.append(
        {"verdict": "INCONCLUSIVE", "undecided": [{"condition": "fpr", "more_benign": None, "more_attack": None}]}
    )
    out = v.summarise({"case": "FPR at 1.5x the target", "size": "calibration-sized"}, {"seeds": seeds})
    assert len(out) == 1
    s = out[0]
    assert (s["undecided_seeds"], s["beyond_limit"], s["with_estimate"], s["settled"]) == (51, 1, 50, 40)
    assert s["settle_rate"] == 0.8 and s["meets_bar"] is True and s["wrong_way"] == 0


def test_extended_side_matches_a_full_check():
    from synthetic import unchanged_scorer

    import hulder

    for seed in range(5):
        d = unchanged_scorer(seed, n_benign=300, n_attack=300, candidate_fpr=0.02, attack_shift=-0.2)
        report = hulder.check(d.reference, d.candidate, d.labels, threshold=d.threshold, target_fpr=0.01)
        for condition in ("fpr", "fnr", "delta_auroc"):
            assert v.extended_side(d, condition) == v.side(report, condition)
