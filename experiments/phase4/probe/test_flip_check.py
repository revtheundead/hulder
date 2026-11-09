"""The pure part of the paired flip check's validation. CPU only, no captured data."""

import math

import flip_check as f
import numpy as np
import pytest


def test_discordant_counts_each_direction():
    reference = np.array([0.1, 0.9, 0.9, 0.1, 0.1])
    candidate = np.array([0.9, 0.1, 0.9, 0.1, 0.9])
    assert f.discordant(reference, candidate, 0.5) == (2, 1)


def test_discordant_flags_strictly_above_the_threshold():
    assert f.discordant(np.array([0.5]), np.array([0.6]), 0.5) == (1, 0)
    assert f.discordant(np.array([0.6]), np.array([0.5]), 0.5) == (0, 1)


def test_net_bound_matches_the_closed_form_when_all_flips_go_one_way():
    # Clopper-Pearson's one-sided lower bound for 85 of 85 is 0.05 ** (1 / 85).
    expected = 85 * (2 * 0.05 ** (1 / 85) - 1) / 2101
    assert math.isclose(f.net_lower_bound(85, 0, 2101), expected, rel_tol=1e-9)


def test_net_bound_is_symmetric_in_direction():
    assert f.net_lower_bound(0, 85, 2101) == f.net_lower_bound(85, 0, 2101)


def test_net_bound_is_zero_without_discordant_pairs():
    assert f.net_lower_bound(0, 0, 2101) == 0.0


def test_direction_p_is_one_without_discordant_pairs():
    assert f.direction_p(0, 0) == 1.0


def test_direction_p_matches_the_exact_two_sided_sign_test():
    assert math.isclose(f.direction_p(3, 1), 0.625)
    assert math.isclose(f.direction_p(85, 0), 2 * 0.5**85)


@pytest.mark.parametrize(
    ("up", "down", "margin", "fires"),
    [
        (85, 0, 0.02, True),  # sweep step C on calibration
        (0, 85, 0.02, True),  # the same, toward misses
        (85, 0, 0.04, False),  # its bound is about 3.8%
        (3, 1, 0.005, False),  # the real rerun: not one-way
        (0, 0, 0.005, False),
        (8, 0, 0.005, False),  # one-way but too small to be material
    ],
)
def test_fires(up, down, margin, fires):
    assert f.fires(up, down, 2101, margin) is fires


@pytest.mark.parametrize("current", ["PASS", "THRESHOLD STALE", "SCORER DEGRADED"])
def test_combined_verdict_never_touches_a_decided_verdict(current):
    assert f.combined_verdict(current, 85, 0, 2101, 0.005) == current


def test_combined_verdict_resolves_inconclusive_when_the_rule_fires():
    assert f.combined_verdict("INCONCLUSIVE", 85, 0, 2101, 0.005) == "THRESHOLD STALE"
    assert f.combined_verdict("INCONCLUSIVE", 3, 1, 2101, 0.005) == "INCONCLUSIVE"


def test_population_flips_reads_logits_at_a_score_threshold():
    reference = np.array([-1.0, 0.1, 2.0])
    out = f.population_flips(reference, reference + 0.5, 0.6)
    # scores 0.27, 0.52, 0.88 move to 0.38, 0.65, 0.92: one row crosses 0.6 upward
    assert out == {"up": 1, "down": 0, "n": 3, "net": 1 / 3}


def test_path_switch_shifts_are_on_both_sides():
    assert sorted(f.PATH_SWITCH_SHIFTS) == [-0.04, -0.02, 0.02, 0.04]


@pytest.mark.parametrize(
    ("fault", "planted_kind", "kind"),
    [
        ("real rerun", "null", "null"),
        ("rerun residuals", "null", "null"),
        ("path switch +0.02", "graded", "path-switch null"),
        ("shift +0.1", "graded", "small shift"),
        ("shift -0.1", "graded", "small shift"),
        ("shift +0.5", "graded", "graded"),
        ("shift +3", "gross", "gross"),
        ("wrong layer, 27 for 18", "gross, either verdict", "gross"),
        ("shift +0.1", "frozen", "frozen"),
    ],
)
def test_flip_kind(fault, planted_kind, kind):
    assert f.flip_kind(fault, planted_kind) == kind


def test_select_margin_takes_the_smallest_that_meets_every_bar():
    meets = {0.005: False, 0.01: True, 0.02: True}
    assert f.select_margin(meets) == 0.01


def test_select_margin_is_none_when_no_candidate_meets_the_bars():
    assert f.select_margin(dict.fromkeys(f.MARGINS, False)) is None


def test_bars_name_every_candidate_margin_and_both_rates():
    document = f.expected_document([])
    assert document["margins"] == [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    assert document["bar"]["null"].startswith("the rule fires in at most 2%")
    assert document["bar"]["small shift"].startswith("the rule fires in at most 5%")
    assert "at most 5%" in document["bar"]["population PASS"]


def row(**over):
    return {
        "split": "calibration",
        "mode": "refit",
        "kind": "graded",
        "planted_kind": "graded",
        "current_expected": "THRESHOLD STALE",
        **over,
    }


def test_tally_counts_only_what_the_rule_adds():
    verdicts = ["INCONCLUSIVE", "INCONCLUSIVE", "THRESHOLD STALE", "PASS"]
    t = f.tally_margin(verdicts, [85, 3, 85, 85], [0, 1, 0, 0], 2101, 0.02, "THRESHOLD STALE")
    assert t["fires"] == 1
    assert t["fires_rate"] == 0.25
    assert t["combined_counts"]["THRESHOLD STALE"] == 2
    assert t["combined_expected_rate"] == 0.5
    assert t["combined_non_pass_rate"] == 0.75
    assert t["only_inconclusive_to_stale"] is True


@pytest.mark.parametrize(
    ("over", "rate", "bar", "meets"),
    [
        ({"kind": "null", "planted_kind": "null", "current_expected": "PASS"}, 0.02, "null", True),
        ({"kind": "null", "planted_kind": "null", "current_expected": "PASS"}, 0.021, "null", False),
        ({"kind": "path-switch null", "current_expected": "PASS"}, 0.021, "path-switch null", False),
        ({"kind": "small shift", "current_expected": "PASS"}, 0.05, "small shift", True),
        ({"kind": "small shift", "current_expected": "PASS"}, 0.051, "small shift", False),
        ({"current_expected": "PASS"}, 0.051, "population PASS", False),
        ({"current_expected": "PASS"}, 0.05, "population PASS", True),
    ],
)
def test_bars_on_the_rate_the_rule_fires(over, rate, bar, meets):
    t = {"fires_rate": rate, "only_inconclusive_to_stale": True}
    assert f.row_bars(row(**over), t)[bar] is meets


def test_graded_rows_with_another_population_verdict_have_no_rate_bar():
    assert f.row_bars(row(), {"fires_rate": 0.9, "only_inconclusive_to_stale": True}) == {"every row": True}


def test_gross_bars_read_the_combined_verdict():
    gross = row(kind="gross", planted_kind="gross")
    either = row(kind="gross", planted_kind="gross, either verdict")
    t = {
        "fires_rate": 0.1,
        "only_inconclusive_to_stale": True,
        "combined_expected_rate": 0.94,
        "combined_non_pass_rate": 1.0,
    }
    assert f.row_bars(gross, t)["gross"] is False
    assert f.row_bars(either, t)["gross"] is True


def test_frozen_rows_have_no_bars():
    assert f.row_bars(row(mode="frozen", kind="frozen"), {"fires_rate": 1.0}) == {}


def test_decide_selects_on_calibration_and_replicates_on_test():
    def summary(split, fails_below):
        return {"split": split, "mode": "refit", "bars": {f"{m:g}": {"null": m >= fails_below} for m in f.MARGINS}}

    rows = [summary("calibration", 0.02), summary("test", 0.03)]
    synthetic = [{"fires_rate": {f"{m:g}": 0.0 for m in f.MARGINS}}]
    out = f.decide(rows, synthetic)
    assert out["selected_margin"] == 0.02
    assert out["test_meets"] is False
    assert out["ships"] is False


def test_decide_ships_nothing_without_a_margin():
    rows = [{"split": "calibration", "mode": "refit", "bars": {f"{m:g}": {"null": False} for m in f.MARGINS}}]
    out = f.decide(rows, [])
    assert out["selected_margin"] is None and out["ships"] is False


def test_summarise_reads_bars_from_the_stored_resamples():
    planted = {
        "split": "calibration",
        "mode": "refit",
        "fault": "real rerun",
        "kind": "null",
        "planted_kind": "null",
        "current_expected": "PASS",
    }
    synthetic = {"split": "synthetic", "mode": "seeds", "fault": "null", "kind": "synthetic null"}
    raw = {
        "calibration/refit/real rerun": {
            "verdicts": ["PASS", "INCONCLUSIVE"],
            "up": [0, 85],
            "down": [0, 0],
            "n": 2101,
        },
        "synthetic/seeds/null": {"verdicts": ["PASS", "PASS"], "up": [0, 0], "down": [0, 0], "n": 2101},
    }
    out = f.summarise([planted, synthetic], raw)
    row = out["rows"][0]
    assert row["current_counts"]["INCONCLUSIVE"] == 1
    assert row["by_margin"]["0.02"]["fires"] == 1
    assert row["bars"]["0.02"] == {"every row": True, "null": False, "population PASS": False}
    assert out["synthetic"][0]["fires_rate"]["0.005"] == 0.0
    assert out["decision"]["selected_margin"] == 0.04  # 85 up bounds at 3.8%
