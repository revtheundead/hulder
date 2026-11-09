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
    assert document["margins"] == [0.005, 0.01, 0.02]
    assert document["bar"]["null"].startswith("the rule fires in at most 2%")
    assert document["bar"]["small shift"].startswith("the rule fires in at most 5%")
