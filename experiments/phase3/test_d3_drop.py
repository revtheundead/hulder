"""The saturation drop's attribution, on worked examples. CPU only."""

import d3_drop as d
import pytest


def rows(*triples):
    """(round, baseline, zero-draft, scoring) saturation in req/s."""
    return [{"round": i, "baseline": b, "b_zero_draft": z, "scoring": s} for i, (b, z, s) in enumerate(triples)]


def test_each_rounds_drops_are_paired_within_the_round():
    result = d.attribute(rows((20.0, 10.0, 5.0), (10.0, 10.0, 5.0)))
    assert result["extraction_drops"] == pytest.approx([0.5, 0.0])
    assert result["scoring_drops"] == pytest.approx([0.5, 0.5])
    assert result["total_drops"] == pytest.approx([0.75, 0.5])


def test_a_drop_almost_all_at_extraction_is_attributed_there():
    result = d.attribute(rows((20.0, 10.0, 10.0), (20.0, 10.0, 9.8), (20.0, 10.0, 10.2)))
    assert result["finding"] == "extraction"


def test_a_drop_almost_all_at_scoring_is_attributed_there():
    result = d.attribute(rows((20.0, 20.0, 10.0), (20.0, 19.5, 10.0), (20.0, 20.5, 10.0)))
    assert result["finding"] == "scoring"


def test_a_drop_split_between_the_steps_is_both():
    # 20 to 15 to 10: half of the total's loss at each step.
    assert d.attribute(rows((20.0, 15.0, 10.0)))["finding"] == "both"


def test_the_share_is_of_the_req_s_lost_not_the_percentages():
    # 20 to 10 loses 10 req/s, then 10 to 8 loses 2: 83% at extraction even though 50% then 20%.
    assert d.attribute(rows((20.0, 10.0, 8.0)))["extraction_share"] == pytest.approx(10 / 12)


def test_under_a_10_percent_mean_drop_nothing_reproduced():
    assert d.attribute(rows((20.0, 19.5, 19.0), (20.0, 19.5, 18.5)))["finding"] == "no drop"


def test_rounds_that_disagree_on_the_step_are_flagged_rather_than_averaged():
    # One round's whole drop at extraction, another's whole drop at scoring.
    result = d.attribute(rows((20.0, 10.0, 10.0), (20.0, 20.0, 10.0)))
    assert result["finding"] == "rounds disagree"


def test_a_scoring_arm_faster_than_the_stock_write_counts_as_no_scoring_loss():
    result = d.attribute(rows((20.0, 8.0, 10.0), (20.0, 8.5, 10.0), (20.0, 8.0, 9.5)))
    assert result["finding"] == "extraction"
    assert result["scoring_share"] == 0
