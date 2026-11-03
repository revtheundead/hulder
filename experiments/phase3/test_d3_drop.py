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


def arm_run(arm, rnd, *tpots):
    return {"arm": arm, "round": rnd, "benches": [{"p50_tpot_ms": t} for t in tpots]}


def test_a_bench_at_over_twice_its_arms_median_tpot_disturbs_its_round():
    runs = [
        arm_run("baseline", 0, 52.0),
        arm_run("baseline", 1, 14.5),
        arm_run("baseline", 2, 14.4),
        arm_run("scoring", 0, 21.0),
        arm_run("scoring", 1, 27.0),
        arm_run("scoring", 2, 24.0),
    ]
    assert d.disturbed_rounds(runs, bench=0) == [0]


def test_exactly_twice_the_median_is_not_disturbed():
    runs = [arm_run("baseline", 0, 20.0), arm_run("baseline", 1, 10.0), arm_run("baseline", 2, 10.0)]
    assert d.disturbed_rounds(runs, bench=0) == []


def test_no_round_is_disturbed_when_the_arms_agree():
    runs = [arm_run("baseline", r, 14.0 + r) for r in range(3)]
    assert d.disturbed_rounds(runs, bench=0) == []


def test_the_gpu_state_parses_nvidia_smis_line():
    assert d.parse_gpu_state("1605, 7000, P0, 123.45, 71, 0x0000000000000000") == {
        "clock_mhz": 1605,
        "memory_clock_mhz": 7000,
        "pstate": "P0",
        "power_w": 123.45,
        "temperature_c": 71,
        "throttle_reasons": "0x0000000000000000",
    }


def test_the_attribution_is_reported_with_and_without_disturbed_rounds():
    triples = [(5.0, 2.0, 13.0), (19.0, 13.0, 12.0), (18.0, 13.0, 13.0)]
    runs = []
    for i, (b, z, s) in enumerate(triples):
        tpot = 50.0 if i == 0 else 14.0
        for arm, rate in (("baseline", b), ("b_zero_draft", z), ("scoring", s)):
            runs.append({"arm": arm, "round": i, "benches": [{"p50_tpot_ms": tpot, "request_throughput": rate}]})
    result = d.summarise(runs, benches=1)["bench_0"]
    assert result["disturbed_rounds"] == [0]
    assert result["all_rounds"]["finding"] == "no drop"
    assert result["undisturbed"]["finding"] == "extraction"
