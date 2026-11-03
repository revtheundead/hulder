"""The D3 spike's pure parts: comparing scores, reading /metrics and the verdict. CPU only."""

import json

import d3_spike as d
import numpy as np
import pytest


def test_compare_scores_counts_replies_without_one_and_takes_the_largest_gap_over_the_rest():
    result = d.compare_scores([0.5, None, 0.25], np.array([0.5000005, 0.9, 0.25]))
    assert result["requests"] == 3
    assert result["compared"] == 2
    assert result["without_score"] == 1
    assert result["max_abs_delta"] == pytest.approx(5e-7)


METRICS = """\
# TYPE hulder_probe_score histogram
hulder_probe_score_count{{engine="0",model_name="m"}} {scores}
hulder_probe_score_sum{{engine="0",model_name="m"}} 1.5
# TYPE hulder_probe_flagged counter
hulder_probe_flagged_total{{engine="0",model_name="m"}} {flagged}
# TYPE hulder_probe_replies_with_score counter
hulder_probe_replies_with_score_total{{engine="0",model_name="m"}} {with_score}
# TYPE hulder_probe_replies_without_score counter
hulder_probe_replies_without_score_total{{engine="0",model_name="m"}} {without}
"""


def test_probe_metrics_is_the_difference_between_two_scrapes():
    pytest.importorskip("prometheus_client")  # from the vllm extra
    before = METRICS.format(scores=2, flagged=1, with_score=2, without=0)
    after = METRICS.format(scores=7, flagged=2, with_score=5, without=2)
    assert d.probe_metrics(before, after) == {
        "scores": 5,
        "flagged": 1,
        "replies_with_score": 3,
        "replies_without_score": 2,
    }


def correct(**changes):
    return {"requests": 360, "compared": 360, "without_score": 0, "max_abs_delta": 5e-7, **changes}


def surface(**changes):
    metrics = {"scores": 360, "flagged": 10, "replies_with_score": 358, "replies_without_score": 2}
    base = {
        "requests": 360,
        "without_score": 2,
        "flagged_in_replies": 9,
        "files_written": 0,
        "metrics": metrics,
    }
    return {**base, **changes}


STREAMED = {"requests": 16, "chunks_with_a_score": 0, "metrics": {"scores": 16}}
GENERATION = {"prompts": 32, "identical": 32, "draft_tokens": 0}


def verdict(**changes):
    args = {"correct": [correct(), correct()], "generation": GENERATION, "surfaces": [surface()], "streamed": STREAMED}
    return d.verdict(**{**args, **changes})


def test_everything_as_expected_passes():
    assert verdict() == {"correct": "pass", "replies_unchanged": "pass", "surfaces": "pass", "overall": "pass"}


def test_one_score_past_the_tolerance_fails_correctness():
    result = verdict(correct=[correct(), correct(max_abs_delta=2e-6)])
    assert result["correct"] == "fail"
    assert result["overall"] == "fail"


def test_replies_without_a_score_at_the_correctness_stage_need_a_decision_rather_than_a_pass():
    result = verdict(correct=[correct(compared=359, without_score=1), correct()])
    assert result["correct"] == "needs decision"
    assert result["overall"] == "needs decision"


def test_a_changed_reply_fails():
    assert verdict(generation={**GENERATION, "identical": 31})["replies_unchanged"] == "fail"


def test_a_draft_token_fails():
    assert verdict(generation={**GENERATION, "draft_tokens": 1})["replies_unchanged"] == "fail"


@pytest.mark.parametrize(
    "changes",
    [
        {
            "metrics": {"scores": 359, "flagged": 10, "replies_with_score": 358, "replies_without_score": 2}
        },  # a lost score
        {
            "metrics": {"scores": 360, "flagged": 10, "replies_with_score": 357, "replies_without_score": 2}
        },  # a reply uncounted
        {"without_score": 3},  # the counter disagrees with the replies
        {"flagged_in_replies": 11},  # more flagged in replies than on /metrics
        {
            "metrics": {"scores": 360, "flagged": 12, "replies_with_score": 358, "replies_without_score": 2}
        },  # more than can be late
        {"files_written": 1},
    ],
)
def test_surfaces_fail_when_the_counts_disagree(changes):
    assert verdict(surfaces=[surface(**changes)])["surfaces"] == "fail"


def test_surfaces_fail_when_a_streamed_score_is_lost_or_a_chunk_carries_one():
    assert verdict(streamed={**STREAMED, "metrics": {"scores": 15}})["surfaces"] == "fail"
    assert verdict(streamed={**STREAMED, "chunks_with_a_score": 1})["surfaces"] == "fail"


def test_a_failure_outranks_a_decision_overall():
    result = verdict(
        correct=[correct(compared=359, without_score=1), correct()], generation={**GENERATION, "identical": 31}
    )
    assert result["overall"] == "fail"


def test_flips_skips_replies_without_a_score_and_stays_a_plain_int_for_the_results_file():
    flips = d.flips([0.6, None, 0.4], np.array([0.4, 0.9, 0.4], dtype=np.float32), threshold=0.5)
    assert flips == 1
    assert json.dumps(flips) == "1"
