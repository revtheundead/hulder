"""draft_counter.py's rule and parsing, on CPU."""

import pytest

pytest.importorskip("prometheus_client")

import draft_counter as d


def arm(warnings=0, hulder=0, vllm=0):
    return {"warnings": warnings, "hulder_draft_tokens": hulder, "vllm_draft_tokens": vllm}


def test_counters_sum_engines_and_read_absent_ones_as_zero():
    text = (
        "# TYPE hulder_draft_tokens counter\n"
        'hulder_draft_tokens_total{engine="0"} 3.0\n'
        'hulder_draft_tokens_total{engine="1"} 2.0\n'
    )
    assert d.counters(text) == {"hulder_draft_tokens": 5, "vllm_draft_tokens": 0, "scores": 0}


def test_only_the_runtimes_lines_count_as_warnings():
    log = "INFO vllm started\nhulder.runtime: 8 draft tokens were scheduled in one step\nWARNING other\n"
    assert d.runtime_warnings(log) == 1


def test_the_rule_passes_quiet_async_on_and_one_warning_with_drafts_counted_async_off():
    assert d.verdict({"async_on": arm(), "async_off": arm(1, 500, 480)}) == {
        "async_on": True,
        "async_off": True,
        "overall": "pass",
    }


@pytest.mark.parametrize(
    "results",
    [
        {"async_on": arm(hulder=1), "async_off": arm(1, 500, 480)},  # a draft with async on
        {"async_on": arm(warnings=1), "async_off": arm(1, 500, 480)},  # a warning with async on
        {"async_on": arm(), "async_off": arm(0, 500, 480)},  # no warning with drafts
        {"async_on": arm(), "async_off": arm(2, 500, 480)},  # warned twice
        {"async_on": arm(), "async_off": arm(1, 400, 480)},  # undercounted against vLLM
        {"async_on": arm(), "async_off": arm(1, 0, 0)},  # never drafted, so nothing was tested
        {"async_on": arm(), "async_off": {"error": "RuntimeError: async_off exited with 1"}},
    ],
)
def test_the_rule_fails_otherwise(results):
    assert d.verdict(results)["overall"] == "fail"
