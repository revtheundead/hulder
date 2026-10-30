"""The D1 L4 run's pure parts: the interleaved schedule, the cost rule's verdicts, the generation
stop rule and the startup facts read from a server log. CPU only."""

import d1_l4 as r
import pytest


def test_schedule_rotates_the_arms_each_round_starting_with_the_baseline():
    assert r.schedule(("a", "b", "c"), rounds=3) == [
        (0, "a"), (0, "b"), (0, "c"),
        (1, "b"), (1, "c"), (1, "a"),
        (2, "c"), (2, "a"), (2, "b"),
    ]  # fmt: skip


def run(throughput, ttft, tpot):
    return {"request_throughput": throughput, "p99_ttft_ms": ttft, "p99_tpot_ms": tpot}


def test_verdict_takes_the_worst_paired_round_against_each_threshold():
    runs = {
        (0, "baseline"): {"saturation": run(10.0, 0, 0), "fixed": run(7, 100.0, 20.0)},
        (1, "baseline"): {"saturation": run(10.0, 0, 0), "fixed": run(7, 100.0, 20.0)},
        (0, "b"): {"saturation": run(9.7, 0, 0), "fixed": run(7, 105.0, 21.0)},
        (1, "b"): {"saturation": run(9.4, 0, 0), "fixed": run(7, 108.0, 23.0)},
    }
    v = r.verdicts(runs, "baseline")["b"]
    assert v["throughput"]["overheads"] == [pytest.approx(0.03), pytest.approx(0.06)]
    assert v["throughput"]["verdict"] == "fail"  # 6% in round 1, above 5%
    assert v["p99_ttft"]["overheads"] == [pytest.approx(0.05), pytest.approx(0.08)]
    assert v["p99_ttft"]["verdict"] == "pass"
    assert v["p99_tpot"]["verdict"] == "fail"  # 15% in round 1, above 10%
    assert v["passes"] is False


def test_verdict_is_inconclusive_when_the_baseline_varies_by_more_than_half_a_threshold():
    runs = {
        (0, "baseline"): {"saturation": run(10.0, 0, 0), "fixed": run(7, 100.0, 20.0)},
        (1, "baseline"): {"saturation": run(10.6, 0, 0), "fixed": run(7, 100.0, 20.0)},
        (0, "b"): {"saturation": run(10.0, 0, 0), "fixed": run(7, 100.0, 20.0)},
        (1, "b"): {"saturation": run(10.6, 0, 0), "fixed": run(7, 100.0, 20.0)},
    }
    v = r.verdicts(runs, "baseline")["b"]
    assert v["throughput"]["baseline_spread"] == pytest.approx(0.6 / 10.3)
    assert v["throughput"]["verdict"] == "inconclusive"
    assert v["p99_ttft"]["verdict"] == "pass"
    assert v["passes"] is None  # not settled while any metric is inconclusive


def diverged(*margins):
    return {
        "identical": 32 - len(margins),
        "of": 32,
        "diverged": [{"prompt": i, "token": 5, "tokens": ["x", "y"], "margins": [m, 0.1]} for i, m in enumerate(margins)],
    }


def test_generation_fails_only_on_more_non_tie_divergences_than_plain_against_plain():
    tie = r.TIE_BOUND
    assert r.generation_fails(diverged(0.0, tie), plain_vs_plain=[diverged()]) is False  # ties only
    assert r.generation_fails(diverged(2 * tie), plain_vs_plain=[diverged()]) is True
    assert r.generation_fails(diverged(2 * tie), plain_vs_plain=[diverged(), diverged(3 * tie)]) is False
    assert r.generation_fails(diverged(None), plain_vs_plain=[diverged()]) is True  # a reply ended early


def test_generation_is_undecided_before_any_plain_against_plain_comparison():
    assert r.generation_fails(diverged(2 * r.TIE_BOUND), plain_vs_plain=[]) is None


LOG = """\
(EngineCore pid=1) INFO 10-30 14:40:18 [kv_cache_utils.py:1234] GPU KV cache size: 53,776 tokens
(EngineCore pid=1) INFO 10-30 14:40:18 [kv_cache_utils.py:2397] Maximum concurrency for 2,048 tokens per request: 26.26x
"""


def test_startup_facts_reads_kv_cache_size_and_maximum_concurrency():
    assert r.startup_facts(LOG) == {"kv_cache_tokens": 53776, "max_concurrency": 26.26}
    assert r.startup_facts("") == {"kv_cache_tokens": None, "max_concurrency": None}
