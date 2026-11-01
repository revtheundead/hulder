"""The D1 L4 run's pure parts: the interleaved schedule, the cost rule's verdicts, the generation
stop rule and the startup facts read from a server log. CPU only."""

import sys
import time

import d1_l4 as r
import d1_local
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


def test_a_metric_fails_when_its_smallest_overhead_clears_the_threshold_plus_the_baseline_spread():
    # The L4 run's throughput: about 40% in every round against a 2.64% baseline spread.
    runs = {
        (0, "baseline"): {"saturation": run(20.31, 0, 0), "fixed": run(7, 100.0, 20.0)},
        (1, "baseline"): {"saturation": run(19.79, 0, 0), "fixed": run(7, 100.0, 20.0)},
        (0, "b"): {"saturation": run(11.93, 0, 0), "fixed": run(7, 100.0, 20.0)},
        (1, "b"): {"saturation": run(11.97, 0, 0), "fixed": run(7, 100.0, 20.0)},
    }
    v = r.verdicts(runs, "baseline")["b"]
    assert v["throughput"]["baseline_spread"] > 0.05 / 2
    assert v["throughput"]["verdict"] == "fail"  # 39.5% > 5% + 2.6%, however noisy the baseline
    assert v["passes"] is False


def test_a_metric_within_the_threshold_plus_the_spread_in_some_round_stays_inconclusive():
    runs = {
        (0, "baseline"): {"saturation": run(10.0, 0, 0), "fixed": run(7, 100.0, 20.0)},
        (1, "baseline"): {"saturation": run(10.6, 0, 0), "fixed": run(7, 100.0, 20.0)},
        (0, "b"): {"saturation": run(9.3, 0, 0), "fixed": run(7, 100.0, 20.0)},  # 7%, under 5% + 5.8%
        (1, "b"): {"saturation": run(8.0, 0, 0), "fixed": run(7, 100.0, 20.0)},
    }
    assert r.verdicts(runs, "baseline")["b"]["throughput"]["verdict"] == "inconclusive"


def test_a_failing_metric_fails_the_arm_even_when_another_is_inconclusive():
    runs = {
        (0, "baseline"): {"saturation": run(10.0, 0, 0), "fixed": run(7, 100.0, 20.0)},
        (1, "baseline"): {"saturation": run(10.6, 0, 0), "fixed": run(7, 100.0, 20.0)},
        (0, "b"): {"saturation": run(10.0, 0, 0), "fixed": run(7, 100.0, 30.0)},
        (1, "b"): {"saturation": run(10.6, 0, 0), "fixed": run(7, 100.0, 30.0)},
    }
    v = r.verdicts(runs, "baseline")["b"]
    assert v["throughput"]["verdict"] == "inconclusive"
    assert v["p99_tpot"]["verdict"] == "fail"
    assert v["passes"] is False


def diverged(*margins):
    return {
        "identical": 32 - len(margins),
        "of": 32,
        "diverged": [
            {"prompt": i, "token": 5, "tokens": ["x", "y"], "margins": [m, 0.1]} for i, m in enumerate(margins)
        ],
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


def reply(*tokens_and_margins):
    """A chat response whose every step has two candidates, the picked one ahead by its margin."""
    content = [
        {
            "token": t,
            "logprob": -0.1,
            "top_logprobs": [{"token": t, "logprob": -0.1}, {"token": "~", "logprob": -0.1 - m}],
        }
        for t, m in tokens_and_margins
    ]
    return {"choices": [{"message": {"content": ""}, "logprobs": {"content": content}}]}


def test_a_prompt_counts_once_however_many_tokens_differ_after_its_first_divergence():
    reference = reply(("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0))
    arm = reply(("a", 1.0), ("x", 1.0), ("y", 1.0), ("z", 1.0))
    comparison = d1_local.compare_generations([reference], [arm])
    assert [(d["prompt"], d["token"]) for d in comparison["diverged"]] == [(0, 1)]
    assert r.non_tie_divergences(comparison) == 1


def test_divergence_table_lists_every_first_divergence_with_its_reference_margin_and_class():
    generation = {"r1_baseline": diverged(0.0), "r0_b_stock": diverged(0.1, 0.5, None)}
    table = r.divergence_table(generation)
    assert [(row["run"], row["prompt"], row["reference_margin"], row["class"]) for row in table["rows"]] == [
        ("r0_b_stock", 0, 0.1, "within bound"),
        ("r0_b_stock", 1, 0.5, "change"),
        ("r0_b_stock", 2, None, "change"),
        ("r1_baseline", 0, 0.0, "exact tie"),
    ]
    assert table["counts"] == {
        "r0_b_stock": {"exact tie": 0, "within bound": 1, "change": 2},
        "r1_baseline": {"exact tie": 1, "within bound": 0, "change": 0},
    }


def test_installed_packages_lists_every_distribution_once_as_name_and_version_sorted():
    lines = r.installed_packages().splitlines()
    assert lines == sorted(set(lines), key=str.lower)
    assert any(line.startswith("pytest==") for line in lines)


def test_preflight_fails_clearly_when_the_model_isnt_in_the_hf_cache(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")  # from the vllm extra
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="isn't in the HF cache"):
        r.preflight("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c")


def test_round_0_gets_the_cold_startup_timeout_and_later_rounds_the_warm_one():
    timeouts = r.Timeouts(cold_startup=480, warm_startup=270, measure=240)
    assert [timeouts.startup(rnd) for rnd in (0, 1, 2)] == [480, 270, 270]


def test_the_worst_case_counts_every_arm_run_at_both_its_timeouts_and_the_stop_grace():
    timeouts = r.Timeouts(cold_startup=400, warm_startup=200, measure=100)
    per_round = [400 + 100 + r.STOP_GRACE_S, 200 + 100 + r.STOP_GRACE_S]
    assert r.worst_case_seconds(timeouts, rounds=2) == len(r.ARMS) * sum(per_round)


def test_serve_stops_a_server_that_doesnt_answer_within_its_startup_timeout(tmp_path):
    pytest.importorskip("httpx")
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    started = time.time()
    with pytest.raises(TimeoutError, match="1 s startup timeout"), r.serve(command, {}, tmp_path / "server.log", 1):
        pass
    assert time.time() - started < 10


def progress_log(completed: int, of: int, elapsed: str) -> str:
    bar = f"\r 50%|██ | {completed}/{of} [{elapsed}<00:52,  6.0it/s]\r"
    return f"Starting main benchmark run...\n  0%|   | 0/{of} [00:00<?, ?it/s]{bar}"


def test_a_fixed_rate_bench_that_completed_fewer_than_had_arrived_before_the_grace_did_not_sustain_the_rate():
    # 13.5 req/s for 110 - 30 s is 1,080 arrivals, more than the 900 completed.
    log = progress_log(900, 1215, "01:50")
    timeout = r.measurement_timeout("fixed", stage_seconds=133.0, rate=13.5, bench_log=log)
    assert timeout == {
        "stage": "fixed",
        "stage_seconds": 133.0,
        "bench_seconds": 110,
        "completed": 900,
        "of": 1215,
        "outcome": "did not sustain the fixed rate",
    }


def test_a_fixed_rate_bench_keeping_up_when_measurement_ran_out_is_out_of_time_not_unsustained():
    # The L4 baseline of 20261030T191447Z at its timeout: 883 completed, 527 arrived 30 s before.
    log = progress_log(883, 1620, "01:09")
    timeout = r.measurement_timeout("fixed", stage_seconds=92.0, rate=13.5, bench_log=log)
    assert timeout["outcome"] == "ran out of measurement time in fixed"


def test_a_timeout_outside_the_fixed_rate_bench_or_without_progress_is_out_of_time():
    generating = r.measurement_timeout("generate", stage_seconds=240.0, rate=None, bench_log=None)
    assert generating["outcome"] == "ran out of measurement time in generate"
    assert generating["completed"] is generating["of"] is generating["bench_seconds"] is None
    starting = r.measurement_timeout("fixed", stage_seconds=20.0, rate=13.5, bench_log="Starting main benchmark run...")
    assert starting["outcome"] == "ran out of measurement time in fixed"


def bench_result(throughput=10.0, ttft=100.0, tpot=20.0):
    return {"failed": 0, "request_throughput": throughput, "p99_ttft_ms": ttft, "p99_tpot_ms": tpot}


def fake_run_all(monkeypatch, tmp_path, rounds, misbehaving):
    """run_all over a fake run_arm, where misbehaving maps (round, arm) to the run's error fields."""
    monkeypatch.setattr(r.d1_local, "compare_generations", lambda a, b: {"diverged": []})

    def run_arm(arm, rnd, cfg, workload, out, rate, timeouts):
        run = {"arm": arm, "round": rnd, "generate": ["reply"], "saturation": bench_result(), "fixed_rate": 7.0}
        if (rnd, arm) in misbehaving:
            return {**run, **misbehaving[(rnd, arm)]}
        return {**run, "fixed": bench_result()}

    monkeypatch.setattr(r, "run_arm", run_arm)
    return r.run_all({}, {}, tmp_path, rounds=rounds, timeouts=r.Timeouts(480, 270, 240))


def timed_out(outcome="did not sustain the fixed rate"):
    timeout = {"stage": "fixed", "stage_seconds": 133.0, "bench_seconds": 110, "completed": 900, "of": 1215}
    timeout["outcome"] = outcome
    return {"error": "TimeoutExpired: the fixed-rate bench timed out", "timeout": timeout}


def test_a_lens_measurement_timeout_is_a_lens_result_and_the_other_arm_runs_continue(monkeypatch, tmp_path):
    results = fake_run_all(monkeypatch, tmp_path, rounds=2, misbehaving={(0, "lens"): timed_out()})
    assert results["stopped"] is None
    assert len(results["runs"]) == 2 * len(r.ARMS)
    lens = results["verdicts"]["lens"]
    assert lens["measurement_timeouts"] == [{"round": 0, **timed_out()["timeout"]}]
    assert lens["passes"] is False
    assert results["verdicts"]["b_stock"]["passes"] is True


def test_a_lens_that_ran_out_of_measurement_time_is_left_undecided(monkeypatch, tmp_path):
    misbehaving = {(0, "lens"): timed_out("ran out of measurement time in fixed")}
    results = fake_run_all(monkeypatch, tmp_path, rounds=2, misbehaving=misbehaving)
    assert results["stopped"] is None
    assert results["verdicts"]["lens"]["passes"] is None


def test_a_lens_that_times_out_every_round_still_gets_a_verdict(monkeypatch, tmp_path):
    misbehaving = {(rnd, "lens"): timed_out() for rnd in range(2)}
    results = fake_run_all(monkeypatch, tmp_path, rounds=2, misbehaving=misbehaving)
    assert results["verdicts"]["lens"] == {
        "measurement_timeouts": [{"round": rnd, **timed_out()["timeout"]} for rnd in range(2)],
        "passes": False,
    }


@pytest.mark.parametrize("arm", ["baseline", "b_stock", "b_noop"])
def test_a_measurement_timeout_in_the_baseline_or_b_still_stops_the_run(monkeypatch, tmp_path, arm):
    results = fake_run_all(monkeypatch, tmp_path, rounds=2, misbehaving={(1, arm): timed_out()})
    assert results["stopped"].startswith(f"round 1 {arm}: TimeoutExpired")


def test_a_lens_startup_failure_still_stops_the_run(monkeypatch, tmp_path):
    startup = {"error": "TimeoutError: the server didn't answer /health within its 480 s startup timeout"}
    results = fake_run_all(monkeypatch, tmp_path, rounds=2, misbehaving={(0, "lens"): startup})
    assert results["stopped"].startswith("round 0 lens: TimeoutError")
    assert len(results["runs"]) == len(r.ARMS)


def test_rejudge_recomputes_the_verdicts_from_a_finished_runs_records():
    def record(rnd, arm, throughput):
        return {"round": rnd, "arm": arm, "saturation": run(throughput, 0, 0), "fixed": run(7, 100.0, 20.0)}

    lens_timeout = {"stage": "fixed", "completed": 1, "of": 9, "outcome": r.UNSUSTAINED}
    results = {
        "runs": [
            record(0, "baseline", 10.0),
            record(0, "b", 6.0),
            {"round": 0, "arm": "lens", "saturation": run(2.0, 0, 0), "timeout": lens_timeout, "error": "x"},
            record(1, "baseline", 10.0),
            record(1, "b", 6.0),
        ]
    }
    v = r.rejudge(results)
    assert v["b"]["throughput"]["overheads"] == [pytest.approx(0.4), pytest.approx(0.4)]
    assert v["b"]["passes"] is False
    assert v["lens"] == {"measurement_timeouts": [{"round": 0, **lens_timeout}], "passes": False}
