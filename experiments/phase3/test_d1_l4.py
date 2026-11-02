"""The D1 L4 run's pure parts: the interleaved schedule, the cost rule's verdicts, the generation
stop rule and the startup facts read from a server log. CPU only."""

import json
import os
import sys
import time

import d1_l4 as r
import d1_local
import numpy as np
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


def test_the_tie_bound_is_two_bf16_logit_steps():
    assert r.BF16_LOGIT_STEP == 2**-3  # bf16's spacing at |logit| 16 to 32
    assert r.TIE_STEPS == 2
    assert r.TIE_BOUND == 0.25


@pytest.mark.parametrize(
    "margin, expected",
    [
        (0.0, "exact tie"),
        (0.125, "within bound"),  # one step
        (0.24999994039535522, "within bound"),  # two steps less float32 rounding: the L4's prompt 13
        (0.25, "within bound"),  # two steps exactly, inclusive
        (0.25 + 5e-7, "within bound"),  # two steps plus float32 rounding, inside the 1e-6 tolerance
        (0.25 + 2e-6, "change"),
        (0.375, "change"),  # three steps
        (None, "change"),  # a reply ended early
    ],
)
def test_a_divergence_within_two_bf16_steps_up_to_float_rounding_is_within_bound(margin, expected):
    assert r.divergence_class(margin) == expected


def test_generation_is_undecided_before_any_plain_against_plain_comparison():
    assert r.generation_fails(diverged(2 * r.TIE_BOUND), plain_vs_plain=[]) is None


LOG = """\
(EngineCore pid=1) INFO 10-30 14:40:18 [kv_cache_utils.py:1234] GPU KV cache size: 53,776 tokens
(EngineCore pid=1) INFO 10-30 14:40:18 [kv_cache_utils.py:2397] Maximum concurrency for 2,048 tokens per request: 26.26x
"""


def test_startup_facts_reads_kv_cache_size_and_maximum_concurrency():
    assert r.startup_facts(LOG) | {"model_runner": None, "scheduled_token_budget": None, "async_scheduling": None} == {
        "kv_cache_tokens": 53776,
        "max_concurrency": 26.26,
        "model_runner": None,
        "scheduled_token_budget": None,
        "async_scheduling": None,
    }
    assert r.startup_facts("")["kv_cache_tokens"] is None
    assert r.startup_facts("")["max_concurrency"] is None


V2_LINE = "(EngineCore pid=33) INFO 10-31 15:26:45 [gpu_worker.py:441] Using V2 Model Runner\n"


def test_startup_facts_reads_the_model_runner_v2_from_its_log_line_and_v1_from_its_absence():
    # vLLM 0.30.0 logs only the V2 runner (gpu_worker.py:441). A server that sized its KV cache
    # without that line ran V1, as the L4 run's lens arm did.
    # The KV-cache line comes after the runner is chosen, so without it the runner is unknown.
    assert r.startup_facts(V2_LINE + LOG)["model_runner"] == "V2"
    assert r.startup_facts(LOG)["model_runner"] == "V1"
    assert r.startup_facts(V2_LINE)["model_runner"] == "V2"
    assert r.startup_facts("")["model_runner"] is None


ARGS_LINE = "(APIServer pid=1) INFO 11-01 23:06:23 [api_utils.py:286] non-default args: {'async_scheduling': %s}\n"
DISABLED_LINE = (
    "(APIServer pid=1) WARNING 11-01 23:06:23 [vllm.py:1458] Async scheduling not supported with "
    "foo-based speculative decoding and will be disabled.\n"
)


def test_startup_facts_reads_async_scheduling_from_the_args_a_disabling_warning_or_the_default():
    # vLLM 0.30.0 logs no line for the resolved value. It's on unless the flag or a warning says
    # otherwise (config/vllm.py:1438-1487). Before the engine has started, it's unknown.
    assert r.startup_facts(LOG)["async_scheduling"] is True
    assert r.startup_facts(ARGS_LINE % "False" + LOG)["async_scheduling"] is False
    assert r.startup_facts(ARGS_LINE % "True" + LOG)["async_scheduling"] is True
    assert r.startup_facts(DISABLED_LINE + LOG)["async_scheduling"] is False
    assert r.startup_facts("")["async_scheduling"] is None


def test_async_scheduling_lists_every_arm_run_by_name_for_the_manifest():
    results = {
        "runs": [
            {"arm": "baseline", "round": 0, "startup": {"async_scheduling": True}},
            {"arm": "b_zero_draft", "round": 0, "startup": {"async_scheduling": False}},
            {"arm": "b_stock", "round": 1, "error": "the server didn't answer /health"},
        ]
    }
    assert r.async_scheduling(results) == {"r0_baseline": True, "r0_b_zero_draft": False, "r1_b_stock": None}


def test_model_runners_lists_each_arms_runners_across_its_arm_runs():
    results = {
        "runs": [
            {"arm": "baseline", "startup": {"model_runner": "V2"}},
            {"arm": "lens", "startup": {"model_runner": "V1"}},
            {"arm": "baseline", "startup": {"model_runner": "V2"}},
            {"arm": "b_stock", "error": "the server didn't answer /health"},
        ]
    }
    assert r.model_runners(results) == {"baseline": ["V2"], "lens": ["V1"]}


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

    def run_arm(arm, rnd, cfg, workload, out, rate, timeouts, calibrate=False, score=False):
        run = {"arm": arm, "round": rnd, "generate": ["reply"], "saturation": bench_result(), "fixed_rate": 7.0}
        if (rnd, arm) in misbehaving:
            return {**run, **misbehaving[(rnd, arm)]}
        return {**run, "fixed": bench_result()}

    monkeypatch.setattr(r, "run_arm", run_arm)
    return r.run_all({}, {}, tmp_path, r.Experiment(r.ARMS, rounds), timeouts=r.Timeouts(480, 270, 240))


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
        ],
        "generation": {"r0_b": diverged(0.0, 0.25 + 5e-7), "r1_baseline": diverged()},
    }
    rejudged = r.rejudge(results)
    assert rejudged["divergences"]["counts"]["r0_b"] == {"exact tie": 1, "within bound": 1, "change": 0}
    assert rejudged["divergences"]["tie_bound"] == {"steps": 2, "step": 0.125, "tolerance": 1e-6}
    v = rejudged["verdicts"]
    assert v["b"]["throughput"]["overheads"] == [pytest.approx(0.4), pytest.approx(0.4)]
    assert v["b"]["passes"] is False
    assert v["lens"] == {"measurement_timeouts": [{"round": 0, **lens_timeout}], "passes": False}


BUDGET_LINE = (
    "(EngineCore pid=1) WARNING 11-01 10:00:00 [vllm.py:2135] max_num_scheduled_tokens is set to 2048 based on"
    " the speculative decoding settings. This may lead to suboptimal performance.\n"
)


def test_startup_facts_reads_the_scheduled_token_budget_only_when_vllm_logs_it():
    # vLLM 0.30.0 logs it only under a speculative config (config/vllm.py:2133), so a server without
    # one records None, not its budget.
    assert r.startup_facts(BUDGET_LINE + LOG)["scheduled_token_budget"] == 2048
    assert r.startup_facts(LOG)["scheduled_token_budget"] is None


def test_the_zero_draft_arm_is_the_stock_connector_with_a_zero_token_draft_schedule(tmp_path):
    cfg = {"model": "m", "serve": [], "layer": 18}
    command, env = r.arm_command("b_zero_draft", cfg, tmp_path)
    stock, _ = r.arm_command("b_stock", cfg, tmp_path)
    speculative = json.loads(command[command.index("--speculative-config") + 1])
    assert speculative.pop("num_speculative_tokens_per_batch_size") == [[1, 1, 0]]
    assert speculative == json.loads(stock[stock.index("--speculative-config") + 1])
    transfer = command[command.index("--kv-transfer-config") + 1]
    assert transfer == stock[stock.index("--kv-transfer-config") + 1]
    assert json.loads(transfer)["kv_connector"] == "ExampleHiddenStatesConnector"
    assert env == {"VLLM_LENS_DISABLE": "1"}


def test_the_fixed_rate_is_70_percent_of_the_slowest_arms_saturation():
    assert r.fixed_rate([20.0, 12.0]) == pytest.approx(8.4)
    assert r.fixed_rate([20.0, 20.0]) == pytest.approx(14.0)


def test_the_zero_draft_experiment_alternates_the_two_arms_over_a_calibration_round_and_three_more():
    assert r.ZERO_DRAFT.calibration_rounds == 1
    assert r.schedule(r.ZERO_DRAFT.arms, r.ZERO_DRAFT.rounds) == [
        (0, "baseline"), (0, "b_zero_draft"),
        (1, "b_zero_draft"), (1, "baseline"),
        (2, "baseline"), (2, "b_zero_draft"),
        (3, "b_zero_draft"), (3, "baseline"),
    ]  # fmt: skip


def test_the_zero_draft_worst_case_is_the_drafts_two_cold_and_six_warm_arm_runs():
    timeouts = r.Timeouts(cold_startup=480, warm_startup=270, measure=240)
    assert r.worst_case_seconds(timeouts, r.ZERO_DRAFT.rounds, r.ZERO_DRAFT.arms) == 2 * 750 + 6 * 540


def test_verdicts_pair_throughput_over_every_round_and_latency_over_the_rounds_that_measured_it():
    def calibration(throughput):
        return {"saturation": run(throughput, 0, 0)}

    def measured(throughput, ttft):
        return {"saturation": run(throughput, 0, 0), "fixed": run(7, ttft, 20.0)}

    runs = {
        (0, "baseline"): calibration(10.0),
        (0, "b"): calibration(9.9),
        (1, "baseline"): measured(10.0, 100.0),
        (1, "b"): measured(9.8, 102.0),
        (2, "baseline"): measured(10.0, 100.0),
        (2, "b"): measured(9.7, 104.0),
    }
    v = r.verdicts(runs)["b"]
    assert v["throughput"]["overheads"] == [pytest.approx(0.01), pytest.approx(0.02), pytest.approx(0.03)]
    assert v["p99_ttft"]["overheads"] == [pytest.approx(0.02), pytest.approx(0.04)]
    assert v["passes"] is True


def calibrated_run_all(monkeypatch, tmp_path, misbehaving=()):
    """run_all over a fake run_arm on the zero-draft experiment, returning its results and calls.
    misbehaving maps (round, arm) to fields merged into that arm-run's result."""
    monkeypatch.setattr(r.d1_local, "compare_generations", lambda a, b: {"diverged": []})
    calls = []
    saturation = {"baseline": 20.0, "b_zero_draft": 12.0}
    misbehaving = dict(misbehaving)

    def run_arm(arm, rnd, cfg, workload, out, rate, timeouts, calibrate=False, score=False):
        calls.append({"arm": arm, "round": rnd, "rate": rate, "calibrate": calibrate})
        result = {
            "arm": arm,
            "round": rnd,
            "generate": ["reply"],
            "saturation": bench_result(saturation[arm]),
            "counters": {"draft_tokens": 0},
            "startup": {"model_runner": "V2"},
        }
        if not calibrate:
            result |= {"fixed": bench_result(), "fixed_rate": rate}
        return {**result, **misbehaving.get((rnd, arm), {})}

    monkeypatch.setattr(r, "run_arm", run_arm)
    return r.run_all({}, {}, tmp_path, r.ZERO_DRAFT, r.Timeouts(480, 270, 240)), calls


def test_the_calibration_round_runs_no_fixed_rate_and_sets_the_rate_from_the_slower_arm(monkeypatch, tmp_path):
    results, calls = calibrated_run_all(monkeypatch, tmp_path)
    assert results["stopped"] is None
    assert [c["calibrate"] for c in calls] == [True, True] + [False] * 6
    assert [c["rate"] for c in calls[:2]] == [None, None]
    assert [c["rate"] for c in calls[2:]] == [pytest.approx(8.4)] * 6
    assert results["fixed_rate"] == pytest.approx(8.4)
    assert "fixed" not in results["runs"][0]
    v = results["verdicts"]["b_zero_draft"]
    assert len(v["throughput"]["overheads"]) == 4
    assert len(v["p99_ttft"]["overheads"]) == 3


@pytest.mark.parametrize(
    "fields, reason",
    [
        ({"startup": {"model_runner": "V1"}}, "startup log isn't V2"),
        ({"startup": {"model_runner": None}}, "startup log isn't V2"),
        ({"counters": {"draft_tokens": 3}}, "3 draft tokens"),
    ],
)
def test_a_zero_draft_arm_run_off_v2_or_with_draft_tokens_stops_the_run(monkeypatch, tmp_path, fields, reason):
    for key in ((0, "b_zero_draft"), (2, "b_zero_draft")):
        results, calls = calibrated_run_all(monkeypatch, tmp_path, {key: fields})
        assert reason in results["stopped"]
        assert results["stopped"].startswith(f"round {key[0]} b_zero_draft")
        assert (calls[-1]["round"], calls[-1]["arm"]) == key


def test_the_zero_draft_stop_rules_dont_apply_to_the_baseline(monkeypatch, tmp_path):
    fields = {"startup": {"model_runner": "V1"}, "counters": {"draft_tokens": 3}}
    results, _ = calibrated_run_all(monkeypatch, tmp_path, {(1, "baseline"): fields})
    assert results["stopped"] is None


def test_rejudge_reads_a_run_with_a_calibration_round():
    results = {
        "runs": [
            {"round": 0, "arm": "baseline", "saturation": run(10.0, 0, 0)},
            {"round": 0, "arm": "b", "saturation": run(9.0, 0, 0)},
            {"round": 1, "arm": "baseline", "saturation": run(10.0, 0, 0), "fixed": run(7, 100.0, 20.0)},
            {"round": 1, "arm": "b", "saturation": run(9.0, 0, 0), "fixed": run(7, 100.0, 20.0)},
        ],
    }
    v = r.rejudge(results)["verdicts"]["b"]
    assert len(v["throughput"]["overheads"]) == 2
    assert len(v["p99_ttft"]["overheads"]) == 1


class FakeVolume:
    """A Modal Volume: read_file sees only what commit has published, like another container."""

    def __init__(self, root):
        self.root, self.committed = root, {}

    def commit(self):
        self.committed = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}

    def read_file(self, path):
        if path not in self.committed:
            raise FileNotFoundError(path)
        yield self.committed[path]


PRINTS_ENV = [sys.executable, "-c", "print('PyTorch version: 2.13.0')"]


def in_container(tmp_path, volume, command=PRINTS_ENV, run_id="R1"):
    return r.in_container(
        {}, {}, tmp_path / "out", r.ZERO_DRAFT, r.Timeouts(270, 270, 240),
        volume.root / r.environment_path(run_id).parent, volume.commit, command=command,
    )  # fmt: skip


def test_the_environment_record_exists_before_the_first_arm_starts(monkeypatch, tmp_path):
    seen = {}

    def run_all(cfg, workload, out, experiment, timeouts):
        seen["text"] = (out / r.COLLECT_ENV_FILE).read_text()
        return {"stopped": None}

    monkeypatch.setattr(r, "run_all", run_all)
    in_container(tmp_path, FakeVolume(tmp_path / "vol"))
    assert "PyTorch version: 2.13.0" in seen["text"]


def test_the_environment_record_is_returned_when_an_arm_hits_a_stop_rule(monkeypatch, tmp_path):
    monkeypatch.setattr(r, "run_all", lambda *a: {"stopped": "round 1 b_stock: 3 failed requests"})
    returned = in_container(tmp_path, FakeVolume(tmp_path / "vol"))
    assert returned["results"]["stopped"].startswith("round 1")
    assert b"PyTorch version" in returned["files"][r.COLLECT_ENV_FILE]
    assert returned["environment"]["returncode"] == 0


def test_a_crash_after_startup_leaves_the_record_on_the_volume_for_the_failure_path(monkeypatch, tmp_path):
    def crashes(*args):
        raise MemoryError("the container is going down")

    monkeypatch.setattr(r, "run_all", crashes)
    volume = FakeVolume(tmp_path / "vol")
    with pytest.raises(MemoryError):
        in_container(tmp_path, volume)
    run_dir = tmp_path / "local"
    run_dir.mkdir()
    assert r.recover_environment(volume.read_file, "R1", run_dir) is True
    assert "PyTorch version" in (run_dir / r.COLLECT_ENV_FILE).read_text()


def test_the_failure_path_reports_a_record_it_cant_read_and_doesnt_raise(tmp_path):
    volume = FakeVolume(tmp_path / "vol")
    assert r.recover_environment(volume.read_file, "never-started", tmp_path) is False
    assert not (tmp_path / r.COLLECT_ENV_FILE).exists()


@pytest.mark.parametrize(
    "command, recorded",
    [
        ([sys.executable, "-c", "print('partial'); raise SystemExit(3)"], "exited 3"),
        (["/nonexistent/python", "-m", "vllm.collect_env"], "couldn't run"),
    ],
)
def test_a_failing_collect_env_writes_the_failure_and_the_run_carries_on(monkeypatch, tmp_path, command, recorded):
    monkeypatch.setattr(r, "run_all", lambda *a: {"stopped": None})
    returned = in_container(tmp_path, FakeVolume(tmp_path / "vol"), command=command)
    assert returned["results"]["stopped"] is None
    assert recorded in returned["files"][r.COLLECT_ENV_FILE].decode()


def test_a_collect_env_past_its_timeout_is_killed_with_its_children_and_noted(tmp_path):
    # A child holding the output open must not hold up the run.
    script = "import subprocess, time; print('partial', flush=True); subprocess.Popen(['sleep', '60']); time.sleep(60)"
    volume = FakeVolume(tmp_path / "vol")
    out = tmp_path / "out"
    out.mkdir()
    started = time.time()
    record = r.record_environment(out, volume.root / "env", volume.commit, [sys.executable, "-c", script], timeout_s=1)
    assert time.time() - started < 10
    assert record["returncode"] is None
    text = (out / r.COLLECT_ENV_FILE).read_text()
    assert "partial" in text and "timed out after 1 s" in text


def test_a_volume_that_cant_commit_doesnt_stop_the_run(tmp_path):
    def fails():
        raise OSError("the Volume is unavailable")

    out = tmp_path / "out"
    out.mkdir()
    record = r.record_environment(out, tmp_path / "vol" / "env", fails, PRINTS_ENV)
    assert record["returncode"] == 0
    assert "the Volume is unavailable" in record["persist_error"]
    assert "PyTorch version" in (out / r.COLLECT_ENV_FILE).read_text()


def test_collect_env_finds_the_images_uv_and_lists_this_environments_packages(tmp_path):
    # Without pip it falls back to `uv pip list`, and crashes if that fails. /.uv isn't on the image's PATH,
    # and with VIRTUAL_ENV unset uv lists the interpreter's packages rather than the venv's.
    script = "import os; print(os.environ['PATH']); print(os.environ['VIRTUAL_ENV'])"
    out = tmp_path / "out"
    out.mkdir()
    r.record_environment(out, tmp_path / "env", lambda: None, [sys.executable, "-c", script])
    path, venv = (out / r.COLLECT_ENV_FILE).read_text().splitlines()
    assert path.split(os.pathsep)[0] == "/.uv"
    assert venv == sys.prefix


def test_the_uv_check_reports_each_environments_exit_code_error_and_package_count(tmp_path, monkeypatch):
    uv = tmp_path / "uv"
    uv.write_text('#!/bin/sh\nif [ -n "$FAKE_OK" ]; then echo a==1; echo b==2; else echo "no cache" >&2; exit 2; fi\n')
    uv.chmod(0o755)
    monkeypatch.setattr(r, "IMAGE_UV_DIR", str(tmp_path))
    out = r.uv_check({"as_is": {}, "with_fix": {"FAKE_OK": "1"}})
    assert out["as_is"] == {"returncode": 2, "packages": 0, "stderr": "no cache"}
    assert out["with_fix"] == {"returncode": 0, "packages": 2, "stderr": ""}


def test_collect_env_fits_in_the_measure_calls_slack_so_the_worst_case_doesnt_move():
    assert r.COLLECT_ENV_TIMEOUT_S == 60
    assert r.MEASURE_SLACK_S == 300
    assert r.COLLECT_ENV_TIMEOUT_S < r.MEASURE_SLACK_S


def test_the_confirmation_run_has_every_arm_in_every_position_over_its_measured_rounds():
    plan = r.schedule(r.CONFIRMATION.arms, r.CONFIRMATION.rounds)
    assert plan == [
        (0, "baseline"), (0, "b_stock"), (0, "b_zero_draft"),
        (1, "b_stock"), (1, "b_zero_draft"), (1, "baseline"),
        (2, "b_zero_draft"), (2, "baseline"), (2, "b_stock"),
        (3, "baseline"), (3, "b_stock"), (3, "b_zero_draft"),
    ]  # fmt: skip
    assert r.CONFIRMATION.calibration_rounds == 1
    assert r.CONFIRMATION.scored_arms == ("b_stock", "b_zero_draft")
    assert r.score_stages(r.CONFIRMATION) == 6


def test_the_confirmation_measure_call_is_the_drafts_7500_seconds():
    # 12 arm-runs at 270 + 240 + 30 s, 6 score stages at 120 s, 300 s of slack.
    assert r.CONFIRMATION_TIMEOUTS == r.Timeouts(cold_startup=270, warm_startup=270, measure=240, score=120)
    assert r.measure_call_seconds(r.CONFIRMATION, r.CONFIRMATION_TIMEOUTS) == 12 * 540 + 6 * 120 + 300


def test_the_earlier_experiments_worst_cases_dont_move():
    timeouts = r.Timeouts(cold_startup=480, warm_startup=270, measure=240)
    assert r.measure_call_seconds(r.COST, timeouts) == 4 * 750 + 8 * 540 + 300
    assert r.measure_call_seconds(r.ZERO_DRAFT, timeouts) == 2 * 750 + 6 * 540 + 300


def test_the_stock_and_zero_draft_throughput_drop_is_paired_by_round_with_no_pass_mark():
    runs = [
        {"round": 0, "arm": "baseline", "saturation": run(20.0, 0, 0)},
        {"round": 0, "arm": "b_stock", "saturation": run(12.0, 0, 0)},
        {"round": 0, "arm": "b_zero_draft", "saturation": run(19.0, 0, 0)},
        {"round": 1, "arm": "b_stock", "saturation": run(10.0, 0, 0)},
        {"round": 1, "arm": "b_zero_draft", "saturation": run(20.0, 0, 0)},
    ]
    assert r.stock_against_zero_draft(runs) == [
        {"round": 0, "b_stock": 12.0, "b_zero_draft": 19.0, "stock_drop": pytest.approx(7 / 19)},
        {"round": 1, "b_stock": 10.0, "b_zero_draft": 20.0, "stock_drop": pytest.approx(0.5)},
    ]


def scored_run_all(monkeypatch, tmp_path, misbehaving=()):
    """run_all over a fake run_arm on the confirmation run, with its calls."""
    monkeypatch.setattr(r.d1_local, "compare_generations", lambda a, b: {"diverged": []})
    calls, misbehaving = [], dict(misbehaving)
    saturation = {"baseline": 20.0, "b_stock": 12.0, "b_zero_draft": 19.0}

    def run_arm(arm, rnd, cfg, workload, out, rate, timeouts, calibrate=False, score=False):
        calls.append({"arm": arm, "round": rnd, "calibrate": calibrate, "score": score})
        drafts = 100 if arm == "b_stock" else 0
        result = {
            "arm": arm,
            "round": rnd,
            "generate": ["reply"],
            "saturation": bench_result(saturation[arm]),
            "counters": {"draft_tokens": drafts},
            "startup": {"model_runner": "V2"},
        }
        if not calibrate:
            result |= {"fixed": bench_result(), "fixed_rate": rate}
        if score:
            result["score"] = {"rows": 2101, "counters": {"draft_tokens": drafts}}
        return {**result, **misbehaving.get((rnd, arm), {})}

    monkeypatch.setattr(r, "run_arm", run_arm)
    return r.run_all({}, {}, tmp_path, r.CONFIRMATION, r.CONFIRMATION_TIMEOUTS), calls


def test_the_score_stage_runs_on_both_b_arms_after_the_calibration_round_and_never_on_the_baseline(
    monkeypatch, tmp_path
):
    results, calls = scored_run_all(monkeypatch, tmp_path)
    assert results["stopped"] is None
    scored = [(c["round"], c["arm"]) for c in calls if c["score"]]
    assert sorted(scored) == [(rnd, arm) for rnd in (1, 2, 3) for arm in ("b_stock", "b_zero_draft")]
    assert results["fixed_rate"] == pytest.approx(8.4)  # 70% of b_stock's 12 req/s, the slowest
    assert [row["round"] for row in results["stock_vs_zero_draft"]] == [0, 1, 2, 3]
    assert set(results["verdicts"]) == {"b_stock", "b_zero_draft"}


@pytest.mark.parametrize(
    "key, fields, reason",
    [
        ((0, "b_stock"), {"counters": {"draft_tokens": 0}}, "b_stock drafted no tokens"),
        ((2, "b_stock"), {"counters": {"draft_tokens": 0}}, "b_stock drafted no tokens"),
        ((2, "b_zero_draft"), {"counters": {"draft_tokens": 5}}, "5 draft tokens"),
        ((2, "b_zero_draft"), {"score": {"counters": {"draft_tokens": 7}}}, "7 draft tokens in its score stage"),
        ((2, "b_zero_draft"), {"startup": {"model_runner": "V1"}}, "startup log isn't V2"),
    ],
)
def test_a_b_arm_that_stops_being_what_it_is_named_stops_the_run(monkeypatch, tmp_path, key, fields, reason):
    results, calls = scored_run_all(monkeypatch, tmp_path, {key: fields})
    assert reason in results["stopped"]
    assert (calls[-1]["round"], calls[-1]["arm"]) == key


def test_the_stock_draft_rule_applies_only_to_a_scored_experiment(monkeypatch, tmp_path):
    # The cost run's fakes and its records carry no counters on b_stock.
    results = fake_run_all(monkeypatch, tmp_path, rounds=1, misbehaving={})
    assert results["stopped"] is None


SCORE_WORKLOAD = {
    "score_ids": ["a", "b", "c", "d"],
    "score_prompts": ["text a", "text b", "text c", "text d"],
    "score_tokens": [[1], [2, 2], [3], [4, 4]],
    "score_output_tokens": 16,
    "score_concurrency": 2,
}
TOKENS_OF = dict(zip(SCORE_WORKLOAD["score_prompts"], SCORE_WORKLOAD["score_tokens"], strict=True))
ROW_OF = {text: i for i, text in enumerate(SCORE_WORKLOAD["score_prompts"])}


def fake_server(monkeypatch, tokens_of=TOKENS_OF, row=lambda i: [float(i), 0.5], delay=0.0, acts_error=None):
    """A fake server: chat_all answers in request order, and each prompt's hidden state is row(its prompt)."""
    seen = {}

    async def chat_all(url, texts, concurrency, **params):
        seen.update(texts=list(texts), concurrency=concurrency, params=params)
        await d1_local.asyncio.sleep(delay)
        return [{"prompt_token_ids": tokens_of[t], "text": t} for t in texts]

    def connector_acts(responses):
        if acts_error:
            raise acts_error
        rows = [row(ROW_OF[response["text"]]) for response in responses]
        return np.array(rows, dtype=np.float16), [1] * len(rows)

    monkeypatch.setattr(r.d1_local, "chat_all", chat_all)
    monkeypatch.setattr(r.d1_local, "connector_acts", connector_acts)
    return seen


def test_the_score_stage_sends_prompts_as_the_capture_templates_them_in_a_seeded_order(monkeypatch, tmp_path):
    seen = fake_server(monkeypatch)
    r.score_stage("url", {"model": "m"}, SCORE_WORKLOAD, 2, tmp_path, 60)
    expected = [SCORE_WORKLOAD["score_prompts"][i] for i in np.random.default_rng(2).permutation(4)]
    assert seen["texts"] == expected
    assert seen["concurrency"] == 2
    assert seen["params"] == {
        "model": "m",
        "max_tokens": 16,
        "min_tokens": 16,
        "ignore_eos": True,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_each_round_sends_the_prompts_in_its_own_order():
    orders = {tuple(np.random.default_rng(rnd).permutation(100)) for rnd in (1, 2, 3)}
    assert len(orders) == 3


def test_the_score_stage_saves_only_the_last_position_rows_back_in_prompt_order(monkeypatch, tmp_path):
    seen = fake_server(monkeypatch)
    facts = r.score_stage("url", {"model": "m"}, SCORE_WORKLOAD, 3, tmp_path, 60)
    assert seen["texts"] != SCORE_WORKLOAD["score_prompts"]  # or this couldn't show the rows being put back
    assert facts["rows"] == 4
    saved = np.load(tmp_path / "score.npz")
    assert saved["ids"].tolist() == ["a", "b", "c", "d"]
    assert saved["activations"].dtype == np.float16
    assert saved["activations"].tolist() == [[0.0, 0.5], [1.0, 0.5], [2.0, 0.5], [3.0, 0.5]]


def test_saved_tokens_that_dont_match_the_expected_ones_stop_the_run(monkeypatch, tmp_path):
    fake_server(monkeypatch, tokens_of={**TOKENS_OF, "text c": [9]})
    with pytest.raises(RuntimeError, match="prompt 2's saved token ids"):
        r.score_stage("url", {"model": "m"}, SCORE_WORKLOAD, 1, tmp_path, 60)
    assert not (tmp_path / "score.npz").exists()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_activation_stops_the_run(monkeypatch, tmp_path, bad):
    fake_server(monkeypatch, row=lambda i: [bad if i == 1 else 0.0, 0.5])
    with pytest.raises(RuntimeError, match="1 rows hold a NaN or infinite activation"):
        r.score_stage("url", {"model": "m"}, SCORE_WORKLOAD, 1, tmp_path, 60)


def test_a_stage_returning_too_few_rows_stops_the_run():
    with pytest.raises(RuntimeError, match="returned 1 of 2 rows"):
        r.score_rows(np.array([0, 1]), [{"prompt_token_ids": [1]}], np.zeros((1, 2)), [[1], [2]])


def test_a_stage_past_its_timeout_raises_the_timeout_the_arm_run_turns_into_a_stop(monkeypatch, tmp_path):
    fake_server(monkeypatch, delay=5.0)
    started = time.time()
    with pytest.raises(TimeoutError):
        r.score_stage("url", {"model": "m"}, SCORE_WORKLOAD, 1, tmp_path, 0.05)
    assert time.time() - started < 2


def test_a_connector_file_holding_other_tokens_stops_the_run_instead_of_crashing_it(monkeypatch, tmp_path):
    # run_arm catches RuntimeError, so its files come back, and last_position raises ValueError.
    fake_server(monkeypatch, acts_error=ValueError("the file's token ids aren't the prompt's"))
    with pytest.raises(RuntimeError, match="doesn't match its prompt"):
        r.score_stage("url", {"model": "m"}, SCORE_WORKLOAD, 1, tmp_path, 60)


class FakeTokenizer:
    """Tokens are each word's length, after a template that shows its arguments."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and add_generation_prompt is True and enable_thinking is False
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [len(word) for word in text.split()]}


SYSTEM_TOKENS = [len(word) for word in d1_local.SYSTEM_PROMPT.split()]


def test_the_preflight_passes_prompts_that_template_to_their_expected_tokens():
    expected = [[*SYSTEM_TOKENS, 3], [*SYSTEM_TOKENS, 2, 2]]
    assert r.check_templated(FakeTokenizer(), ["abc", "ab cd"], expected) == 2


def test_the_preflight_stops_on_a_template_mismatch_before_any_gpu_time():
    with pytest.raises(RuntimeError, match="prompt 1 templates to other tokens"):
        r.check_templated(FakeTokenizer(), ["abc", "ab cd"], [[*SYSTEM_TOKENS, 3], [*SYSTEM_TOKENS, 9]])


def test_the_score_workload_takes_each_prompts_tokens_from_kept_and_refuses_a_missing_one():
    workload = r.score_workload(["a", "b"], ["text a", "text b"], {"a": [1], "b": [2, 3], "other": [9]})
    assert workload == {
        "score_ids": ["a", "b"],
        "score_prompts": ["text a", "text b"],
        "score_tokens": [[1], [2, 3]],
        "score_output_tokens": 16,
        "score_concurrency": 64,
    }
    with pytest.raises(KeyError):
        r.score_workload(["a", "missing"], ["text a", "text b"], {"a": [1]})
