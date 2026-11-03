"""The Phase 3 overhead table's run (d1_l4.py): its experiments, arms, stop rules and score stage. CPU only."""

import json

import d1_l4 as r
import numpy as np
import pytest


def test_qwen3_has_no_lens_arm_and_one_calibration_round_plus_three_measured():
    experiment = r.OVERHEAD_QWEN3
    assert experiment.arms == ("baseline", "scoring", "scoring_caching")
    assert (experiment.rounds, experiment.calibration_rounds) == (4, 1)
    assert experiment.scored_arms == ("scoring", "scoring_caching")
    assert len(r.schedule(experiment.arms, experiment.rounds)) == 12
    assert r.score_stages(experiment) == 6


def test_the_small_model_keeps_lens_with_one_calibration_round_plus_two_measured():
    experiment = r.OVERHEAD_15B
    assert experiment.arms == ("baseline", "scoring", "scoring_caching", "lens")
    assert (experiment.rounds, experiment.calibration_rounds) == (3, 1)
    assert experiment.scored_arms == ("scoring", "scoring_caching")
    assert len(r.schedule(experiment.arms, experiment.rounds)) == 12
    assert r.score_stages(experiment) == 4


def test_the_measure_calls_worst_cases_in_seconds():
    # 12 arm-runs at 270 + 240 + 30, 6 stages at 120, and 300 of slack.
    assert r.measure_call_seconds(r.OVERHEAD_QWEN3, r.OVERHEAD_QWEN3_TIMEOUTS) == 6480 + 720 + 300
    # 4 cold arm-runs at 420 + 240 + 30, 8 warm at 540, 4 stages at 120, and 300 of slack.
    assert r.measure_call_seconds(r.OVERHEAD_15B, r.OVERHEAD_15B_TIMEOUTS) == 2760 + 4320 + 480 + 300


CFG = {
    "model": "m",
    "layer": 18,
    "probe_path": "/probes/p.npz",
    "serve": ["--seed", "0", "--no-enable-prefix-caching"],
}


def flags(command):
    return dict(zip(command[::2], command[1::2]))


def test_the_scoring_arm_is_the_scoring_connector_on_zero_draft_with_no_file_write(tmp_path):
    command, env = r.arm_command("scoring", CFG, tmp_path)
    speculative = json.loads(command[command.index("--speculative-config") + 1])
    transfer = json.loads(command[command.index("--kv-transfer-config") + 1])
    assert speculative["num_speculative_tokens_per_batch_size"] == [[1, 1, 0]]
    assert speculative["draft_model_config"]["hf_config"]["eagle_aux_hidden_state_layer_ids"] == [18]
    assert transfer["kv_connector"] == "ScoringHiddenStatesConnector"
    assert transfer["kv_connector_module_path"] == "scoring_connector"
    extra = transfer["kv_connector_extra_config"]
    assert extra["probe_path"] == "/probes/p.npz"
    assert extra["use_synchronization_lock"] is False
    assert not extra.get("save_hidden_states")
    assert "--no-enable-prefix-caching" in command
    assert str(r.HERE) in env["PYTHONPATH"]
    assert env["VLLM_LENS_DISABLE"] == "1"


def test_the_caching_arm_is_the_scoring_arm_with_prefix_caching_on(tmp_path):
    scoring, _ = r.arm_command("scoring", CFG, tmp_path)
    caching, _ = r.arm_command("scoring_caching", CFG, tmp_path)
    assert "--no-enable-prefix-caching" not in caching
    assert "--enable-prefix-caching" in caching
    without = [a for a in scoring if a != "--no-enable-prefix-caching"]
    assert [a for a in caching if a != "--enable-prefix-caching"] == without


def test_the_baseline_keeps_prefix_caching_off(tmp_path):
    command, _ = r.arm_command("baseline", CFG, tmp_path)
    assert "--no-enable-prefix-caching" in command
    assert "--speculative-config" not in command


def scoring_run(arm="scoring", **startup):
    return {
        "arm": arm,
        "startup": {"model_runner": "V2", "async_scheduling": True, **startup},
        "counters": {"draft_tokens": 0},
    }


@pytest.mark.parametrize("arm", ["scoring", "scoring_caching"])
def test_a_scoring_arm_stops_off_v2_with_async_off_or_with_drafts(arm):
    assert r.zero_draft_stop(scoring_run(arm)) is None
    assert "isn't V2" in r.zero_draft_stop(scoring_run(arm, model_runner="V1"))
    assert "async scheduling" in r.zero_draft_stop(scoring_run(arm, async_scheduling=False))
    assert "draft tokens" in r.zero_draft_stop({**scoring_run(arm), "counters": {"draft_tokens": 3}})


def fake_runs(monkeypatch, tmp_path, experiment, saturation, diverged=()):
    monkeypatch.setattr(
        r.d1_local,
        "compare_generations",
        lambda a, b: {
            "diverged": [{"prompt": 0, "token": 1, "tokens": ["a", "b"], "margins": [9.0, 9.0]}]
            if b == ["changed"]
            else []
        },
    )
    rates = []

    def run_arm(arm, rnd, cfg, workload, out, rate, timeouts, calibrate=False, score=False):
        rates.append((rnd, arm, rate))
        bench = {"failed": 0, "request_throughput": saturation[arm], "p99_ttft_ms": 100.0, "p99_tpot_ms": 20.0}
        result = {
            "arm": arm,
            "round": rnd,
            "generate": ["changed"] if arm in diverged else ["reply"],
            "saturation": bench,
            "counters": {"draft_tokens": 0},
            "startup": {"model_runner": "V2", "async_scheduling": True},
        }
        return {**result, **({} if calibrate else {"fixed": bench, "fixed_rate": rate})}

    monkeypatch.setattr(r, "run_arm", run_arm)
    return r.run_all({}, {}, tmp_path, experiment, r.OVERHEAD_15B_TIMEOUTS), rates


def test_the_fixed_rate_leaves_lens_out_since_it_would_set_a_rate_nobody_serves_at(monkeypatch, tmp_path):
    saturation = {"baseline": 20.0, "scoring": 19.0, "scoring_caching": 18.0, "lens": 5.0}
    results, rates = fake_runs(monkeypatch, tmp_path, r.OVERHEAD_15B, saturation)
    assert results["fixed_rate"] == pytest.approx(0.7 * 18.0)
    assert [rate for rnd, _, rate in rates if rnd > 0] == [pytest.approx(0.7 * 18.0)] * 8


def test_a_changed_reply_stops_the_scoring_arm_but_only_gets_reported_with_caching_on(monkeypatch, tmp_path):
    saturation = {"baseline": 20.0, "scoring": 19.0, "scoring_caching": 18.0}
    results, _ = fake_runs(monkeypatch, tmp_path, r.OVERHEAD_QWEN3, saturation, diverged=("scoring_caching",))
    assert results["stopped"] is None
    assert results["generation"]["r0_scoring_caching"]["diverged"]
    results, _ = fake_runs(monkeypatch, tmp_path, r.OVERHEAD_QWEN3, saturation, diverged=("scoring",))
    assert "greedy replies changed" in results["stopped"]


def reply(tokens, score):
    return {"prompt_token_ids": tokens, "kv_transfer_params": {"hulder_score": score}}


EXPECTED = [[1], [2], [3], [4]]
ORDER = np.array([2, 0, 3, 1])


def stage_replies(scores_in_prompt_order):
    return [reply(EXPECTED[i], scores_in_prompt_order[i]) for i in ORDER]


def test_reply_scores_come_back_in_prompt_order_with_nan_for_a_missing_one():
    scores = r.reply_scores(ORDER, stage_replies([0.1, None, 0.3, 0.4]), EXPECTED)
    assert scores[[0, 2, 3]].tolist() == pytest.approx([0.1, 0.3, 0.4])
    assert np.isnan(scores[1])


def test_reply_scores_stop_on_too_few_rows_or_other_tokens():
    with pytest.raises(RuntimeError, match="3 of 4 rows"):
        r.reply_scores(ORDER[:3], stage_replies([0.1, 0.2, 0.3, 0.4])[:3], EXPECTED)
    replies = stage_replies([0.1, 0.2, 0.3, 0.4])
    replies[0]["prompt_token_ids"] = [99]
    with pytest.raises(RuntimeError, match="token ids"):
        r.reply_scores(ORDER, replies, EXPECTED)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_reply_scores_stop_on_a_score_that_isnt_a_probability(bad):
    with pytest.raises(RuntimeError, match="isn't a probability"):
        r.reply_scores(ORDER, stage_replies([0.1, bad, 0.3, 0.4]), EXPECTED)


def scores_with(missing, n=1000):
    return np.array([np.nan] * missing + [0.5] * (n - missing))


def test_the_stage_summary_counts_and_allows_up_to_one_percent_missing():
    assert r.stage_summary(scores_with(10), metrics_scores=1000)["without_score"] == 10
    with pytest.raises(RuntimeError, match="11 of 1000 replies"):
        r.stage_summary(scores_with(11), metrics_scores=1000)


def test_the_stage_summary_stops_on_a_score_lost_on_the_way_to_metrics():
    with pytest.raises(RuntimeError, match="999 scores for 1000 requests"):
        r.stage_summary(scores_with(0), metrics_scores=999)


def test_the_stage_summary_reports_the_shift_over_rows_with_a_score_and_stops_past_the_limit():
    scores = np.array([np.nan, 0.5, 0.6])
    reference = np.array([0.9, 0.4, 0.6])
    summary = r.stage_summary(scores, metrics_scores=3, reference=reference, allowed_missing=1.0)
    assert summary["mean_abs_dscore"] == pytest.approx(0.05)
    assert "mean_abs_dscore" not in r.stage_summary(scores, metrics_scores=3, allowed_missing=1.0)
    with pytest.raises(RuntimeError, match="above its stop rule"):
        r.stage_summary(scores, metrics_scores=3, reference=reference, shift_stop=0.04, allowed_missing=1.0)


def test_the_shift_stop_is_three_times_d1s_highest_zero_draft_stage():
    assert r.SCORE_SHIFT_STOP == pytest.approx(3 * 0.001951430574990809)


METRICS = """\
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries_total{{engine="0",model_name="m"}} {queries}
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits_total{{engine="0",model_name="m"}} {hits}
"""


def test_cache_counters_are_the_difference_between_two_scrapes():
    pytest.importorskip("prometheus_client")  # from the vllm extra
    before, after = METRICS.format(queries=100, hits=16), METRICS.format(queries=500, hits=48)
    assert r.cache_counters(before, after) == {"queried_tokens": 400, "hit_tokens": 32}


def test_the_scoring_arms_stage_only_stops_on_shift_for_the_caching_off_arm():
    workload = {"score_reference": [0.5], "score_shift_stop": 0.01}
    assert r.shift_stop_for("scoring", workload) == 0.01
    assert r.shift_stop_for("scoring_caching", workload) is None
    assert r.shift_stop_for("scoring", {}) is None


def test_the_reply_stage_saves_scores_in_prompt_order_and_sends_prompts_as_the_capture_templates_them(
    monkeypatch, tmp_path
):
    import d3_spike

    sent = {}

    async def chat_all(url, texts, concurrency, **params):
        sent.update(texts=texts, concurrency=concurrency, params=params)
        by_text = {"a": ([1], 0.1), "b": ([2], 0.2), "c": ([3], 0.3)}
        return [reply(*by_text[t]) for t in texts]

    monkeypatch.setattr(r.d1_local, "chat_all", chat_all)
    monkeypatch.setattr(pytest.importorskip("httpx"), "get", lambda *a, **k: type("R", (), {"text": ""}))
    monkeypatch.setattr(d3_spike, "settled_metrics", lambda url, before, expected: {"scores": expected})
    workload = {
        "score_prompts": ["a", "b", "c"],
        "score_ids": ["x", "y", "z"],
        "score_tokens": [[1], [2], [3]],
        "score_output_tokens": 16,
        "score_concurrency": 64,
    }
    result = r.reply_score_stage("url", {"model": "m"}, workload, 1, tmp_path, 60, "scoring")
    assert sent["concurrency"] == 64
    assert sent["params"]["max_tokens"] == sent["params"]["min_tokens"] == 16
    assert sent["params"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert result["rows"] == 3
    assert result["without_score"] == 0
    with np.load(tmp_path / "score.npz", allow_pickle=False) as saved:
        assert saved["ids"].tolist() == ["x", "y", "z"]
        assert saved["scores"].tolist() == pytest.approx([0.1, 0.2, 0.3])
