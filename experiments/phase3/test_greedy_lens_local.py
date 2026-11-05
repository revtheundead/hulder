"""The pure part of the local greedy and lens follow-ups. CPU only."""

import greedy_lens_local as g


def reply(*steps):
    content = [{"token": t, "top_logprobs": [{"token": t, "logprob": lp, "bytes": None}]} for t, lp in steps]
    return {"choices": [{"logprobs": {"content": content}}]}


def test_a_step_keeps_its_token_and_candidates_without_the_rest():
    assert g.step_at(reply(("a", -0.5), ("b", -1.0)), 1) == {"token": "b", "top": [{"token": "b", "logprob": -1.0}]}


def test_a_position_past_the_end_of_a_reply_has_no_step():
    assert g.step_at(reply(("a", -0.5)), 1) is None


def test_a_failed_collect_with_a_plain_text_body_counts_as_no_results():
    assert g.parsed_body("Internal Server Error") == {}


def test_a_collect_body_keeps_its_results():
    assert g.parsed_body('{"results": {"a": 1}}') == {"results": {"a": 1}}


def test_the_local_servers_keep_the_run_flags_with_this_gpus_memory_fraction():
    cfg = g.local_cfg(["--logprobs-mode", "processed_logprobs"], utilization=0.9)
    assert "--no-enable-prefix-caching" in cfg["serve"]
    assert cfg["serve"][cfg["serve"].index("--gpu-memory-utilization") + 1] == "0.9"
    assert cfg["serve"][-2:] == ["--logprobs-mode", "processed_logprobs"]
    assert "greedy_params" not in cfg
