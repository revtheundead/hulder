"""The D3 spike's pure parts: finding each request's last prompt token at the end of prefill. CPU only."""

from types import SimpleNamespace as NS

import d3_scoring as s

GROUP = 1
BLOCK_SIZE = 16


def step(new=(), cached=(), scheduled=None, blocks=None):
    """A SchedulerOutput as build_connector_meta sees it, before the step's tokens are counted."""
    return NS(
        scheduled_new_reqs=[NS(req_id=r, prompt_token_ids=[0] * n, num_computed_tokens=c) for r, n, c in new],
        scheduled_cached_reqs=NS(req_ids=[r for r, _ in cached], num_computed_tokens=[c for _, c in cached]),
        num_scheduled_tokens=scheduled or {},
        kv_connector_block_state=NS(get_block_ids=lambda req_id: ([100, 101, 102], (blocks or {})[req_id])),
    )


def tracker():
    return s.PrefillTracker(group_id=GROUP, block_size=BLOCK_SIZE)


def test_a_prompt_prefilled_in_one_step_reports_its_last_token_in_the_hidden_state_group():
    # 20 tokens: position 19 is offset 3 of the second block, 9.
    output = step(new=[("a", 20, 0)], scheduled={"a": 20}, blocks={"a": [7, 9]})
    assert tracker().finished_prefill(output) == [("a", 9, 3)]


def test_a_prompt_filling_its_last_block_exactly_reports_that_blocks_last_offset():
    output = step(new=[("a", 32, 0)], scheduled={"a": 32}, blocks={"a": [7, 9]})
    assert tracker().finished_prefill(output) == [("a", 9, 15)]


def test_a_chunked_prefill_reports_only_on_the_step_that_schedules_its_last_token():
    t = tracker()
    assert t.finished_prefill(step(new=[("a", 20, 0)], scheduled={"a": 16}, blocks={"a": [7]})) == []
    assert t.finished_prefill(step(cached=[("a", 16)], scheduled={"a": 4}, blocks={"a": [7, 9]})) == [("a", 9, 3)]


def test_a_prefix_cache_hit_still_reports_the_recomputed_last_token():
    output = step(new=[("a", 20, 16)], scheduled={"a": 4}, blocks={"a": [7, 9]})
    assert tracker().finished_prefill(output) == [("a", 9, 3)]


def test_decode_steps_report_nothing():
    t = tracker()
    t.finished_prefill(step(new=[("a", 20, 0)], scheduled={"a": 20}, blocks={"a": [7, 9]}))
    assert t.finished_prefill(step(cached=[("a", 20)], scheduled={"a": 1}, blocks={"a": [7, 9]})) == []


def test_a_preempted_request_prefilled_again_is_reported_once():
    t = tracker()
    t.finished_prefill(step(new=[("a", 20, 0)], scheduled={"a": 20}, blocks={"a": [7, 9]}))
    assert t.finished_prefill(step(cached=[("a", 0)], scheduled={"a": 20}, blocks={"a": [3, 4]})) == []


def test_a_forgotten_request_leaves_no_state():
    t = tracker()
    t.finished_prefill(step(new=[("a", 20, 0)], scheduled={"a": 20}, blocks={"a": [7, 9]}))
    t.forget("a")
    assert t.finished_prefill(step(new=[("a", 20, 0)], scheduled={"a": 20}, blocks={"a": [7, 9]})) == [("a", 9, 3)]
