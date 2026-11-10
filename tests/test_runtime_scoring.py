"""hulder.runtime's scheduler-side parts: finding each request's last prompt token at the end of
prefill, and what each reply carries. CPU only, on the base install."""

from types import SimpleNamespace as NS

import hulder.runtime.scoring as s

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


def ledger():
    return s.ScoreLedger(threshold=0.5)


def test_a_score_that_arrives_before_the_reply_ends_goes_in_the_reply():
    book = ledger()
    book.score_arrived("a", 0.7)
    assert book.reply_finished("a") == 0.7
    assert book.drain() == {"scores": [0.7], "flagged": 1, "replies_with_score": 1, "replies_without_score": 0}


def test_a_reply_that_ends_first_comes_back_without_a_score_and_is_counted_but_the_late_score_still_counts():
    book = ledger()
    assert book.reply_finished("a") is None
    book.score_arrived("a", 0.2)
    assert book.drain() == {"scores": [0.2], "flagged": 0, "replies_with_score": 0, "replies_without_score": 1}


def test_a_late_score_is_not_held_for_a_later_request_with_the_same_id():
    book = ledger()
    book.reply_finished("a")
    book.score_arrived("a", 0.2)
    assert book.reply_finished("a") is None


def test_flagging_is_strictly_above_the_threshold():
    book = ledger()
    book.score_arrived("a", 0.5)
    assert book.drain()["flagged"] == 0


def test_drain_starts_a_new_interval():
    book = ledger()
    book.score_arrived("a", 0.7)
    book.drain()
    assert book.drain() == {"scores": [], "flagged": 0, "replies_with_score": 0, "replies_without_score": 0}


def test_a_request_that_never_finished_prefill_leaves_nothing_waiting_for_a_score():
    book = ledger()
    assert book.reply_finished("a", score_expected=False) is None
    book.score_arrived("a", 0.7)  # can't happen for this request, so it would be held if it did
    assert book.reply_finished("a") == 0.7
    assert book.drain()["replies_without_score"] == 1


def test_the_tracker_says_whether_a_request_reached_the_end_of_prefill():
    t = tracker()
    t.finished_prefill(
        step(new=[("a", 20, 0), ("b", 20, 0)], scheduled={"a": 20, "b": 16}, blocks={"a": [7, 9], "b": [5]})
    )
    assert t.reported("a") and not t.reported("b")


def test_no_drafts_under_zero_draft_means_no_warning_and_nothing_counted():
    watch = s.DraftWatch()
    assert watch.observe({}) is None
    assert watch.observe({"a": []}) is None
    assert watch.drain() == 0


def test_the_first_drafted_step_warns_once_with_its_count_and_every_draft_is_counted():
    watch = s.DraftWatch()
    warning = watch.observe({"a": [5], "b": [-1]})
    assert "2 draft tokens" in warning
    assert "--no-async-scheduling" in warning and "data_parallel_size" in warning
    assert watch.observe({"a": [7]}) is None
    assert watch.drain() == 3
    assert watch.drain() == 0
