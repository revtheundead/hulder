"""The scoring connector's scheduler-side logic, with no vLLM or torch, so its tests run on the
base install. connector.py wraps these for vLLM 0.30.0's connector interface (docs/design.md §5 D3).
"""

# The connector's metrics on /metrics, as (name, help). Prometheus exposes each counter with a
# _total suffix, and the histogram as _bucket, _count and _sum.
SCORE_HISTOGRAM = ("hulder_probe_score", "Probe score at the end of prefill.")
COUNTERS = {
    "flagged": ("hulder_probe_flagged", "Scores strictly above the probe's threshold."),
    "replies_with_score": ("hulder_probe_replies_with_score", "Finished requests with a score ready."),
    "replies_without_score": ("hulder_probe_replies_without_score", "Requests that ended before their score."),
    "draft_tokens": ("hulder_draft_tokens", "Draft tokens scheduled, which zero-draft keeps at 0."),
}


def exposed_metric_names() -> set[str]:
    """Every series name the connector puts on /metrics."""
    histogram = SCORE_HISTOGRAM[0]
    return {f"{name}_total" for name, _ in COUNTERS.values()} | {f"{histogram}_{s}" for s in ("bucket", "count", "sum")}


class PrefillTracker:
    """Finds where each request's last prompt token sits in the hidden-state cache, on the step
    that schedules it. A request is reported once, even if it's preempted and prefilled again."""

    def __init__(self, group_id: int, block_size: int):
        self.group_id = group_id
        self.block_size = block_size
        self._prompt_lengths: dict[str, int] = {}
        self._reported: set[str] = set()

    def finished_prefill(self, scheduler_output) -> list[tuple[str, int, int]]:
        """(request id, block id, offset in the block) per request whose prefill ends this step.
        build_connector_meta runs before the scheduler adds this step's tokens to the computed
        count (scheduler.py:1436 and 1455 at 0.30.0), so computed counts exclude them."""
        computed: dict[str, int] = {}
        for request in scheduler_output.scheduled_new_reqs:
            if request.prompt_token_ids is not None:
                self._prompt_lengths[request.req_id] = len(request.prompt_token_ids)
            computed[request.req_id] = request.num_computed_tokens
        cached = scheduler_output.scheduled_cached_reqs
        computed.update(zip(cached.req_ids, cached.num_computed_tokens))

        finished = []
        for req_id, before in computed.items():
            length = self._prompt_lengths.get(req_id)
            if length is None or req_id in self._reported:
                continue
            if before + scheduler_output.num_scheduled_tokens.get(req_id, 0) < length:
                continue
            self._reported.add(req_id)
            block_ids = scheduler_output.kv_connector_block_state.get_block_ids(req_id)[self.group_id]
            block, offset = divmod(length - 1, self.block_size)
            finished.append((req_id, block_ids[block], offset))
        return finished

    def reported(self, req_id: str) -> bool:
        return req_id in self._reported

    def forget(self, req_id: str) -> None:
        self._prompt_lengths.pop(req_id, None)
        self._reported.discard(req_id)


class ScoreLedger:
    """What a finished reply carries, and the counts /metrics reads. Scores arrive with engine
    steps, so a reply can end first. It then goes without one, and the late score still counts
    towards the distribution, since the probe did read that request."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._held: dict[str, float] = {}
        self._awaiting_late_score: set[str] = set()
        self._reset()

    def _reset(self) -> None:
        self._scores: list[float] = []
        self._flagged = 0
        self._with_score = 0
        self._without_score = 0

    def score_arrived(self, req_id: str, score: float) -> None:
        self._scores.append(score)
        self._flagged += score > self.threshold
        if req_id in self._awaiting_late_score:
            self._awaiting_late_score.discard(req_id)
        else:
            self._held[req_id] = score

    def reply_finished(self, req_id: str, score_expected: bool = True) -> float | None:
        score = self._held.pop(req_id, None)
        if score is not None:
            self._with_score += 1
            return score
        self._without_score += 1
        if score_expected:
            self._awaiting_late_score.add(req_id)
        return None

    def drain(self) -> dict:
        counts = {
            "scores": self._scores,
            "flagged": self._flagged,
            "replies_with_score": self._with_score,
            "replies_without_score": self._without_score,
        }
        self._reset()
        return counts


class DraftWatch:
    """Counts the draft tokens each step schedules, which zero-draft keeps at 0. vLLM 0.30.0 drops
    the schedule without an error when async scheduling is off or data_parallel_size is above 1, and
    each decode step then carries a drafted token again. vLLM's own draft counter sums the same
    scheduled_spec_decode_tokens (scheduler.py:1977-1998 at 0.30.0)."""

    WARNING = (
        "hulder.runtime: {n} draft token(s) scheduled in one step, where zero-draft schedules none, "
        "so decode steps carry a drafted token again. With one, the stock connector lost 38.5% of "
        "saturation throughput on an L4. Likely causes: async scheduling off (--no-async-scheduling), "
        "data_parallel_size above 1, or Model Runner V1. Scores are unaffected. Every draft is counted in "
        "hulder_draft_tokens on /metrics. This warning is logged once."
    )

    def __init__(self):
        self._interval = 0
        self._warned = False

    def observe(self, scheduled_spec_decode_tokens: dict[str, list[int]]) -> str | None:
        """Counts one step's drafts, returning the warning on the first step that has any."""
        n = sum(len(tokens) for tokens in scheduled_spec_decode_tokens.values())
        self._interval += n
        if n and not self._warned:
            self._warned = True
            return self.WARNING.format(n=n)
        return None

    def drain(self) -> int:
        n, self._interval = self._interval, 0
        return n
