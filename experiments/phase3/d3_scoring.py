"""The D3 spike's scheduler-side logic, with no vLLM or torch, so its tests run on the base install.

scoring_connector.py wraps these for vLLM 0.30.0's connector interface (docs/design.md §5 D3).
"""


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
