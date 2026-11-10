"""The runtime's connector: vLLM 0.30.0's ExampleHiddenStatesConnector, scoring a linear probe at
the end of prefill (docs/design.md §5 D3).

On the step that schedules a request's last prompt token, the scheduler side tells the worker
where that token's hidden state will be. The worker queues the probe on its copy stream, with no
host sync, and polls the result on later steps. Scores come back through worker metadata, into a
non-streamed reply's kv_transfer_params as hulder_score, and to /metrics. A reply that ends first
goes without one, and is counted. Nothing is written to disk.

Loaded by `vllm serve` through kv_connector_module_path, with kv_connector_extra_config's
probe_path. hulder.runtime.serve_args builds those arguments. Imports vLLM and torch.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole, KVConnectorWorkerMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    ExampleHiddenStatesConnector,
    ExampleHiddenStatesConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorPromMetrics, KVConnectorStats
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.metrics.utils import create_metric_per_engine

from hulder import Probe
from hulder.runtime.scoring import DraftWatch, PrefillTracker, ScoreLedger

logger = logging.getLogger(__name__)

SCORE_BUCKETS = [i / 20 for i in range(1, 20)]


def probe_score(hidden, mean, std, weights, bias: float):
    """hulder.apply_probe in torch, in float32, for a (n, d) batch on any device."""
    return torch.sigmoid(((hidden.float() - mean) / std) @ weights + bias)


@dataclass
class ScoresWorkerMeta(KVConnectorWorkerMetadata):
    scores: dict[str, float]

    def aggregate(self, other: "ScoresWorkerMeta") -> "ScoresWorkerMeta":
        return ScoresWorkerMeta({**self.scores, **other.scores})


@dataclass
class ScoringMetadata(ExampleHiddenStatesConnectorMetadata):
    # (request id, block id, offset in the block) of each last prompt token this step.
    to_score: list[tuple[str, int, int]] = field(default_factory=list)


class ScoreStats(KVConnectorStats):
    """ScoreLedger.drain()'s counts and DraftWatch.drain()'s drafts, for the /metrics and log paths."""

    COUNTS = ("flagged", "replies_with_score", "replies_without_score", "draft_tokens")

    def reset(self):
        self.data = {"scores": [], **dict.fromkeys(self.COUNTS, 0)}

    def aggregate(self, other: "ScoreStats") -> "ScoreStats":
        return ScoreStats(
            {
                "scores": self.data["scores"] + other.data["scores"],
                **{key: self.data[key] + other.data[key] for key in self.COUNTS},
            }
        )

    def reduce(self) -> dict[str, int | float]:
        scores = self.data["scores"]
        return {
            "scores": len(scores),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            **{key: self.data[key] for key in self.COUNTS},
        }

    def is_empty(self) -> bool:
        reduced = self.reduce()
        return not any(reduced[k] for k in ("scores", "replies_with_score", "replies_without_score", "draft_tokens"))


class ScorePromMetrics(KVConnectorPromMetrics):
    def __init__(self, vllm_config, metric_types, labelnames, per_engine_labelvalues):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        self.scores = create_metric_per_engine(
            self._histogram_cls(
                name="hulder_probe_score",
                documentation="Probe score at the end of prefill.",
                buckets=SCORE_BUCKETS,
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )
        self.counters = {
            key: create_metric_per_engine(
                self._counter_cls(name=name, documentation=documentation, labelnames=labelnames),
                per_engine_labelvalues,
            )
            for key, name, documentation in (
                ("flagged", "hulder_probe_flagged", "Scores strictly above the probe's threshold."),
                ("replies_with_score", "hulder_probe_replies_with_score", "Finished requests with a score ready."),
                (
                    "replies_without_score",
                    "hulder_probe_replies_without_score",
                    "Requests that ended before their score.",
                ),
                ("draft_tokens", "hulder_draft_tokens", "Draft tokens scheduled, which zero-draft keeps at 0."),
            )
        }

    def observe(self, transfer_stats_data: dict, engine_idx: int = 0):
        for score in transfer_stats_data["scores"]:
            self.scores[engine_idx].observe(score)
        for key, counter in self.counters.items():
            counter[engine_idx].inc(transfer_stats_data[key])


class ScoringHiddenStatesConnector(ExampleHiddenStatesConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config.get_from_extra_config
        self._probe = Probe.load(extra("probe_path", None))

        # Scheduler side.
        self._tracker = PrefillTracker(self._cache_kv_group_id, self._block_size)
        self._ledger = ScoreLedger(float(self._probe.metadata["threshold"]))
        self._drafts = DraftWatch()

        # Worker side.
        self._probe_tensors = None
        self._pending_scores: list[tuple[list[str], torch.Tensor, torch.cuda.Event]] = []
        self._ready_scores: dict[str, float] = {}
        self._scoring: set[str] = set()
        self._finished_awaiting_score: set[str] = set()

    # ==============================
    # Worker-side methods
    # ==============================
    def register_kv_caches(self, kv_caches):
        super().register_kv_caches(kv_caches)
        device = self._kv_cache.device
        self._probe_tensors = tuple(
            torch.from_numpy(a.astype(np.float32)).to(device)
            for a in (self._probe.mean, self._probe.std, self._probe.weights)
        )

    @torch.inference_mode()
    def _submit_scores(self, to_score: list[tuple[str, int, int]]) -> None:
        """The stock connector's stream pattern (example_hidden_states_connector.py:371-397 at
        0.30.0): nothing here waits on the GPU."""
        req_ids = [req_id for req_id, _, _ in to_score]
        blocks = torch.tensor([block for _, block, _ in to_score], dtype=torch.long)
        offsets = torch.tensor([offset for _, _, offset in to_score], dtype=torch.long)

        copy_stream = self._get_copy_stream()
        ready = torch.cuda.Event()
        ready.record()
        copy_stream.wait_event(ready)
        with torch.cuda.stream(copy_stream):
            with gpu_sync_allowed():
                blocks = blocks.to(self._kv_cache.device, non_blocking=True)
                offsets = offsets.to(self._kv_cache.device, non_blocking=True)
            hidden = self._kv_cache[blocks, :, offsets][:, 0]  # [n, layers, d], with one layer
            mean, std, weights = self._probe_tensors
            scores = probe_score(hidden, mean, std, weights, float(self._probe.bias))
            pinned = torch.empty_like(scores, device="cpu", pin_memory=True)
            pinned.copy_(scores, non_blocking=True)
        done = torch.cuda.Event()
        done.record(copy_stream)
        self._pending_scores.append((req_ids, pinned, done))
        self._scoring.update(req_ids)

    def _poll_scores(self) -> None:
        still_pending = []
        for req_ids, pinned, done in self._pending_scores:
            if not done.query():
                still_pending.append((req_ids, pinned, done))
                continue
            self._ready_scores.update(zip(req_ids, pinned.tolist()))
            self._scoring.difference_update(req_ids)
        self._pending_scores = still_pending

    def get_finished(self, finished_req_ids):
        """A request's blocks stay held until its score is read, so the slot can't be reused first."""
        if self.has_connector_metadata():
            metadata = self._get_connector_metadata()
            if isinstance(metadata, ScoringMetadata) and metadata.to_score:
                self._submit_scores(metadata.to_score)
        self._poll_scores()

        done, _ = super().get_finished(finished_req_ids)
        candidates = set(done or ()) | self._finished_awaiting_score
        self._finished_awaiting_score = candidates & self._scoring
        return (candidates - self._scoring) or None, None

    def build_connector_worker_meta(self):
        if not self._ready_scores:
            return None
        ready, self._ready_scores = self._ready_scores, {}
        return ScoresWorkerMeta(ready)

    # ==============================
    # Scheduler-side methods
    # ==============================
    def build_connector_meta(self, scheduler_output):
        stock = super().build_connector_meta(scheduler_output)
        if warning := self._drafts.observe(scheduler_output.scheduled_spec_decode_tokens):
            logger.warning(warning)
        return ScoringMetadata(
            pending_saves=stock.pending_saves,
            new_req_filenames=stock.new_req_filenames,
            to_score=self._tracker.finished_prefill(scheduler_output),
        )

    def update_connector_output(self, connector_output):
        super().update_connector_output(connector_output)
        if connector_output.kv_connector_worker_meta is not None:
            for req_id, score in connector_output.kv_connector_worker_meta.scores.items():
                self._ledger.score_arrived(req_id, score)

    def request_finished(self, request, block_ids):
        req_id = request.request_id
        score = self._ledger.reply_finished(req_id, score_expected=self._tracker.reported(req_id))
        self._tracker.forget(req_id)
        # The stock connector queues a file write here. Skip it, and hold the blocks only while
        # get_finished says the score is still being read.
        self._request_filenames.pop(req_id, None)
        return True, {"hulder_score": score}

    def get_kv_connector_stats(self):
        if self._role != KVConnectorRole.SCHEDULER:
            return None
        stats = ScoreStats({**self._ledger.drain(), "draft_tokens": self._drafts.drain()})
        return None if stats.is_empty() else stats

    @classmethod
    def build_kv_connector_stats(cls, data=None):
        stats = ScoreStats(data or {})
        if data is None:
            stats.reset()
        return stats

    @classmethod
    def build_prom_metrics(cls, vllm_config, metric_types, labelnames, per_engine_labelvalues):
        return ScorePromMetrics(vllm_config, metric_types, labelnames, per_engine_labelvalues)
