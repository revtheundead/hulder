"""Snapshot files: one serving config's scorer inputs over a labelled calibration set. Numpy only."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hulder.gate.pairing import PROMPT_IDS_KEY, prompt_ids_sha256
from hulder.gate.validate import example_ids, require_binary_labels


@dataclass(frozen=True)
class Snapshot:
    """Example ids, labels, and either scores (n,) or activations (n, d), in the same order.
    Metadata describes the run, as check's reference_meta and candidate_meta."""

    ids: np.ndarray
    labels: np.ndarray
    scores: np.ndarray | None = None
    activations: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    FORMAT_VERSION = 1

    def __post_init__(self) -> None:
        if (self.scores is None) == (self.activations is None):
            raise ValueError("a snapshot holds exactly one of scores or activations")
        n = len(self.ids)
        if np.ndim(self.ids) != 1:
            raise ValueError(f"ids have shape {np.shape(self.ids)}; pass one id per example")
        example_ids(self.ids, n, "ids")
        if np.shape(self.labels) != (n,):
            raise ValueError(f"labels have shape {np.shape(self.labels)} for {n} ids; pass one label per example")
        if self.scores is not None and np.shape(self.scores) != (n,):
            raise ValueError(f"scores have shape {np.shape(self.scores)} for {n} ids; pass one score per example")
        if self.activations is not None and (np.ndim(self.activations) != 2 or len(self.activations) != n):
            raise ValueError(
                f"activations have shape {np.shape(self.activations)} for {n} ids; pass (n, d), one row per example"
            )

    def save(self, path: str | Path) -> None:
        """Write an .npz of plain arrays plus JSON metadata, with the ids' hash for pairing
        and a content hash that load verifies. Never pickled, so safe to share."""
        require_binary_labels(self.labels)  # before the int8 cast, which would turn 0.7 into 0
        arrays = {
            "ids": np.asarray(self.ids, dtype=str),
            "labels": np.asarray(self.labels, dtype=np.int8),
            **({"scores": self.scores} if self.scores is not None else {"activations": self.activations}),
        }
        metadata = {"format_version": self.FORMAT_VERSION, **self.metadata, PROMPT_IDS_KEY: prompt_ids_sha256(self.ids)}
        np.savez(
            path,
            metadata=np.array(json.dumps(metadata)),
            content_sha256=np.array(_content_sha256(arrays)),
            **arrays,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Snapshot":
        with np.load(path, allow_pickle=False) as f:
            metadata = json.loads(str(f["metadata"]))
            version = metadata.pop("format_version")
            if version != cls.FORMAT_VERSION:
                raise ValueError(f"snapshot file format {version}, this hulder reads {cls.FORMAT_VERSION}")
            arrays = {k: f[k] for k in ("ids", "labels", "scores", "activations") if k in f}
            if str(f["content_sha256"]) != _content_sha256(arrays):
                raise ValueError(f"{path}: the arrays don't match the file's content_sha256; the file is damaged")
            ids = f["ids"]
            if PROMPT_IDS_KEY in metadata and metadata[PROMPT_IDS_KEY] != prompt_ids_sha256(ids):
                raise ValueError(f"{path}: the ids don't match the file's {PROMPT_IDS_KEY}; the file was altered")
            return cls(
                ids=ids,
                labels=f["labels"],
                scores=f.get("scores"),
                activations=f.get("activations"),
                metadata=metadata,
            )


def _content_sha256(arrays: dict[str, np.ndarray]) -> str:
    """Hash of each array's name, dtype, shape and bytes, in name order."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}:{a.dtype.str}:{a.shape};".encode())
        h.update(a.tobytes())
    return h.hexdigest()
