"""Snapshot files: one serving config's scorer inputs over a labelled calibration set. Numpy only."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hulder.gate.atomic import savez_atomically
from hulder.gate.npz import read_npz
from hulder.gate.pairing import PROMPT_IDS_KEY, PairingError, prompt_ids_sha256
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

    # 2: the content hash covers the metadata too (2026-10-10).
    FORMAT_VERSION = 2

    def __post_init__(self) -> None:
        if (self.scores is None) == (self.activations is None):
            raise ValueError("a snapshot holds exactly one of scores or activations")
        n = len(self.ids)
        if np.ndim(self.ids) != 1:
            raise ValueError(f"ids have shape {np.shape(self.ids)}; pass one id per example")
        example_ids(self.ids, n, "ids")
        recorded = self.metadata.get(PROMPT_IDS_KEY)
        if recorded is not None and recorded != prompt_ids_sha256(self.ids):
            raise PairingError(
                f"The ids don't match the {PROMPT_IDS_KEY} in the metadata: the ids or the metadata belong to another run."
            )
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
        and a content hash that load verifies. Never pickled, so safe to share. Written
        atomically, so an interrupted save leaves any previous file whole. Like np.savez,
        it adds .npz to a path without it."""
        require_binary_labels(self.labels)  # before the int8 cast, which would turn 0.7 into 0
        arrays = {
            "ids": np.asarray(self.ids, dtype=str),
            "labels": np.asarray(self.labels, dtype=np.int8),
            **({"scores": self.scores} if self.scores is not None else {"activations": self.activations}),
        }
        metadata = {"format_version": self.FORMAT_VERSION, **self.metadata, PROMPT_IDS_KEY: prompt_ids_sha256(self.ids)}
        text = json.dumps(metadata)
        savez_atomically(
            path,
            metadata=np.array(text),
            content_sha256=np.array(_content_sha256(arrays, json.loads(text))),
            **arrays,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Snapshot":
        stored, metadata = read_npz(path, "snapshot", ("content_sha256", "ids", "labels"))
        if "format_version" not in metadata:
            raise ValueError(f"{path}: no format_version in its metadata, so it isn't a hulder snapshot")
        if metadata["format_version"] != cls.FORMAT_VERSION:
            raise ValueError(
                f"{path}: snapshot file format {metadata['format_version']}, this hulder reads {cls.FORMAT_VERSION}"
            )
        arrays = {k: stored[k] for k in ("ids", "labels", "scores", "activations") if k in stored}
        if str(stored["content_sha256"]) != _content_sha256(arrays, metadata):
            raise ValueError(
                f"{path}: the arrays or metadata don't match the file's content_sha256; the file is damaged"
            )
        ids = arrays["ids"]
        if PROMPT_IDS_KEY not in metadata:
            raise ValueError(f"{path}: no {PROMPT_IDS_KEY} in its metadata, which save always writes")
        if metadata[PROMPT_IDS_KEY] != prompt_ids_sha256(ids):
            raise ValueError(f"{path}: the ids don't match the file's {PROMPT_IDS_KEY}; the file was altered")
        del metadata["format_version"]
        return cls(
            ids=ids,
            labels=arrays["labels"],
            scores=arrays.get("scores"),
            activations=arrays.get("activations"),
            metadata=metadata,
        )


def _content_sha256(arrays: dict[str, np.ndarray], metadata: dict) -> str:
    """Hash of each array's name, dtype, shape and bytes, in name order, then the metadata
    as canonical JSON (sorted keys, no spaces)."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}:{a.dtype.str}:{a.shape};".encode())
        h.update(a.tobytes())
    h.update(b"metadata;" + json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    return h.hexdigest()
