"""Snapshot files: one serving config's scorer inputs over a labelled calibration set. Numpy only."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hulder.gate.check import PROMPT_IDS_KEY, prompt_ids_sha256


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

    def save(self, path: str | Path) -> None:
        """Write an .npz of plain arrays plus JSON metadata, with the ids' hash for pairing.
        Never pickled, so safe to share."""
        values = {"scores": self.scores} if self.scores is not None else {"activations": self.activations}
        metadata = {"format_version": self.FORMAT_VERSION, **self.metadata, PROMPT_IDS_KEY: prompt_ids_sha256(self.ids)}
        np.savez(
            path,
            ids=np.asarray(self.ids, dtype=str),
            labels=np.asarray(self.labels, dtype=np.int8),
            metadata=np.array(json.dumps(metadata)),
            **values,
        )

    @classmethod
    def load(cls, path: str | Path) -> "Snapshot":
        with np.load(path, allow_pickle=False) as f:
            metadata = json.loads(str(f["metadata"]))
            version = metadata.pop("format_version")
            if version != cls.FORMAT_VERSION:
                raise ValueError(f"snapshot file format {version}, this hulder reads {cls.FORMAT_VERSION}")
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
