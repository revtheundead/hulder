"""Bring-your-own scores: one JSONL file per serving config (docs/design.md §4). Numpy and the standard
library only.

Each line is a JSON object with an "id" (a string or an integer), a "score" (a finite number) and a
"label" (0 benign, 1 attack). Other keys and blank lines are ignored. An optional first line
{"metadata": {...}} describes the run, as a snapshot's metadata does."""

import json
import math
from pathlib import Path

import numpy as np

from hulder.gate.pairing import PairingError
from hulder.gate.snapshot import Snapshot
from hulder.gate.validate import example_ids


def load_scores_jsonl(path: str | Path, order: np.ndarray | None = None) -> Snapshot:
    """The file as a Snapshot of scores. With order, the reference's ids, rows are put in that order,
    and ids on one side only raise PairingError."""
    path = Path(path)
    metadata: dict = {}
    ids, scores, labels = [], [], []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path} line {number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{where}: not valid JSON ({e.msg})") from None
        if not isinstance(row, dict):
            raise ValueError(f"{where}: each line must be a JSON object")  # noqa: TRY004, a damaged file
        if "metadata" in row:
            if ids:
                raise ValueError(f"{where}: a metadata line must come first")
            if not isinstance(row["metadata"], dict):
                raise ValueError(f"{where}: 'metadata' must be an object")
            metadata = row["metadata"]
            continue
        ids.append(_id(row, where))
        scores.append(_score(row, where))
        labels.append(_label(row, where))
    if not ids:
        raise ValueError(f"{path}: no rows")
    ids = example_ids(ids, len(ids), str(path))
    snapshot = Snapshot(ids=np.array(ids), labels=np.array(labels), scores=np.array(scores), metadata=metadata)
    return snapshot if order is None else _aligned(snapshot, order, path)


def _field(row: dict, key: str, where: str):
    if key not in row:
        raise ValueError(f"{where}: no {key!r}")
    return row[key]


def _id(row: dict, where: str) -> str:
    value = _field(row, "id", where)
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise ValueError(f"{where}: 'id' must be a string or an integer")  # noqa: TRY004, a damaged file
    return str(value)


def _score(row: dict, where: str) -> float:
    value = _field(row, "score", where)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where}: 'score' must be a number")  # noqa: TRY004, a damaged file
    if not math.isfinite(value):
        raise ValueError(f"{where}: 'score' is {value}; scores must be finite")
    return float(value)


def _label(row: dict, where: str) -> int:
    value = _field(row, "label", where)
    if isinstance(value, bool) or value not in (0, 1):
        raise ValueError(f"{where}: 'label' must be 0 (benign) or 1 (attack)")
    return int(value)


def _aligned(snapshot: Snapshot, order: np.ndarray, path: Path) -> Snapshot:
    """The snapshot's rows in the reference's id order."""
    want = [str(i) for i in order]
    index = {i: k for k, i in enumerate(snapshot.ids.tolist())}
    wanted = set(want)
    extra = [i for i in index if i not in wanted]
    missing = [i for i in want if i not in index]
    if extra or missing:
        parts = []
        if extra:
            parts.append(f"has {len(extra)} id{'s' * (len(extra) != 1)} the reference lacks, e.g. {extra[0]!r}")
        if missing:
            parts.append(f"lacks {len(missing)} of the reference's, e.g. {missing[0]!r}")
        raise PairingError(f"{path} {' and '.join(parts)}: the runs must score the same examples.")
    rows = np.array([index[i] for i in want])
    return Snapshot(
        ids=snapshot.ids[rows],
        labels=snapshot.labels[rows],
        scores=snapshot.scores[rows],
        metadata=snapshot.metadata,
    )
