"""Checks on the arrays check and Snapshot take, so malformed input raises before any statistic."""

from collections import Counter
from collections.abc import Iterable

import numpy as np


def example_ids(ids: Iterable, n: int, role: str) -> list[str]:
    """The ids as strings, as prompt_ids_sha256 hashes them. One per example, none repeated."""
    ids = [str(i) for i in ids]
    if len(ids) != n:
        raise ValueError(f"{role} has {len(ids)} ids for {n} examples; each example needs exactly one id")
    repeated = [i for i, count in Counter(ids).items() if count > 1]
    if repeated:
        raise ValueError(
            f"{role} repeats {len(repeated)} ids, e.g. {repeated[0]!r}; each example needs its own id, "
            "or pairing can't tell examples apart"
        )
    return ids


def require_binary_labels(labels: np.ndarray) -> None:
    """0 is benign and 1 is attack, and both must be present, since FPR needs benign
    examples and FNR and AUROC need attacks."""
    labels = np.asarray(labels)
    other = sorted(set(np.unique(labels).tolist()) - {0, 1})
    if other:
        raise ValueError(f"labels must be 0 (benign) or 1 (attack); found {', '.join(map(str, other[:5]))}")
    for value, name in ((0, "benign"), (1, "attack")):
        if not (labels == value).any():
            raise ValueError(f"labels have no {name} ({value}) examples; the gate needs both classes")
