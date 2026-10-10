"""Checks on the arrays check and Snapshot take, so malformed input raises before any statistic."""

from collections import Counter
from collections.abc import Iterable


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
