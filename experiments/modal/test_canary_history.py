"""The canary hash against the recorded Qwen3-4B captures of 2026-10-13 and 2026-10-15.

A check that canary_sha256 is implemented correctly: on batch 1 of every recorded capture, whose
bit-exact clusters docs/results/phase0/modal/bf16_clusters.json lists, equal hashes must mean the
same cluster. It shows the hash tells the known paths apart. It can't show there are no others, so
a hash not seen here is an unknown path, not a path switch. Needs the snapshots under
.cache/modal-runs, which aren't committed, so it skips where they're absent, as in CI.
"""

import json

import pytest
import smoke_lib as lib

RUNS = lib.ROOT / ".cache" / "modal-runs"
CLUSTERS = lib.ROOT / "docs" / "results" / "phase0" / "modal" / "bf16_clusters.json"


def canary_hash(name: str) -> str:
    """The canary hash of a recorded capture, named <run id>/<label>, from its first batch."""
    return lib.recorded_canary_hash(RUNS / name)


def recorded() -> list[dict]:
    groups = json.loads(CLUSTERS.read_text())
    names = [m["name"] for g in groups for c in g["clusters"] for m in c]
    missing = [n for n in names if not (RUNS / n / f"{lib.LAYERS[0]}.npz").exists()]
    if missing:
        pytest.skip(f"{len(missing)} recorded captures aren't under {RUNS}, e.g. {missing[0]}")
    return groups


def test_canary_hashes_split_the_recorded_bf16_captures_exactly_as_their_clusters():
    groups = [g for g in recorded() if g["identity"][0].get("quantization") is None]
    clusters = [frozenset(m["name"] for m in c) for g in groups for c in g["clusters"]]
    assert sorted(map(len, clusters)) == [2, 5, 16]  # 18 default captures on two paths, 5 invariant
    by_hash: dict[str, set[str]] = {}
    for name in set().union(*clusters):
        by_hash.setdefault(canary_hash(name), set()).add(name)
    assert sorted(map(frozenset, by_hash.values()), key=sorted) == sorted(clusters, key=sorted)


def test_the_fp8_captures_hash_unlike_every_bf16_cluster():
    groups = recorded()
    bf16 = {
        canary_hash(m["name"])
        for g in groups
        if g["identity"][0].get("quantization") is None
        for c in g["clusters"]
        for m in c
    }
    fp8 = [m["name"] for g in groups if g["identity"][0].get("quantization") for c in g["clusters"] for m in c]
    assert len(fp8) == 2
    for name in fp8:
        assert canary_hash(name) not in bf16, name
