"""The Qwen3 probe's score stage and the deviation comparison on synthetic data: the frozen
selection's hash, one read of test and boundary-pairs per cell, rates at the frozen thresholds
with their intervals, pair-clustered boundary-pairs AUROC, and each deviated row against its twin. CPU only."""

import hashlib
import json

import numpy as np
import pytest

import hulder

pytest.importorskip("sklearn")
import deviations as dev
import train_qwen3 as t

SIZES = {"train": 300, "validation": 200, "calibration": 400, "test": 200, "bpairs": 120}


class FakeKept:
    """Rows per split and cell, the same on every read, with a count of every read."""

    def __init__(self):
        self.reads = {}

    def ids(self, split):
        if split == "bpairs":
            return [f"bpairs/apibp_{i // 2:04d}_{'ab'[i % 2]}" for i in range(SIZES[split])]
        return [f"{split}/{i}" for i in range(SIZES[split])]

    def rows(self, split, layer, position):
        self.reads[(split, layer, position)] = self.reads.get((split, layer, position), 0) + 1
        n = SIZES[split]
        y = np.arange(n) % 2
        rng = np.random.default_rng([n, layer, -position])
        x = rng.normal(size=(n, 8)) + (0.5 + layer / 100 - position / 50) * y[:, None]
        return x.astype(np.float16), y


@pytest.fixture
def frozen(tmp_path):
    """A selection frozen by the select stage on the fake rows, with its sha256."""
    kept = FakeKept()
    path = tmp_path / "selection.json"
    t.select_stage(kept.rows, path, probe_dir=tmp_path / "probes", source={})
    return {"kept": FakeKept(), "path": path, "sha": hashlib.sha256(path.read_bytes()).hexdigest(), "tmp": tmp_path}


def score(frozen, **overrides):
    kwargs = {
        "selection": frozen["path"],
        "probe_dir": frozen["tmp"] / "probes",
        "out": frozen["tmp"] / "scored.json",
        "expected_sha256": frozen["sha"],
        "source": {},
        **overrides,
    }
    return t.score_stage(frozen["kept"].rows, frozen["kept"].ids, **kwargs)


def test_the_score_stage_refuses_a_selection_other_than_the_committed_one(frozen):
    with pytest.raises(SystemExit, match="sha256"):
        score(frozen, expected_sha256="0" * 64)
    assert not (frozen["tmp"] / "scored.json").exists()
    assert frozen["kept"].reads == {}


def test_the_score_stage_refuses_a_probe_file_unlike_the_selections(frozen):
    probe = frozen["tmp"] / "probes" / "27_-9.npz"
    p = hulder.Probe.load(probe)
    hulder.Probe(weights=p.weights * 2, bias=p.bias, mean=p.mean, std=p.std, metadata=p.metadata).save(probe)
    with pytest.raises(SystemExit, match="27_-9"):
        score(frozen)


def test_the_score_stage_scores_test_and_boundary_pairs_exactly_once_for_all_6_cells(frozen):
    result = score(frozen)
    assert set(result["cells"]) == {f"{layer}_{position}" for layer, position in t.CELLS}
    for layer, position in t.CELLS:
        assert frozen["kept"].reads[("test", layer, position)] == 1
        assert frozen["kept"].reads[("bpairs", layer, position)] == 1
    with pytest.raises(SystemExit, match="already scored"):
        score(frozen)


def test_rates_are_at_the_frozen_threshold_with_clopper_pearson_intervals(frozen):
    result = score(frozen)
    selection = json.loads(frozen["path"].read_text())
    probe = hulder.Probe.load(frozen["tmp"] / "probes" / "18_-1.npz")
    x, y = FakeKept().rows("test", 18, -1)
    flagged = hulder.apply_probe(probe, x) > selection["cells"]["18_-1"]["threshold"]
    k, n = int((flagged & (y == 0)).sum()), int((y == 0).sum())
    fpr = result["cells"]["18_-1"]["test"]["fpr"]
    assert (fpr["k"], fpr["n"]) == (k, n)
    assert fpr["ci95"] == pytest.approx(list(hulder.clopper_pearson(k, n)))
    assert result["selection_sha256"] == frozen["sha"]


def test_boundary_pairs_report_the_surface_feature_retest_against_qwen25(frozen):
    cell = score(frozen)["cells"]["18_-1"]["bpairs"]
    assert set(cell) >= {"auroc", "ci95", "fpr", "fnr", "median_benign_score", "bootstrap"}
    assert cell["bootstrap"] == "pair-clustered"
    assert cell["qwen25_auroc"] == 0.839


def test_a_boundary_pairs_id_names_its_pair():
    assert t.pair_of("bpairs/apibp_0059_a") == t.pair_of("bpairs/apibp_0059_b") == "apibp_0059"


def snapshot(ids, acts):
    return hulder.Snapshot(ids=np.array(ids), labels=np.arange(len(ids)) % 2, activations=acts)


def test_only_rows_that_differ_from_their_twin_are_compared_matched_by_id():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 8)).astype(np.float16)
    probe = hulder.Probe(weights=np.ones(8), bias=0.0, mean=np.zeros(8), std=np.ones(8))
    twin = snapshot([f"p/{i}" for i in range(6)], x)
    moved = x[[4, 1, 2]].copy()
    moved[1] += np.float16(0.5)  # p/1 deviates; p/4 and p/2 are as the twin's
    deviated = snapshot(["p/4", "p/1", "p/2"], moved)
    out = dev.compare(probe, threshold=0.5, deviated=deviated, twin=twin)
    assert out["compared"] == 3 and out["deviated"] == 1
    (row,) = out["rows"]
    assert row["id"] == "p/1"
    logits = hulder.gate.probe.probe_logits(probe, np.stack([x[1], moved[1]]))
    assert row["delta_logit"] == pytest.approx(float(logits[1] - logits[0]))
    assert row["delta_score"] == pytest.approx(
        float(hulder.apply_probe(probe, moved[1:2])[0] - hulder.apply_probe(probe, x[1:2])[0])
    )


def test_a_decision_flip_is_a_row_whose_side_of_the_threshold_changes():
    probe = hulder.Probe(weights=np.ones(1), bias=0.0, mean=np.zeros(1), std=np.ones(1))
    twin = snapshot(["a", "b"], np.array([[-0.1], [2.0]], dtype=np.float16))
    deviated = snapshot(["a", "b"], np.array([[0.1], [2.5]], dtype=np.float16))
    out = dev.compare(probe, threshold=0.5, deviated=deviated, twin=twin)
    assert {r["id"]: r["flip"] for r in out["rows"]} == {"a": True, "b": False}
    assert out["flips"] == 1
