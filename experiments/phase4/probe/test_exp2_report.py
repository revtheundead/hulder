"""Experiment 2's report on synthetic captures: the control against kept/, the FP8 majority and
its stop rules, the shuffled rerun reordered by prompt id, the real hulder check, and the
boundary-pairs comparison. CPU only."""

import json

import numpy as np
import pytest

import hulder

pytest.importorskip("sklearn")
import exp2_report as r

NAMES = ("18_-1", "27_-1")
D = 4


def snap(ids, acts, labels=None, **meta):
    labels = np.arange(len(ids)) % 2 if labels is None else labels
    return hulder.Snapshot(
        ids=np.array(ids),
        labels=np.asarray(labels),
        activations=np.asarray(acts, dtype=np.float16),
        metadata={"layer": 18, "position": -1, "base_model": "m", "batch_size": 8, **meta},
    )


def rows(n, seed=0):
    return np.random.default_rng(seed).normal(size=(n, D)).astype(np.float16)


# (1) The control against kept/.


def test_a_control_equal_to_kepts_rows_is_identical_on_every_array_and_batch():
    kept_ids = [f"p{i}" for i in range(40)]
    acts = rows(40)
    picked = list(range(8)) + list(range(24, 40))
    kept = {name: snap(kept_ids, acts) for name in NAMES}
    control = {name: snap([kept_ids[i] for i in picked], acts[picked]) for name in NAMES}
    out = r.control_vs_kept(control, kept, picked)
    assert out["identical"] and out["batches"] == 3
    assert all(a == {"ids_match": True, "labels_match": True, "differing_batches": []} for a in out["arrays"].values())


def test_one_changed_bit_names_its_array_and_batch_and_the_control_isnt_identical():
    kept_ids = [f"p{i}" for i in range(40)]
    acts = rows(40)
    picked = list(range(8)) + list(range(24, 40))
    kept = {name: snap(kept_ids, acts) for name in NAMES}
    changed = acts[picked].copy()
    changed[12, 1] = -changed[12, 1] if changed[12, 1] else np.float16(1)  # row 12 is batch 2
    control = {
        "18_-1": snap([kept_ids[i] for i in picked], acts[picked]),
        "27_-1": snap([kept_ids[i] for i in picked], changed),
    }
    out = r.control_vs_kept(control, kept, picked)
    assert not out["identical"]
    assert out["arrays"]["18_-1"]["differing_batches"] == []
    assert out["arrays"]["27_-1"]["differing_batches"] == [2]


def test_a_control_of_other_prompts_isnt_identical():
    kept_ids = [f"p{i}" for i in range(16)]
    acts = rows(16)
    kept = {name: snap(kept_ids, acts) for name in NAMES}
    control = {name: snap(kept_ids[::-1], acts[::-1]) for name in NAMES}
    out = r.control_vs_kept(control, kept, list(range(16)))
    assert not out["identical"] and not out["arrays"]["18_-1"]["ids_match"]


# (2) The FP8 majority and R1 to R3.


def replicas(n=24, **changes):
    ids = [f"p{i}" for i in range(n)]
    base = {name: rows(n, seed=k) for k, name in enumerate(NAMES)}
    out, workers = {}, {}
    for label in ("fp8_r1", "fp8_r2", "fp8_r3"):
        arrays = {name: base[name].copy() for name in NAMES}
        for name, row in changes.get(label, ()):
            arrays[name][row] += np.float16(1)
        out[label] = {name: snap(ids, arrays[name], gpu_uuid=label, canary_sha256="c") for name in NAMES}
        workers[label] = {"prompt_token_ids_sha256": "t", "canary_token_ids_sha256": "ct", "canary_sha256": "c"}
    return out, workers


def test_unanimous_replicas_keep_every_row_and_stop_nothing():
    reps, workers = replicas()
    out = r.fp8_majority(reps, workers, recorded={})
    assert out["stops"] == [] and out["counts"] == {"batches": 3, "agree": 3, "split": 0, "no_majority": 0}
    assert np.array_equal(out["kept"]["18_-1"].activations, reps["fp8_r1"]["18_-1"].activations)
    assert out["majority_canary"]["canary_sha256"] == "c"


def test_a_split_batch_keeps_the_majority_and_names_the_outlier():
    reps, workers = replicas(fp8_r2=[("27_-1", 9)])
    out = r.fp8_majority(reps, workers, recorded={})
    assert out["stops"] == [] and out["counts"]["split"] == 1
    assert out["batches"][1] == {"batch": 2, "kind": "split", "majority": ["fp8_r1", "fp8_r3"], "outliers": ["fp8_r2"]}
    assert np.array_equal(out["kept"]["27_-1"].activations, reps["fp8_r1"]["27_-1"].activations)


def test_r2_stops_when_over_1_percent_of_batches_have_no_majority():
    # r1 and r2 each differ from r3, and from each other, on batch 1.
    reps, workers = replicas(fp8_r1=[("18_-1", 0)], fp8_r2=[("18_-1", 1)])
    out = r.fp8_majority(reps, workers, recorded={})
    assert out["counts"]["no_majority"] == 1
    assert any(s.startswith("R2:") for s in out["stops"]) and out["kept"] is None


def test_r1_stops_on_other_tokens_or_other_metadata():
    reps, workers = replicas()
    workers["fp8_r3"]["prompt_token_ids_sha256"] = "other"
    assert any(
        s.startswith("R1:") and "prompt_token_ids_sha256" in s for s in r.fp8_majority(reps, workers, {})["stops"]
    )
    reps, workers = replicas()
    for name in NAMES:
        reps["fp8_r2"][name].metadata["quantization"] = "awq"
    assert any(s.startswith("R1:") and "quantization" in s for s in r.fp8_majority(reps, workers, {})["stops"])


def test_r3_stops_when_the_canaries_have_no_majority():
    reps, workers = replicas()
    for label, h in zip(workers, ("a", "b", "c"), strict=True):
        workers[label]["canary_sha256"] = h
    out = r.fp8_majority(reps, workers, recorded={})
    assert any(s.startswith("R3:") for s in out["stops"]) and out["majority_canary"] is None


def test_the_majority_canary_is_matched_against_recorded_paths():
    reps, workers = replicas()
    out = r.fp8_majority(reps, workers, recorded={"c": {"cluster": "1.0", "members": ["x"]}})
    assert out["majority_canary"]["cluster"] == "1.0"
    assert r.fp8_majority(reps, workers, recorded={})["majority_canary"]["cluster"] == "unknown path"


# (3) The shuffled rerun, reordered by prompt id.


def test_select_puts_rows_in_the_given_order_and_rehashes_ids_and_tokens():
    s = snap(["a", "b", "c"], rows(3), labels=[0, 1, 0])
    tokens = {"a": [1], "b": [2, 3], "c": [4]}
    out = r.select(s, ["c", "a"], tokens)
    assert [str(i) for i in out.ids] == ["c", "a"] and list(out.labels) == [0, 0]
    assert np.array_equal(out.activations, s.activations[[2, 0]])
    assert out.metadata["prompt_ids_sha256"] == hulder.prompt_ids_sha256(["c", "a"])
    assert out.metadata["prompt_token_ids_sha256"] == r.lib.token_ids_sha256([[4], [1]])
    with pytest.raises(KeyError):
        r.select(s, ["a", "z"], tokens)


def probe(threshold=0.5):
    return hulder.Probe(
        weights=np.ones(D),
        bias=0.0,
        mean=np.zeros(D),
        std=np.ones(D),
        metadata={"layer": 18, "position": -1, "base_model": "m", "threshold": threshold, "target_fpr": 0.05},
    )


def test_the_noise_floor_counts_identical_rows_score_shifts_and_flips():
    acts = rows(16)
    moved = acts.copy()
    moved[3] = -moved[3] if moved[3].sum() else moved[3] + np.float16(1)  # flips row 3's side of 0.5
    out = r.paired_shift(
        probe(), 0.5, snap([f"p{i}" for i in range(16)], acts), snap([f"p{i}" for i in range(16)], moved)
    )
    assert out["rows_bit_identical"] == 15 and out["n"] == 16
    assert out["flips"]["k"] == 1
    assert out["mean_abs_dscore"] > 0 and out["max_abs_dlogit"] > 0


def test_the_paired_shift_refuses_unpaired_snapshots():
    with pytest.raises(ValueError, match="pair"):
        r.paired_shift(probe(), 0.5, snap(["a", "b"], rows(2)), snap(["b", "a"], rows(2)))


# (4) The real hulder check, run as a user would.


def gate_inputs(tmp_path, shift=0.0):
    n = 400
    y = np.arange(n) % 2
    x = (np.random.default_rng(1).normal(size=(n, D)) + 2.0 * y[:, None]).astype(np.float16)
    ids = [f"p{i}" for i in range(n)]
    paths = {}
    for name, a in (("ref", x), ("rerun", x), ("cand", x + np.float16(shift))):
        paths[name] = tmp_path / f"{name}.npz"
        snap(ids, a, labels=y).save(paths[name])
    p = probe(threshold=float(np.quantile(hulder.apply_probe(probe(), x[y == 0]), 0.95, method="higher")))
    paths["probe"] = tmp_path / "probe.npz"
    p.save(paths["probe"])
    return paths


def test_the_check_runs_the_cli_and_records_its_exit_code_text_and_json(tmp_path):
    paths = gate_inputs(tmp_path, shift=0.5)
    out = r.run_check(paths["probe"], paths["ref"], paths["rerun"], paths["cand"])
    expected = hulder.check_snapshots(
        hulder.Snapshot.load(paths["ref"]),
        hulder.Snapshot.load(paths["cand"]),
        hulder.Snapshot.load(paths["rerun"]),
        threshold=hulder.Probe.load(paths["probe"]).metadata["threshold"],
        target_fpr=0.05,
        probe=hulder.Probe.load(paths["probe"]),
    )
    assert out["exit_code"] == expected.verdict.exit_code
    assert out["json"]["verdict"] == expected.verdict.value
    assert f"Exit code {expected.verdict.exit_code}" in out["text"]
    assert out["command"][1:3] == ["check", "--probe"] and "--noise-floor" in out["command"]


def test_an_unchanged_candidate_passes(tmp_path):
    paths = gate_inputs(tmp_path)
    out = r.run_check(paths["probe"], paths["ref"], paths["rerun"], paths["cand"])
    assert out["exit_code"] == 0 and out["json"]["verdict"] == "PASS"


# (5) Boundary-pairs, labelled as not counting.


def test_boundary_pairs_compares_rates_flips_and_clustered_auroc_and_doesnt_count():
    n = 40
    ids = [f"bpairs/apibp_{i // 2:04d}_{'ab'[i % 2]}" for i in range(n)]
    y = np.array([1, 0] * (n // 2))
    x = (np.random.default_rng(2).normal(size=(n, D)) + 1.5 * y[:, None]).astype(np.float16)
    ref, cand, rerun = snap(ids, x, labels=y), snap(ids, x + np.float16(0.25), labels=y), snap(ids, x, labels=y)
    out = r.boundary_pairs(probe(), 0.5, ref, cand, rerun)
    assert out["counts_toward_verdict"] is False and "not" in out["note"]
    for side in ("reference", "candidate"):
        assert set(out[side]) == {"fpr", "fnr", "auroc"} and len(out[side]["auroc"]["ci95"]) == 2
    assert out["shift"]["candidate"]["n"] == n and out["shift"]["rerun"]["flips"]["k"] == 0


# The orchestration: stops before any check when the control or the majority fails.


def test_the_report_stops_before_the_check_when_the_control_differs(monkeypatch, tmp_path):
    monkeypatch.setattr(r, "control_vs_kept", lambda *a, **k: {"identical": False, "arrays": {}, "batches": 414})
    monkeypatch.setattr(r, "load_run", lambda *a, **k: {"stops": [], "workers": {}, "token_ids": {}, "names": []})
    called = []
    monkeypatch.setattr(r, "fp8_majority", lambda *a, **k: called.append(1))
    out = r.report(tmp_path, tmp_path, tmp_path, tmp_path / "out", splits={})
    assert out["stopped_at"] == "control" and called == [] and "checks" not in out
    assert json.loads(json.dumps(out)) == out  # serialisable as written
