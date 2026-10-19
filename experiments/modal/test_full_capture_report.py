"""The full-capture report on fake runs of three replicas: each batch classed across them, the
kept per-batch majority, the stop rules, the pilot comparison and the Volume status. CPU only."""

import json

import full_capture_report as report
import numpy as np
import pytest
import smoke_lib as lib

import hulder

SMOKE_IDS = [r["id"] for r in lib.smoke_prompts(64)]
# 20 prompts: batches 2 (rows 0 to 7), 3 (rows 8 to 15) and 4 (rows 16 to 19, short).
IDS = [f"slabs/train/{i}" for i in range(16)] + [f"slabs/test/{i}" for i in range(4)]
KEYS = ("18_-1", "27_-1")
# Prompt i was fed the token ids [i, i + 1000].
TOKENS = [[i, i + 1000] for i in range(len(IDS))]
DELETED = {"files": {}, "fetched": True, "backed_up": True, "deleted_from_volume": True, "problem": None}


@pytest.fixture
def world(tmp_path):
    """A recorded capture whose canary is cluster 0.0, the replicas' shared rows, and a pilot run
    holding the first batch of them, as the real pilot holds the first 125."""
    rng = np.random.default_rng(0)
    runs = tmp_path / "runs"
    (runs / "r0" / "a").mkdir(parents=True)
    for layer in lib.LAYERS:
        acts = rng.normal(size=(64, 8)).astype(np.float16)
        hulder.Snapshot(ids=np.array(SMOKE_IDS), labels=np.arange(64) % 2, activations=acts).save(
            runs / "r0" / "a" / f"{layer}"
        )
    clusters = tmp_path / "clusters.json"
    clusters.write_text(json.dumps([{"identity": [{}, {}], "clusters": [[{"name": "r0/a"}]]}]))
    rows = {key: rng.normal(size=(len(IDS), 8)).astype(np.float16) for key in KEYS}
    pilot = tmp_path / "pilot" / "pilot"
    for key, a in rows.items():
        save_snapshot(pilot / f"{key}.npz", IDS[:8], a[:8], {"gpu_uuid": "GPU-p"})
    return {
        "tmp": tmp_path,
        "runs": runs,
        "clusters": clusters,
        "rows": rows,
        "canary": lib.recorded_canary_hash(runs / "r0" / "a"),
    }


def save_snapshot(path, ids, acts, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"layer": int(path.stem.split("_")[0]), "position": int(path.stem.split("_")[1]), **metadata}
    hulder.Snapshot(ids=np.array(ids), labels=np.arange(len(ids)) % 2, activations=acts, metadata=meta).save(path)


def make_run(world, changes=None, *, canary=None, workers=None, failed=None, volume=None):
    """Three replicas r1 to r3 holding the shared rows, changed where changes says:
    {label: [(key, row, delta)]}."""
    run_dir = world["tmp"] / "run"
    configs = {}
    for i, label in enumerate(("r1", "r2", "r3")):
        out = run_dir / label
        out.mkdir(parents=True)
        (out / "capture.log").write_text("captured\n")
        worker = {
            "canary_sha256": (canary or {}).get(label, world["canary"]),
            "prompt_token_ids_sha256": lib.token_ids_sha256(TOKENS),
            "canary_token_ids_sha256": "c" * 64,
            "gpu_uuid": f"GPU-{3 - i}",
            "gpu_driver": "610.57.04",
            "snapshots": [f"{key}.npz" for key in KEYS],
            **(workers or {}).get(label, {}),
        }
        manifest = {"returncode": 0, "worker": worker}
        if failed == label:
            manifest = {"returncode": -9, "error": "timed out after 720 s"}
        else:
            lib.save_token_ids(out / lib.TOKEN_IDS_FILE, IDS, TOKENS)
            for key, a in world["rows"].items():
                a = a.copy()
                for k, row, delta in (changes or {}).get(label, []):
                    if k == key:
                        a[row] += np.float16(delta)
                meta = {
                    "model": "Qwen/Qwen3-4B",
                    "gpu_uuid": worker["gpu_uuid"],
                    "gpu_driver": worker["gpu_driver"],
                    "canary_sha256": worker["canary_sha256"],
                    "prompt_ids_sha256": hulder.prompt_ids_sha256(IDS),
                    "prompt_token_ids_sha256": worker["prompt_token_ids_sha256"],
                }
                save_snapshot(out / f"{key}.npz", IDS, a, meta)
        (out / "manifest.json").write_text(json.dumps(manifest))
        configs[label] = {
            "container": i,
            "config": "qwen3_4b_bf16_full",
            "returncode": manifest["returncode"],
            "volume": (volume or {}).get(label, DELETED),
        }
    run = {"run_id": "R", "commit": "c", "estimated_usd": 0.3, "stopped": None, "configs": configs}
    (run_dir / "manifest.json").write_text(json.dumps(run))
    return run_dir


def summarise(world, run_dir, splits=None):
    return report.summarise(
        run_dir, world["tmp"] / "pilot", world["clusters"], world["runs"], splits=splits, reference_cluster="0.0"
    )


def test_each_batch_is_classed_across_the_replicas_with_the_disagreement_rate_and_its_interval(world):
    # Batch 3: r2 alone differs, at one array. Batch 4: r2 and r3 differ, from r1 and each other.
    changes = {"r2": [("27_-1", 9, 1), ("18_-1", 17, 1)], "r3": [("18_-1", 18, 1)]}
    summary = summarise(world, make_run(world, changes))
    assert [(b["batch"], b["kind"], b["outliers"]) for b in summary["batches"]] == [
        (2, "agree", []),
        (3, "split", ["r2"]),
        (4, "no majority", ["r1", "r2", "r3"]),
    ]
    counts = summary["counts"]
    assert (counts["batches"], counts["agree"], counts["split"], counts["no_majority"]) == (3, 1, 1, 1)
    # 2 of 3 batches not unanimous: R's binom.test(2, 3) gives [0.0943, 0.9916].
    assert counts["disagreement_rate"] == pytest.approx(2 / 3)
    assert counts["disagreement_interval"] == pytest.approx([0.09429932, 0.99159624], abs=1e-7)
    assert summary["outliers_per_replica"] == {"r1": 1, "r2": 2, "r3": 1}


def test_the_report_states_the_split_rate_expected_from_the_rough_deviation_estimate(world):
    expected = summarise(world, make_run(world))["expected"]
    assert expected["deviations"] == {"batches_off_path": 1, "batches": 35}
    assert expected["per_batch_rate"] == pytest.approx(1 / 35)
    assert expected["per_batch_interval"] == pytest.approx(list(hulder.clopper_pearson(1, 35)))
    assert expected["split_rate"] == pytest.approx(3468 / 42875)
    assert expected["two_or_more_deviate_rate"] == pytest.approx(103 / 42875)
    assert expected["split_rate_range"] == pytest.approx(
        [lib.expected_split_rates(p)["split"] for p in hulder.clopper_pearson(1, 35)]
    )
    assert "independent" in expected["note"]


def test_each_replicas_canary_hash_is_listed_with_its_recorded_cluster(world):
    summary = summarise(world, make_run(world, canary={"r3": "f" * 64}))
    assert summary["canaries"] == {
        "r1": {"canary_sha256": world["canary"], "cluster": "0.0"},
        "r2": {"canary_sha256": world["canary"], "cluster": "0.0"},
        "r3": {"canary_sha256": "f" * 64, "cluster": "unknown path"},
    }
    assert summary["majority_canary"] == {"canary_sha256": world["canary"], "cluster": "0.0", "replicas": ["r1", "r2"]}


def test_the_kept_dataset_is_each_batchs_majority_with_the_contributing_gpus(world):
    run_dir = make_run(world, {"r2": [("18_-1", 9, 1)]})  # batch 3: r2 is the outlier
    summary = summarise(world, run_dir)
    assert summary["stop_reasons"] == []
    assert summary["kept"] == {"dir": "kept", "prompts": 20, "dropped_batches": [], "dropped": []}
    for key in KEYS:
        kept = hulder.Snapshot.load(run_dir / "kept" / f"{key}.npz")
        r1 = hulder.Snapshot.load(run_dir / "r1" / f"{key}.npz")
        assert kept.activations.tobytes() == r1.activations.tobytes()
        assert list(kept.ids) == IDS and list(kept.labels) == list(r1.labels)
        assert kept.metadata == {
            **r1.metadata,
            "gpu_uuid": ["GPU-1", "GPU-2", "GPU-3"],
            "gpu_driver": "610.57.04",
            "canary_sha256": world["canary"],
            "prompt_ids_sha256": hulder.prompt_ids_sha256(IDS),
        }


def test_a_replica_in_no_majority_contributes_no_gpu(world):
    changes = {"r3": [("18_-1", row, 1) for row in (0, 8, 16)]}  # r3 is the outlier on every batch
    run_dir = make_run(world, changes)
    summary = summarise(world, run_dir)
    assert hulder.Snapshot.load(run_dir / "kept" / "18_-1.npz").metadata["gpu_uuid"] == ["GPU-2", "GPU-3"]
    assert summary["findings"] == ["r3 is the outlier on 3 of 3 batches, a path switch rather than sporadic batches"]


def test_dropped_batches_are_listed_with_their_splits_and_the_kept_token_hash_is_recomputed(world, monkeypatch):
    monkeypatch.setattr(report, "NO_MAJORITY_LIMIT", 0.5)  # so 1 of 3 batches dropped doesn't stop it
    run_dir = make_run(world, {"r2": [("18_-1", 17, 1)], "r3": [("18_-1", 18, 1)]})  # batch 4
    splits = {i: i.split("/")[1] for i in IDS}
    summary = summarise(world, run_dir, splits=splits)
    assert summary["kept"] == {
        "dir": "kept",
        "prompts": 16,
        "dropped_batches": [4],
        "dropped": [{"id": i, "split": "test"} for i in IDS[16:]],
    }
    kept = hulder.Snapshot.load(run_dir / "kept" / "18_-1.npz")
    assert list(kept.ids) == IDS[:16]
    assert kept.metadata["prompt_ids_sha256"] == hulder.prompt_ids_sha256(IDS[:16])
    # Recomputed from the replicas' token ids for the kept 16, and saved with them.
    assert kept.metadata["prompt_token_ids_sha256"] == lib.token_ids_sha256(TOKENS[:16])
    assert lib.load_token_ids(run_dir / "kept" / lib.TOKEN_IDS_FILE) == (IDS[:16], TOKENS[:16])


def test_r1_a_token_ids_file_unlike_its_recorded_hash_stops_before_keeping(world):
    run_dir = make_run(world)
    lib.save_token_ids(run_dir / "r2" / lib.TOKEN_IDS_FILE, IDS, [[0, 1], *TOKENS[1:]])
    summary = summarise(world, run_dir)
    assert_stopped(summary, run_dir, "R1: r2's token_ids.npz doesn't match its prompt_token_ids_sha256")


def assert_stopped(summary, run_dir, *reasons):
    assert summary["stop_reasons"] == list(reasons)
    assert summary["kept"] is None and not (run_dir / "kept").exists()


def test_r2_more_than_1_percent_of_batches_without_a_majority_stops_before_keeping(world):
    run_dir = make_run(world, {"r2": [("18_-1", 17, 1)], "r3": [("18_-1", 18, 1)]})
    summary = summarise(world, run_dir)
    assert_stopped(summary, run_dir, "R2: 1 of 3 batches (33.3%) have no majority, over the 1% limit")


@pytest.mark.parametrize("key", ["prompt_token_ids_sha256", "canary_token_ids_sha256"])
def test_r1_replicas_fed_other_tokens_stop_before_keeping(world, key):
    run_dir = make_run(world, workers={"r3": {key: "x" * 64}})
    summary = summarise(world, run_dir)
    # r3's own token ids no longer match the hash it recorded for them, either.
    also = (
        ["R1: r3's token_ids.npz doesn't match its prompt_token_ids_sha256"] if key == "prompt_token_ids_sha256" else []
    )
    assert_stopped(summary, run_dir, f"R1: the replicas' {key} differ", *also)


def test_r1_replicas_whose_snapshot_metadata_differs_beyond_the_gpu_and_canary_stop(world):
    run_dir = make_run(world)
    path = run_dir / "r3" / "27_-1.npz"
    s = hulder.Snapshot.load(path)
    hulder.Snapshot(ids=s.ids, labels=s.labels, activations=s.activations, metadata={**s.metadata, "model": "x"}).save(
        path
    )
    assert_stopped(summarise(world, run_dir), run_dir, "R1: the replicas' 27_-1.npz metadata differs in ['model']")


def test_r3_a_majority_canary_off_the_majority_path_stops_before_keeping(world):
    run_dir = make_run(world, canary={"r2": "f" * 64, "r3": "f" * 64})
    summary = summarise(world, run_dir)
    assert_stopped(summary, run_dir, "R3: the replicas' majority canary is unknown path, not cluster 0.0")


def test_r3_three_different_canaries_have_no_majority_and_stop(world):
    run_dir = make_run(world, canary={"r2": "e" * 64, "r3": "f" * 64})
    summary = summarise(world, run_dir)
    assert summary["majority_canary"] is None
    assert_stopped(summary, run_dir, "R3: the replicas' canaries have no majority")


def test_a_failed_replica_is_a_stop(world):
    run_dir = make_run(world, failed="r3")
    summary = summarise(world, run_dir)
    assert_stopped(summary, run_dir, "r3 failed (returncode -9): timed out after 720 s")


def test_the_kept_batches_are_compared_with_the_pilots(world):
    summary = summarise(world, make_run(world))
    assert summary["vs_pilot"] == [{"batch": 2, "identical": True}]
    assert summary["findings"] == []


def test_kept_batches_unlike_the_pilots_are_a_finding_not_a_stop(world):
    shift = [("18_-1", 3, 1)]
    summary = summarise(world, make_run(world, {"r1": shift, "r2": shift, "r3": shift}))
    assert summary["vs_pilot"] == [{"batch": 2, "identical": False}]
    assert summary["findings"] == ["the kept batches [2] differ from the pilot's"]
    assert summary["stop_reasons"] == [] and summary["kept"] is not None


def test_a_pilot_of_other_prompts_is_refused(world):
    pilot = world["tmp"] / "pilot" / "pilot"
    for key, a in world["rows"].items():
        save_snapshot(pilot / f"{key}.npz", IDS[1:9], a[:8], {})
    with pytest.raises(ValueError, match="pilot"):
        summarise(world, make_run(world))


def test_the_report_says_whether_each_replicas_snapshots_left_the_volume(world):
    kept_on = {**DELETED, "backed_up": False, "deleted_from_volume": False}
    volume = {"r2": kept_on, "r3": {**kept_on, "problem": "the backup of 18_-1.npz doesn't match"}}
    summary = summarise(world, make_run(world, volume=volume))
    assert summary["volume"] == {
        "r1": "deleted from the hulder-runs Volume, after the fetch and the backup matched the container's sha256",
        "r2": "still on the hulder-runs Volume: no backup dir was given",
        "r3": "still on the hulder-runs Volume: the backup of 18_-1.npz doesn't match",
    }
