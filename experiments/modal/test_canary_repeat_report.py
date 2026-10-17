"""The canary_repeat report on fake runs: every batch against the majority path and the pilot's
canary_check, the containers against each other, their tokens and cuBLASLt algorithms. CPU only."""

import json

import canary_repeat_report as report
import numpy as np
import pytest
import smoke_lib as lib

import hulder

SMOKE_IDS = [r["id"] for r in lib.smoke_prompts(64)]
TRACE = "     9 [cublasLt][N][Trace][cublasLtTSTMatmul] Adesc=[type=R_16BF rows=2560] algo=[algoId={algo}]\n"


def save(directory, ids, acts):
    directory.mkdir(parents=True, exist_ok=True)
    for layer, a in acts.items():
        hulder.Snapshot(ids=np.array(ids), labels=np.arange(len(ids)) % 2, activations=a).save(directory / f"{layer}")


@pytest.fixture
def world(tmp_path):
    """A recorded majority-path capture with a cuBLASLt trace (cluster 0.0), and a pilot run whose
    canary_check matches it on batches 2 to 7 and differs on batch 8, as the real one did."""
    rng = np.random.default_rng(0)
    majority = {layer: rng.normal(size=(64, 8)).astype(np.float16) for layer in lib.LAYERS}
    runs = tmp_path / "runs"
    save(runs / "r0" / "a", SMOKE_IDS, majority)
    (runs / "r0" / "a" / "cublas_fingerprint.txt").write_text(TRACE.format(algo=13))
    clusters = tmp_path / "clusters.json"
    clusters.write_text(json.dumps([{"identity": [{}, {}], "clusters": [[{"name": "r0/a"}]]}]))
    pilot_rows = {layer: a[8:].copy() for layer, a in majority.items()}
    for a in pilot_rows.values():
        a[48:] += np.float16(0.5)  # batch 8
    save(tmp_path / "pilot" / "canary_check", SMOKE_IDS[8:], pilot_rows)
    return {"majority": majority, "pilot_rows": pilot_rows, "runs": runs, "clusters": clusters, "tmp": tmp_path}


def make_run(world, rows_by_label, *, algo_by_label=None, hash_by_label=None, failed=None):
    run_dir = world["tmp"] / "run"
    majority_hash = lib.recorded_canary_hash(world["runs"] / "r0" / "a")
    configs = {}
    for i, (label, rows) in enumerate(rows_by_label.items()):
        out = run_dir / label
        out.mkdir(parents=True)
        (out / "capture.log").write_text("captured\n")
        if failed == label:
            manifest = {"returncode": -9, "error": "timed out after 480 s"}
        else:
            save(out, SMOKE_IDS[8:], rows)
            (out / "cublas_fingerprint.txt").write_text(TRACE.format(algo=(algo_by_label or {}).get(label, 13)))
            worker = {
                "canary_sha256": (hash_by_label or {}).get(label, majority_hash),
                "prompt_token_ids_sha256": "t" * 64,
                "canary_token_ids_sha256": "c" * 64,
                "gpu_uuid": f"GPU-{i}",
                "gpu_driver": "610.57.04",
                "engine_seconds": 30.0,
                "modal": {"image_id": "im-1", "task_id": f"ta-{i}", "region": "us-east"},
                "nvidia_packages": {"nvidia-cublas": "13.1.1.3"},
            }
            manifest = {"returncode": 0, "worker": worker, "log": {"timings": []}}
        (out / "manifest.json").write_text(json.dumps(manifest))
        configs[label] = {"container": i, "returncode": manifest["returncode"]}
    run = {"run_id": "R", "commit": "c", "estimated_usd": 0.1, "configs": configs}
    (run_dir / "manifest.json").write_text(json.dumps(run))
    return run_dir


def summarise(world, run_dir):
    return report.summarise(run_dir, world["tmp"] / "pilot", world["clusters"], world["runs"], reference_cluster="0.0")


def test_each_batch_is_compared_with_the_majority_path_the_pilot_and_the_other_container(world):
    majority_rows = {layer: a[8:] for layer, a in world["majority"].items()}
    run_dir = make_run(world, {"a_canary_check": majority_rows, "b_canary_check": world["pilot_rows"]})
    summary = summarise(world, run_dir)
    a, b = summary["configs"]["a_canary_check"], summary["configs"]["b_canary_check"]
    assert [x["identical"] for x in a["vs_majority"]] == [True] * 8
    assert [x["batch"] for x in a["vs_majority"]] == list(range(1, 9))
    assert [x["identical"] for x in b["vs_majority"]] == [True] * 7 + [False]
    assert [x["identical"] for x in a["vs_pilot"]] == [True] * 6 + [False]
    assert [x["identical"] for x in b["vs_pilot"]] == [True] * 7
    assert [x["identical"] for x in summary["between_containers"]] == [True] * 6 + [False]
    assert summary["stop_reasons"] == []
    assert summary["findings"] == ["b_canary_check: batches [8] differ from the majority path (cluster 0.0)"]


def test_the_containers_tokens_and_records_are_reported(world):
    rows = {layer: a[8:] for layer, a in world["majority"].items()}
    summary = summarise(world, make_run(world, {"a_canary_check": rows, "b_canary_check": rows}))
    assert summary["token_hashes_agree"] is True
    a = summary["configs"]["a_canary_check"]
    assert (a["gpu_uuid"], a["gpu_driver"], a["container"]) == ("GPU-0", "610.57.04", "R#0")
    assert a["modal"] == {"image_id": "im-1", "task_id": "ta-0", "region": "us-east"}
    assert a["nvidia_cublas"] == "13.1.1.3"


def test_each_containers_cublas_algorithms_are_matched_against_the_recorded_traces(world):
    rows = {layer: a[8:] for layer, a in world["majority"].items()}
    run_dir = make_run(world, {"a_canary_check": rows, "b_canary_check": rows}, algo_by_label={"b_canary_check": 21})
    summary = summarise(world, run_dir)
    assert summary["configs"]["a_canary_check"]["cublas_same_as"] == ["r0/a"]
    assert summary["configs"]["b_canary_check"]["cublas_same_as"] == []


def test_an_unseen_canary_hash_marks_batch_1_as_differing_and_is_a_finding(world):
    rows = {layer: a[8:] for layer, a in world["majority"].items()}
    run_dir = make_run(world, {"a_canary_check": rows}, hash_by_label={"a_canary_check": "f" * 64})
    summary = summarise(world, run_dir)
    assert summary["configs"]["a_canary_check"]["vs_majority"][0] == {
        "batch": 1,
        "rows": 8,
        "identical": False,
        "by": "canary hash",
    }
    assert summary["findings"] == [
        "a_canary_check: unknown path, canary hash ffffffffffff… on GPU-0, driver 610.57.04, container R#0",
        "a_canary_check: batches [1] differ from the majority path (cluster 0.0)",
    ]
    assert summary["stop_reasons"] == []


def test_a_failed_container_is_a_stop_and_the_other_is_still_reported(world):
    rows = {layer: a[8:] for layer, a in world["majority"].items()}
    run_dir = make_run(world, {"a_canary_check": rows, "b_canary_check": rows}, failed="b_canary_check")
    summary = summarise(world, run_dir)
    assert summary["stop_reasons"] == ["b_canary_check failed (returncode -9): timed out after 480 s"]
    assert [x["identical"] for x in summary["configs"]["a_canary_check"]["vs_majority"]] == [True] * 8
    assert summary["between_containers"] is None


def test_token_hashes_that_werent_recorded_dont_count_as_agreeing(world):
    rows = {layer: a[8:] for layer, a in world["majority"].items()}
    run_dir = make_run(world, {"a_canary_check": rows, "b_canary_check": rows})
    for label in ("a_canary_check", "b_canary_check"):
        path = run_dir / label / "manifest.json"
        manifest = json.loads(path.read_text())
        del manifest["worker"]["prompt_token_ids_sha256"], manifest["worker"]["canary_token_ids_sha256"]
        path.write_text(json.dumps(manifest))
    assert summarise(world, run_dir)["token_hashes_agree"] is None
