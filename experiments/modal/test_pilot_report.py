"""The pilot's report against its stop rules. CPU only, on fake runs; the last tests read the recorded
Qwen3-4B captures under .cache/modal-runs and skip where they're absent, as in CI."""

import json

import numpy as np
import pilot_report as report
import pytest
import smoke_lib as lib

import hulder

SMOKE_IDS = [r["id"] for r in lib.smoke_prompts(64)]
IM_END = "<|im_end|>"


def save(directory, ids, acts: dict[int, np.ndarray], suffix=""):
    directory.mkdir(parents=True, exist_ok=True)
    labels = np.arange(len(ids)) % 2
    for layer, a in acts.items():
        hulder.Snapshot(ids=np.array(ids), labels=labels, activations=a).save(directory / f"{layer}{suffix}")


@pytest.fixture
def recorded(tmp_path):
    """One recorded capture of the 64 smoke prompts, r0/a, alone in its cluster. Returns its rows,
    the runs directory and the clusters file."""
    rng = np.random.default_rng(0)
    rows = {layer: rng.normal(size=(64, 8)).astype(np.float16) for layer in lib.LAYERS}
    save(tmp_path / "runs" / "r0" / "a", SMOKE_IDS, rows)
    clusters = tmp_path / "clusters.json"
    clusters.write_text(json.dumps([{"identity": [{"quantization": None}, {}], "clusters": [[{"name": "r0/a"}]]}]))
    return rows, tmp_path / "runs", clusters


def make_run(tmp_path, recorded_rows, *, canary_rows=None, canary_hash=None, pilot_minus_9=None, pilot=True):
    """A pilot run directory: canary_check and pilot, as capture.py writes them."""
    run_dir = tmp_path / "run"
    hash_ = canary_hash or lib.recorded_canary_hash(tmp_path / "runs" / "r0" / "a")
    rows = canary_rows or {layer: a[8:] for layer, a in recorded_rows.items()}
    save(run_dir / "canary_check", SMOKE_IDS[8:], rows)
    n = 1000
    pilot_ids = [f"slabs/train/{i}" for i in range(n)]
    save(run_dir / "pilot", pilot_ids, {lib.LAYERS[0]: np.zeros((n, 8), np.float16)}, suffix="_-1")
    workers = {
        "canary_check": {
            "canary_sha256": hash_,
            "gpu_uuid": "GPU-1",
            "gpu_driver": "580.95.05",
            "engine_seconds": 40.0,
            "batch_seconds": [0.5] * 8,
        },
        "pilot": {
            "canary_sha256": hash_,
            "gpu_uuid": "GPU-1",
            "gpu_driver": "580.95.05",
            "engine_seconds": 12.0,
            "batch_seconds": [0.9] + [0.25] * 125,
            "prompt_ids_sha256": hulder.prompt_ids_sha256(pilot_ids),
            "position_tokens": {
                "-1": ["\n\n"] * n,
                "-9": pilot_minus_9 or [IM_END] * n,
                "-10": ["."] * (n - 1) + ["?"],
            },
        },
    }
    labels = ["canary_check", "pilot"] if pilot else ["canary_check"]
    configs = {}
    for label in labels:
        (run_dir / label).mkdir(exist_ok=True)
        init = {"canary_check": "10.49", "pilot": "9.61"}[label]
        line = f"(EngineCore pid=30) INFO 10-15 11:43:12 [core.py:372] init engine (profile, create kv cache, warmup model) took {init} s (compilation: 0.66 s)"
        manifest = {"returncode": 0, "seconds": 60.0, "worker": workers[label], "log": {"timings": [line]}}
        (run_dir / label / "manifest.json").write_text(json.dumps(manifest))
        (run_dir / label / "capture.log").write_text("captured\n")
        configs[label] = {"container": 0, "config": label, "returncode": 0}
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": "R", "commit": "c", "estimated_usd": 0.1, "configs": configs})
    )
    return run_dir


def test_a_clean_run_fires_no_stop_rule_and_matches_its_recorded_cluster(tmp_path, recorded):
    rows, runs, clusters = recorded
    summary = report.summarise(make_run(tmp_path, rows), clusters, runs)
    assert summary["stop_reasons"] == [] and summary["findings"] == []
    assert summary["canary_check_rows"]["status"] == "bit-identical"
    assert summary["canary_check_rows"]["against"] == "r0/a"
    for label in ("canary_check", "pilot"):
        config = summary["configs"][label]
        assert config["canary_match"] == {"cluster": "0.0", "members": ["r0/a"]}
        assert (config["gpu_uuid"], config["gpu_driver"]) == ("GPU-1", "580.95.05")
    assert summary["pilot"]["canary_hash_vs_canary_check"] == "same"


def test_the_pilot_timing_leaves_out_the_canary_batch_and_starts_from_the_slower_engine(tmp_path, recorded):
    rows, runs, clusters = recorded
    pilot = report.summarise(make_run(tmp_path, rows), clusters, runs)["pilot"]
    assert pilot["canary_batch_seconds"] == 0.9
    assert pilot["engine_seconds"] == {"canary_check": 40.0, "pilot": 12.0}
    timing = pilot["timing"]
    assert timing["batches"] == 125 and timing["mean_batch_seconds"] == pytest.approx(0.25)
    assert timing["projection_seconds"] == pytest.approx(40 + 2040 * 0.25)


def test_canary_check_rows_that_differ_from_the_matched_cluster_stop_the_run(tmp_path, recorded):
    rows, runs, clusters = recorded
    moved = {layer: a[8:].copy() for layer, a in rows.items()}
    moved[27][3, 5] = np.nextafter(moved[27][3, 5], np.float16(np.inf))
    summary = report.summarise(make_run(tmp_path, rows, canary_rows=moved), clusters, runs)
    assert summary["canary_check_rows"]["status"] == "differs"
    assert summary["canary_check_rows"]["layers"] == {"18": "56/56 identical", "27": "55/56 identical"}
    assert summary["stop_reasons"] == ["canary_check hashes like recorded cluster 0.0 but its rows differ from r0/a"]


def test_an_unseen_canary_hash_is_an_unknown_path_and_a_finding_not_a_stop(tmp_path, recorded):
    rows, runs, clusters = recorded
    summary = report.summarise(make_run(tmp_path, rows, canary_hash="f" * 64), clusters, runs)
    assert summary["stop_reasons"] == []
    assert summary["canary_check_rows"]["status"] == "not applicable"
    assert summary["configs"]["canary_check"]["canary_match"] == "unknown path"
    assert summary["findings"] == [
        "canary_check: unknown path, canary hash ffffffffffff… on GPU-1, driver 580.95.05, container R#0",
        "pilot: unknown path, canary hash ffffffffffff… on GPU-1, driver 580.95.05, container R#0",
    ]


def test_a_prompt_without_im_end_at_minus_9_is_listed_as_a_finding(tmp_path, recorded):
    rows, runs, clusters = recorded
    minus_9 = [IM_END] * 1000
    minus_9[17] = "ĊĊ"
    summary = report.summarise(make_run(tmp_path, rows, pilot_minus_9=minus_9), clusters, runs)
    positions = summary["pilot"]["positions"]
    assert positions["-9"]["im_end_on_every_prompt"] is False
    assert positions["-9"]["not_im_end"] == [{"id": "slabs/train/17", "token": "ĊĊ"}]
    assert positions["-10"]["distinct"] == 2
    assert summary["findings"] == ["pilot: 1 of 1000 prompts lack <|im_end|> at position -9"]
    assert summary["stop_reasons"] == []


def test_a_failed_config_and_the_budget_guard_stop_the_run(tmp_path, recorded):
    rows, runs, clusters = recorded
    run_dir = make_run(tmp_path, rows, pilot=False)
    (run_dir / "canary_check" / "capture.log").write_text(
        "ValueError: the canary's 1030 prompt tokens exceed the scheduler's per-step budget of 1023\n"
    )
    manifest = json.loads((run_dir / "canary_check" / "manifest.json").read_text())
    del manifest["worker"]
    (run_dir / "canary_check" / "manifest.json").write_text(json.dumps({**manifest, "returncode": 1}))
    run = json.loads((run_dir / "manifest.json").read_text())
    run["configs"]["canary_check"]["returncode"] = 1
    run["stopped"] = "container 0 failed"
    (run_dir / "manifest.json").write_text(json.dumps(run))
    summary = report.summarise(run_dir, clusters, runs)
    assert summary["stop_reasons"] == [
        "the run stopped: container 0 failed",
        "canary_check failed (returncode 1): the canary budget guard fired",
        "pilot did not run",
    ]


def test_a_timed_out_config_stops_the_run(tmp_path, recorded):
    rows, runs, clusters = recorded
    run_dir = make_run(tmp_path, rows)
    manifest = json.loads((run_dir / "pilot" / "manifest.json").read_text())
    manifest.update(returncode=-9, error="timed out after 1200 s")
    (run_dir / "pilot" / "manifest.json").write_text(json.dumps(manifest))
    summary = report.summarise(run_dir, clusters, runs)
    assert summary["stop_reasons"] == ["pilot failed (returncode -9): timed out after 1200 s"]


# On the recorded captures: the comparison and the hash lookup behave on real activations.

RECORDED = lib.ROOT / ".cache" / "modal-runs"
CLUSTERS = lib.ROOT / "docs" / "results" / "phase0" / "modal" / "bf16_clusters.json"


def bf16_default_clusters():
    groups = json.loads(CLUSTERS.read_text())
    [group] = [
        g
        for g in groups
        if g["identity"][0].get("quantization") is None and not g["identity"][1].get("VLLM_BATCH_INVARIANT")
    ]
    names = [m["name"] for c in group["clusters"] for m in c]
    if not all((RECORDED / n / f"{lib.LAYERS[0]}.npz").exists() for n in names):
        pytest.skip(f"the recorded captures aren't under {RECORDED}")
    return sorted(([m["name"] for m in c] for c in group["clusters"]), key=len)


def recorded_rows(name):
    return {layer: hulder.Snapshot.load(RECORDED / name / f"{layer}.npz").activations[8:] for layer in lib.LAYERS}


def test_rows_8_to_64_agree_within_the_majority_path_and_differ_across_paths():
    minority, majority = bf16_default_clusters()
    assert (len(minority), len(majority)) == (2, 16)
    same = report.compare_rows(recorded_rows(majority[0]), recorded_rows(majority[1]))
    assert same["status"] == "bit-identical"
    other = report.compare_rows(recorded_rows(majority[0]), recorded_rows(minority[0]))
    assert other["status"] == "differs"


def test_the_recorded_hashes_name_each_captures_own_cluster():
    minority, majority = bf16_default_clusters()
    hashes = report.recorded_hashes(CLUSTERS, RECORDED)
    by_member = {m: h for h, entry in hashes.items() for m in entry["members"]}
    assert by_member[majority[0]] == by_member[majority[-1]] != by_member[minority[0]]
    assert hashes[by_member[minority[0]]]["members"] == minority


def test_two_recorded_clusters_with_one_canary_hash_are_refused(tmp_path, recorded):
    # Two groups whose captures hold the same canary would make a hash name two clusters.
    rows, runs, clusters = recorded
    save(runs / "r1" / "b", SMOKE_IDS, rows)
    groups = json.loads(clusters.read_text())
    groups.append({"identity": [{"quantization": "fp8"}, {}], "clusters": [[{"name": "r1/b"}]]})
    clusters.write_text(json.dumps(groups))
    with pytest.raises(ValueError, match=r"recorded clusters 0\.0 and 1\.0 share a canary hash"):
        report.recorded_hashes(clusters, runs)


def test_vllms_own_init_engine_time_is_reported_next_to_engine_seconds(tmp_path, recorded):
    rows, runs, clusters = recorded
    run_dir = make_run(tmp_path, rows)
    configs = report.summarise(run_dir, clusters, runs)["configs"]
    assert (configs["canary_check"]["engine_seconds"], configs["canary_check"]["vllm_init_engine_seconds"]) == (
        40.0,
        10.49,
    )
    assert (configs["pilot"]["engine_seconds"], configs["pilot"]["vllm_init_engine_seconds"]) == (12.0, 9.61)
    # Without the line, as in a log cut short, the time is None rather than a guess.
    manifest = json.loads((run_dir / "pilot" / "manifest.json").read_text())
    (run_dir / "pilot" / "manifest.json").write_text(json.dumps({**manifest, "log": {"timings": []}}))
    assert report.summarise(run_dir, clusters, runs)["configs"]["pilot"]["vllm_init_engine_seconds"] is None
