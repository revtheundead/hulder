"""The Modal runner's pure helpers. CPU only, no Modal."""

import json

import numpy as np
import pytest
import smoke_lib as lib


def test_smoke_prompts_are_balanced_and_in_file_order():
    rows = lib.smoke_prompts(64)
    assert [r["label"] for r in rows].count(0) == 32
    assert [r["label"] for r in rows].count(1) == 32
    every = [json.loads(line)["id"] for line in lib.PROMPTS.read_text().splitlines()]
    positions = [every.index(r["id"]) for r in rows]
    assert positions == sorted(positions)


def test_one_smoke_prompt_is_the_first_attack():
    first_attack = next(json.loads(line) for line in lib.PROMPTS.read_text().splitlines() if '"label": 1' in line)
    assert lib.smoke_prompts(1) == [first_attack]


def test_smoke_prompts_refuse_more_than_the_file_holds(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"id": "a", "label": 0, "text": "x"}\n{"id": "b", "label": 1, "text": "y"}\n')
    with pytest.raises(ValueError, match="too few"):
        lib.smoke_prompts(4, path)


def test_online_fp8_differs_from_bf16_only_in_quantization():
    bf16, online = lib.CONFIGS["qwen3_4b_bf16"][0], lib.CONFIGS["qwen3_4b_fp8_online"][0]
    assert {k for k in bf16.keys() | online.keys() if bf16.get(k) != online.get(k)} == {"quantization"}


def test_every_config_pins_a_full_revision_and_thinking_off():
    for cfg, _ in lib.CONFIGS.values():
        assert len(cfg["revision"]) == 40
        assert cfg["enable_thinking"] is False


def test_every_experiment_names_known_configs_with_unique_labels_and_fits_the_budget():
    for containers, _, timeout_s in lib.EXPERIMENTS.values():
        labels = [e[0] for group in containers for e in group]
        assert len(labels) == len(set(labels))
        assert all(e[1] in lib.CONFIGS for group in containers for e in group)
        assert max(map(len, containers)) <= lib.MAX_CONFIGS_PER_CONTAINER
        worst = sum(lib.worst_case_cost(len(group), timeout_s) for group in containers)
        assert worst < lib.BUDGET_USD


def test_a_comparison_runs_in_one_container_and_the_cross_container_run_in_two():
    containers, single_use, _ = lib.EXPERIMENTS["fp8_online"]
    assert len(containers) == 1 and not single_use
    containers, single_use, _ = lib.EXPERIMENTS["cross_container"]
    assert single_use and [[e[1] for e in group] for group in containers] == [["qwen3_4b_bf16"]] * 2


def test_the_batch_invariant_experiment_logs_cublas_on_default_runs_only_and_fits_today():
    [group], single_use, timeout_s = lib.EXPERIMENTS["batch_invariant"]
    assert not single_use and timeout_s == 600
    assert [e[2] for e in group] == [{lib.CUBLAS_LOG: "1"}] * 3 + [lib.BATCH_INVARIANT] * 3
    assert 0.86 + lib.worst_case_cost(len(group), timeout_s) <= lib.BUDGET_USD  # spent today, estimated


def test_the_two_path_census_traces_five_defaults_then_one_invariant_in_each_of_two_containers():
    containers, single_use, timeout_s = lib.EXPERIMENTS["two_path_census"]
    assert single_use and len(containers) == 2 and timeout_s == 8 * 60
    for group in containers:
        assert [e[1] for e in group] == ["qwen3_4b_bf16"] * 6
        assert [e[2] for e in group] == [{lib.CUBLAS_LOG: "1"}] * 5 + [lib.BATCH_INVARIANT]
    worst = sum(lib.worst_case_cost(len(group), timeout_s) for group in containers)
    assert 0.90 + worst <= lib.BUDGET_USD  # spent on 2026-10-15 before the run, from Modal's usage page


TRACE_LINE = (
    "[2026-10-13 15:11:50][cublasLt][{pid}][Trace][cublasLtTSTMatmul] A=0X2B3FD{ptr} "
    "Adesc=[type=R_16BF rows=2560 cols=6144 ld=2560] workSpace=0X0 algo=[algoId={algo} tile=MATMUL_TILE_128x128] "
    "stream=0X0\n"
)


def test_matmul_signatures_read_raw_traces_fingerprints_and_old_summaries_alike():
    raw = TRACE_LINE.format(pid=1234, ptr="0000000", algo=5) * 3
    fingerprint = "\n".join(lib.cublas_fingerprint(raw))
    # The 2026-10-13 batch_invariant run kept count-prefixed lines without timestamps (cublas_summary.txt).
    summary = "      3 " + TRACE_LINE.format(pid=99, ptr="27A3000", algo=5).split("]", 1)[1]
    other = TRACE_LINE.format(pid=1234, ptr="0000000", algo=21)
    [signature] = lib.matmul_signatures(raw)
    assert lib.matmul_signatures(fingerprint) == lib.matmul_signatures(summary) == {signature}
    assert signature.startswith("[cublasLt]") and "algoId=5" in signature and "0X" not in signature
    assert lib.matmul_signatures(other) != {signature}


def test_cublas_fingerprint_keeps_shapes_and_algorithms_and_drops_pointers_times_and_pids():
    line = (
        "[2026-10-13 15:11:50][cublasLt][{pid}][Trace][cublasLtTSTMatmul] A=0X2B3FD{ptr} "
        "Adesc=[type=R_16BF rows=2560 cols=6144 ld=2560] workSpace=0X0 algo=[algoId={algo} tile=MATMUL_TILE_128x128] "
        "stream=0X0\n"
    )
    run_a = line.format(pid=1234, ptr="0000000", algo=5) * 2 + "I! cuBLAS (v13.1.1) cublasSetStream_v2 called:\n"
    run_b = line.format(pid=5678, ptr="27A3000", algo=5) * 2
    switched = line.format(pid=5678, ptr="27A3000", algo=21) * 2
    assert lib.cublas_fingerprint(run_a) == lib.cublas_fingerprint(run_b)
    [entry] = lib.cublas_fingerprint(run_a)
    assert entry.split()[0] == "2" and "0X" not in entry and "algoId=5" in entry
    assert lib.cublas_fingerprint(run_b) != lib.cublas_fingerprint(switched)


def test_downloads_cover_each_model_once():
    assert lib.downloads("fp8_online") == [("Qwen/Qwen3-4B", lib.QWEN3_4B["revision"])]


def test_gpu_metadata_reads_nvidia_smi():
    line = "NVIDIA L4, GPU-aa966112-71db-0357-ec9c-cd62942e54c2, 580.95.05, 23034 MiB\n"
    assert lib.gpu_metadata(line) == {"gpu_uuid": "GPU-aa966112-71db-0357-ec9c-cd62942e54c2", "gpu_driver": "580.95.05"}


@pytest.mark.parametrize("key", ["gpu_uuid", "gpu_driver"])
def test_runs_on_another_gpu_or_driver_warn_when_checked(key):
    import hulder

    ids, labels = np.array([f"p{i}" for i in range(40)]), np.r_[np.zeros(20, int), np.ones(20, int)]
    scores = np.linspace(0.01, 0.99, 40)
    meta = {"base_model": "m", "model": "m", "gpu": "NVIDIA L4", "gpu_uuid": "GPU-a", "gpu_driver": "580.95.05"}
    reference = hulder.Snapshot(ids=ids, labels=labels, scores=scores, metadata=meta)
    candidate = hulder.Snapshot(ids=ids, labels=labels, scores=scores, metadata={**meta, key: "other"})
    report = hulder.check_snapshots(reference, candidate, threshold=0.5, target_fpr=0.05)
    assert any(key in warning for warning in report.warnings)


def test_log_facts_keeps_each_kind_of_line_once():
    lines = [
        "INFO 10-13 [gpu_worker.py:641] Available KV cache memory: 4.51 GiB",
        "INFO 10-13 [kv_cache_utils.py:2396] GPU KV cache size: 32,848 tokens",
        "INFO 10-13 [__init__.py:727] Selected CutlassFP8ScaledMMLinearKernel for Fp8LinearMethod",
        "INFO 10-13 [__init__.py:727] Selected CutlassFP8ScaledMMLinearKernel for Fp8LinearMethod",
        "INFO 10-13 [default_loader.py:431] Loading weights took 3.21 seconds",
        "captured 8/64",
    ]
    facts = lib.log_facts("\n".join(lines))
    assert [line.split("] ")[1] for line in facts["kv_cache"]] == [
        "Available KV cache memory: 4.51 GiB",
        "GPU KV cache size: 32,848 tokens",
    ]
    assert len(facts["kernels"]) == 1
    assert facts["timings"] == ["INFO 10-13 [default_loader.py:431] Loading weights took 3.21 seconds"]


def test_an_hour_of_l4_costs_the_published_rate():
    assert lib.estimate_cost(3600, gpu=True, cpu=0, memory_gib=0) == pytest.approx(0.80, abs=0.005)
    with_extras = lib.estimate_cost(3600, gpu=True, cpu=lib.GPU_CPU, memory_gib=lib.GPU_MEMORY_GIB)
    assert with_extras == pytest.approx(1.24, abs=0.01)


def test_relative_l2_is_zero_for_equal_and_scale_free():
    a = np.random.default_rng(0).normal(size=(3, 2, 8)).astype(np.float16)
    assert np.all(lib.relative_l2(a, a) == 0)
    assert lib.relative_l2(a, 1.01 * a.astype(np.float64)) == pytest.approx(0.01)


def test_compare_measures_each_run_against_the_first_per_layer(tmp_path):
    import compare

    import hulder

    rng = np.random.default_rng(0)
    ids, labels = np.array([f"p{i}" for i in range(4)]), np.array([0, 0, 1, 1])
    acts = {layer: rng.normal(size=(4, 8)).astype(np.float16) for layer in lib.LAYERS}
    moved = {"bf16_a": (0.0, "GPU-1"), "bf16_b": (0.0, "GPU-2"), "fp8": (0.5, "GPU-1")}
    for name, (step, uuid) in moved.items():
        (tmp_path / name).mkdir()
        log = {"timings": ["Loading weights took 194.81 seconds", "init engine took 11.85 s"]}
        (tmp_path / name / "manifest.json").write_text(json.dumps({"nvidia_smi": f"NVIDIA L4, {uuid}", "log": log}))
        for layer, a in acts.items():
            snapshot = hulder.Snapshot(
                ids=ids, labels=labels, activations=a + np.float16(step), metadata={"gpu_uuid": uuid}
            )
            snapshot.save(tmp_path / name / f"{layer}")
    run = {"run_id": "r", "commit": "c", "configs": {name: {"container": 0} for name in moved}}
    (tmp_path / "manifest.json").write_text(json.dumps(run))
    summary = compare.summarise(tmp_path)
    assert summary["reference"] == "bf16_a"
    assert len(summary["shifts"]) == 2 * len(lib.LAYERS)
    rerun = summary["shifts"][f"bf16_b vs bf16_a, layer {lib.LAYERS[0]}"]
    assert rerun["bit_identical"] == "4/4" and rerun["metadata_differs"] == ["gpu_uuid"]
    fp8 = summary["shifts"][f"fp8 vs bf16_a, layer {lib.LAYERS[0]}"]
    assert fp8["relative_l2_mean"] > 0 and fp8["metadata_differs"] == []
    assert summary["runs"]["fp8"]["timings"] == ["init engine took 11.85 s"]


def test_clusters_group_bit_exact_captures_per_config_and_engine_env(tmp_path):
    import clusters

    import hulder

    rng = np.random.default_rng(1)
    ids, labels = np.array([f"p{i}" for i in range(4)]), np.array([0, 0, 1, 1])
    path_one = {layer: rng.normal(size=(4, 8)).astype(np.float16) for layer in lib.LAYERS}
    path_two = {layer: (a * np.float16(1.01)) for layer, a in path_one.items()}
    config = {"model": "m", "layers": list(lib.LAYERS)}
    runs = {  # label -> (activations, engine_env, uuid)
        "a": (path_one, {}, "GPU-1"),
        "b": (path_two, {}, "GPU-1"),
        "c": (path_one, {"CUBLASLT_LOG_LEVEL": "2"}, "GPU-2"),  # logging doesn't change the identity
        "inv": (path_one, lib.BATCH_INVARIANT, "GPU-1"),
    }
    for label, (acts, env, uuid) in runs.items():
        (tmp_path / label).mkdir()
        worker = {"capture_config": config, "snapshots": ["x"]}
        manifest = {"nvidia_smi": f"NVIDIA L4, {uuid}, 580.95.05, 23034 MiB", "engine_env": env, "worker": worker}
        (tmp_path / label / "manifest.json").write_text(json.dumps(manifest))
        for layer, a in acts.items():
            hulder.Snapshot(ids=ids, labels=labels, activations=a).save(tmp_path / label / f"{layer}")
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "r", "configs": {k: {} for k in runs}}))
    by_env = {json.dumps(r["identity"][1]): r for r in clusters.clusters(clusters.members([tmp_path]))}
    default = by_env["{}"]
    assert [[m["name"] for m in c] for c in default["clusters"]] == [["r/a", "r/c"], ["r/b"]]
    assert default["between_clusters_relative_l2_mean"]["0 vs 1"][f"layer {lib.LAYERS[0]}"] > 0
    assert [[m["name"] for m in c] for c in by_env[json.dumps(lib.BATCH_INVARIANT)]["clusters"]] == [["r/inv"]]


def test_a_base_model_warning_stops_the_run_naming_the_override_that_resolves_it():
    warning = "Qwen/Qwen3-4B-FP8@96b30dc's model card couldn't be read (offline). Pass the base model as override=."
    error = lib.base_model_error("qwen3_4b_fp8", warning)
    assert warning in error
    assert "BASE_MODEL_OVERRIDES" in error and "experiments/modal/smoke_lib.py" in error
    assert '"qwen3_4b_fp8": "<base model>"' in error


def test_no_config_needs_a_base_model_override():
    # base_model_for resolves every config's base model itself; an entry here is for unusual checkpoints.
    assert lib.BASE_MODEL_OVERRIDES == {}


def test_clusters_report_which_captures_chose_the_same_cublas_algorithms(tmp_path):
    import clusters

    import hulder

    ids, labels = np.array([f"p{i}" for i in range(4)]), np.array([0, 0, 1, 1])
    acts = {layer: np.ones((4, 8), np.float16) for layer in lib.LAYERS}
    traces = {  # label -> (file, contents); "untraced" has no trace
        "a": ("cublas_fingerprint.txt", "\n".join(lib.cublas_fingerprint(TRACE_LINE.format(pid=1, ptr="0", algo=5)))),
        "old": ("cublas_summary.txt", "      2 " + TRACE_LINE.format(pid=2, ptr="0", algo=5).split("]", 1)[1]),
        "b": ("cublas_fingerprint.txt", "\n".join(lib.cublas_fingerprint(TRACE_LINE.format(pid=3, ptr="0", algo=21)))),
        "untraced": None,
    }
    for label, trace in traces.items():
        (tmp_path / label).mkdir()
        worker = {"capture_config": {"model": "m"}, "snapshots": ["x"]}
        manifest = {"nvidia_smi": "NVIDIA L4, GPU-1, 580.95.05, 23034 MiB", "engine_env": {}, "worker": worker}
        (tmp_path / label / "manifest.json").write_text(json.dumps(manifest))
        if trace:
            (tmp_path / label / trace[0]).write_text(trace[1])
        for layer, a in acts.items():
            hulder.Snapshot(ids=ids, labels=labels, activations=a).save(tmp_path / label / f"{layer}")
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "r", "configs": {k: {} for k in traces}}))
    [result] = clusters.clusters(clusters.members([tmp_path]))
    sets = result["cublas_signature_sets"]
    assert [s["members"] for s in sets] == [["r/a", "r/old"], ["r/b"]]
    assert sets[0]["signatures"] == 1 and "only_here" not in sets[0]
    [only_b], [only_first] = sets[1]["only_here"], sets[1]["only_in_first"]
    assert "algoId=21" in only_b and "algoId=5" in only_first
    assert result["untraced"] == ["r/untraced"]
