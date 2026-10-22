"""The Modal runner's pure helpers. CPU only, no Modal."""

import hashlib
import json
from pathlib import Path

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


def test_the_canary_is_the_first_batch_of_the_64_smoke_prompts_as_captured_since_2026_10_13():
    # Literal ids of batch 1 of every Qwen3-4B capture so far, not smoke_prompts(8), which differs.
    assert [r["id"] for r in lib.canary_prompts()] == [
        "bpairs/apibp_0059_a",
        "bpairs/apibp_0112_a",
        "bpairs/apibp_0079_b",
        "bpairs/apibp_0553_b",
        "bpairs/apibp_0019_b",
        "bpairs/apibp_0022_b",
        "bpairs/apibp_0227_b",
        "bpairs/apibp_0029_b",
    ]
    assert len(lib.canary_prompts()) == lib.COMMON["batch_size"]


def _canary_acts(n=8, d=6, positions=(-1,), layers=lib.LAYERS, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, len(layers), len(positions), d)).astype(np.float16)


def test_equal_canary_activations_hash_equal_and_one_ulp_anywhere_changes_the_hash():
    acts = _canary_acts()
    assert lib.canary_sha256(acts, lib.LAYERS, (-1,)) == lib.canary_sha256(acts.copy(), lib.LAYERS, (-1,))
    nudged = acts.copy()
    nudged[5, 1, 0, 3] = np.nextafter(nudged[5, 1, 0, 3], np.float16(np.inf))
    assert lib.canary_sha256(nudged, lib.LAYERS, (-1,)) != lib.canary_sha256(acts, lib.LAYERS, (-1,))


def test_the_canary_hash_depends_on_which_layer_and_prompt_each_value_belongs_to():
    acts = _canary_acts()
    base = lib.canary_sha256(acts, lib.LAYERS, (-1,))
    assert lib.canary_sha256(acts[:, ::-1], lib.LAYERS, (-1,)) != base  # layer 18's values under 27
    assert lib.canary_sha256(acts[::-1], lib.LAYERS, (-1,)) != base  # prompts reordered
    # The same values listed in another layer order are the same capture.
    assert lib.canary_sha256(acts[:, ::-1], lib.LAYERS[::-1], (-1,)) == base


def test_the_canary_hash_reads_only_layers_18_and_27_at_position_minus_1():
    # The pilot captures positions -1, -9 and -10, and could add layers; its hash still compares
    # with the historical captures, which hold position -1 at layers 18 and 27 only.
    acts = _canary_acts()
    wide = _canary_acts(positions=(-9, -1, -10), layers=(4, *lib.LAYERS), seed=1)
    wide[:, 1:, 1] = acts[:, :, 0]
    assert lib.canary_sha256(wide, (4, *lib.LAYERS), (-9, -1, -10)) == lib.canary_sha256(acts, lib.LAYERS, (-1,))
    with pytest.raises(ValueError, match="layer 27"):
        lib.canary_sha256(acts[:, :1], (18,), (-1,))
    with pytest.raises(ValueError, match="position -1"):
        lib.canary_sha256(acts, lib.LAYERS, (-9,))


def test_the_canary_hash_ignores_memory_layout_and_byte_order_but_fixes_dtype_and_shape():
    acts = _canary_acts()
    base = lib.canary_sha256(acts, lib.LAYERS, (-1,))
    view = np.swapaxes(np.ascontiguousarray(np.swapaxes(acts, 0, 3)), 0, 3)  # same values, not C-contiguous
    assert not view.flags.c_contiguous
    assert lib.canary_sha256(view, lib.LAYERS, (-1,)) == base
    assert lib.canary_sha256(acts.astype(">f2"), lib.LAYERS, (-1,)) == base
    with pytest.raises(TypeError, match="float16"):
        lib.canary_sha256(acts.astype(np.float32), lib.LAYERS, (-1,))
    # The same bytes as 4 prompts of width 12 are another canary.
    reshaped = np.ascontiguousarray(acts.transpose(0, 2, 1, 3)).reshape(4, 1, 2, 12).transpose(0, 2, 1, 3)
    assert lib.canary_sha256(reshaped, lib.LAYERS, (-1,)) != base


def test_a_canary_batch_exactly_at_the_token_budget_passes_and_one_token_over_fails():
    # The counts exist only after capture, so this detects a canary split across forward passes
    # after it happened; it can't prevent one.
    lib.check_canary_budget([500, 523], 1023)
    with pytest.raises(ValueError, match=r"1024 prompt tokens.*budget of 1023"):
        lib.check_canary_budget([500, 524], 1023)


def test_every_worker_spec_carries_the_canary_apart_from_the_requested_prompts():
    spec = lib.worker_spec(("bf16", "qwen3_4b_bf16", {"X": "1"}), 300)
    assert [r["id"] for r in spec["canary"]] == [r["id"] for r in lib.canary_prompts()]
    assert spec["prompts"] == lib.smoke_prompts(64)
    assert (spec["config_name"], spec["env"], spec["timeout_s"], spec["save"]) == (
        "qwen3_4b_bf16",
        {"X": "1"},
        300,
        True,
    )


CANARY_IDS = [r["id"] for r in lib.canary_prompts()]
PILOT_IDS = [r["id"] for r in lib.smoke_prompts(64)[8:24]]  # a request that doesn't start with the canary
SMOKE_IDS = [r["id"] for r in lib.smoke_prompts(64)]  # one that does, as every historical capture


def test_the_canary_is_captured_first_unless_the_request_begins_with_it_in_order():
    assert lib.capture_order(PILOT_IDS, CANARY_IDS, batch_size=8) == CANARY_IDS + PILOT_IDS
    assert lib.capture_order(SMOKE_IDS, CANARY_IDS, batch_size=8) == SMOKE_IDS


@pytest.mark.parametrize(
    "request_ids",
    [
        PILOT_IDS[:3] + CANARY_IDS[2:3] + PILOT_IDS[3:],  # one canary id inside the request
        CANARY_IDS[::-1] + PILOT_IDS,  # the canary first but out of order
        CANARY_IDS[:7] + PILOT_IDS,  # most of it first
        SMOKE_IDS + CANARY_IDS[:1],  # all of it first, then one again
    ],
    ids=["inside", "reordered", "partial", "repeated"],
)
def test_canary_ids_anywhere_but_exactly_first_are_refused(request_ids):
    with pytest.raises(ValueError, match="canary prompt"):
        lib.capture_order(request_ids, CANARY_IDS, batch_size=8)


def test_the_canary_must_fill_exactly_one_batch():
    with pytest.raises(ValueError, match="one batch"):
        lib.capture_order(PILOT_IDS, CANARY_IDS, batch_size=16)


def _captured(n, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 2, 3, 4)).astype(np.float16)


def test_the_split_hashes_the_canary_rows_and_keeps_exactly_the_requested_rows():
    order = CANARY_IDS + PILOT_IDS
    acts = _captured(len(order))
    split = lib.split_canary(order, PILOT_IDS, CANARY_IDS, acts, [50] * len(order), lib.LAYERS, (-9, -1, -10), 1023)
    assert split.canary_sha256 == lib.canary_sha256(acts[:8], lib.LAYERS, (-9, -1, -10))
    assert split.canary_prompt_tokens == 400
    assert np.array_equal(split.activations, acts[8:])  # row i is PILOT_IDS[i]


def test_a_request_that_begins_with_the_canary_keeps_every_row():
    acts = _captured(64)
    split = lib.split_canary(SMOKE_IDS, SMOKE_IDS, CANARY_IDS, acts, [50] * 64, lib.LAYERS, (-1, -9, -10), 1023)
    assert np.array_equal(split.activations, acts)
    assert split.canary_sha256 == lib.canary_sha256(acts[:8], lib.LAYERS, (-1, -9, -10))


def test_the_split_refuses_a_canary_over_the_budget_and_rows_in_another_order():
    order = CANARY_IDS + PILOT_IDS
    acts, counts = _captured(len(order)), [128] * 8 + [50] * len(PILOT_IDS)
    with pytest.raises(ValueError, match="1024 prompt tokens.*budget of 1023"):
        lib.split_canary(order, PILOT_IDS, CANARY_IDS, acts, counts, lib.LAYERS, (-1,), 1023)
    lib.split_canary(order, PILOT_IDS, CANARY_IDS, acts, counts, lib.LAYERS, (-1,), 1024)  # exactly at it
    with pytest.raises(ValueError, match="capture order"):
        lib.split_canary(PILOT_IDS + CANARY_IDS, PILOT_IDS, CANARY_IDS, acts, counts, lib.LAYERS, (-1,), 1024)


def slabs_parquet(path, shuffle_seed=None):
    """A splits.parquet stand-in: 30 train rows among other splits, optionally in shuffled row order.
    Skips where pandas isn't installed, as in CI, which installs the test group only."""
    pd = pytest.importorskip("pandas")

    rows = [{"id": f"slabs/train/{i}", "split": "train", "text": f"train {i}", "label": i % 2} for i in range(30)]
    rows += [{"id": f"slabs/test/{i}", "split": "test", "text": f"test {i}", "label": 0} for i in range(10)]
    frame = pd.DataFrame(rows)
    if shuffle_seed is not None:
        frame = frame.sample(frac=1, random_state=shuffle_seed)
    frame.to_parquet(path)
    return path


def test_the_slabs_train_sample_is_n_distinct_train_rows_fixed_by_its_seed(tmp_path):
    path = slabs_parquet(tmp_path / "splits.parquet")
    sample = lib.slabs_train_prompts(10, 0, path)
    assert len({r["id"] for r in sample}) == 10
    assert all(r["id"].startswith("slabs/train/") for r in sample)
    for r in sample:
        i = int(r["id"].rsplit("/", 1)[1])
        assert r == {"id": f"slabs/train/{i}", "text": f"train {i}", "label": i % 2}
    assert lib.slabs_train_prompts(10, 0, path) == sample
    assert lib.slabs_train_prompts(10, 1, path) != sample


def test_a_shuffled_parquet_gives_the_same_slabs_sample(tmp_path):
    ordered = slabs_parquet(tmp_path / "ordered.parquet")
    shuffled = slabs_parquet(tmp_path / "shuffled.parquet", shuffle_seed=7)
    assert lib.slabs_train_prompts(10, 0, shuffled) == lib.slabs_train_prompts(10, 0, ordered)


def test_the_slabs_sample_refuses_more_than_the_train_split_holds(tmp_path):
    with pytest.raises(ValueError, match="30 train rows"):
        lib.slabs_train_prompts(31, 0, slabs_parquet(tmp_path / "splits.parquet"))


SPLITS = {
    "train": 30,
    "validation": 5,
    "calibration": 6,
    "test": 6,
    "bpairs_train": 4,
    "bpairs_validation": 2,
    "bpairs_test": 3,
}


def all_splits_parquet(path, shuffle_seed=None, extra=()):
    """A splits.parquet stand-in with every split, its rows in shuffled order if asked."""
    pd = pytest.importorskip("pandas")

    rows = [
        {"id": f"{split}/{i}", "split": split, "text": f"{split} {i}", "label": i % 2}
        for split, n in [*SPLITS.items(), *extra]
        for i in range(n)
    ]
    frame = pd.DataFrame(rows)
    if shuffle_seed is not None:
        frame = frame.sample(frac=1, random_state=shuffle_seed)
    frame.to_parquet(path)
    return path


def test_all_splits_put_train_first_in_the_pilots_order_then_each_other_split_whole(tmp_path):
    path = all_splits_parquet(tmp_path / "splits.parquet")
    prompts = lib.slabs_all_prompts(0, path)
    assert len(prompts) == sum(SPLITS.values()) == len({r["id"] for r in prompts})
    # The pilot's sample is a prefix, so the full capture's first batches have the pilot's composition.
    assert prompts[:10] == lib.slabs_train_prompts(10, 0, path)
    assert prompts[:30] == lib.slabs_train_prompts(30, 0, path)
    # Then validation, calibration, test and the boundary-pairs splits, each in one run.
    splits = [r["id"].split("/")[0] for r in prompts]
    assert list(dict.fromkeys(splits)) == list(SPLITS)
    assert lib.slabs_all_prompts(0, all_splits_parquet(tmp_path / "shuffled.parquet", shuffle_seed=3)) == prompts
    assert lib.slabs_all_prompts(1, path) != prompts


def test_all_splits_refuse_a_split_they_dont_know(tmp_path):
    with pytest.raises(ValueError, match="holdout"):
        lib.slabs_all_prompts(0, all_splits_parquet(tmp_path / "splits.parquet", extra=[("holdout", 2)]))


def test_all_splits_of_the_real_parquet_are_16332_prompts_with_the_pilots_1000_first():
    if not lib.SLABS_SPLITS.exists():
        pytest.skip("needs .cache/probe/splits.parquet")
    pytest.importorskip("pandas")
    prompts = lib.prompts_for(("slabs_all", 0))
    assert len(prompts) == 16332 == len({r["id"] for r in prompts})
    assert prompts[:1000] == lib.prompts_for(("slabs_train", 1000, 0))


def test_the_full_config_is_the_pilots_with_every_split_behind_the_canary_as_its_source():
    pilot_fields, _ = lib.CONFIGS["qwen3_4b_bf16_pilot"]
    fields, source = lib.CONFIGS["qwen3_4b_bf16_full"]
    assert fields == pilot_fields
    assert source == ("canary_first", ("slabs_all", 0))


def needs_the_real_parquet(source: tuple) -> bool:
    return source[0] != "smoke" and not (source[0] == "canary_first" and source[1][0] == "smoke")


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param(
            (name, e),
            marks=pytest.mark.xfail(
                strict=True,
                raises=ValueError,
                reason="its one prompt is the canary's first, and it predates the canary; not rerun since",
            ),
        )
        if e[1] == "qwen3_8b_bf16_load"
        else (name, e)
        for name, (containers, _, _) in lib.EXPERIMENTS.items()
        for group in containers
        for e in group
    ],
    ids=lambda x: f"{x[0]}-{x[1][0]}",
)
def test_every_experiments_worker_spec_passes_capture_order(entry):
    # What the worker does first, so a request it would refuse fails here rather than on Modal.
    name, e = entry
    if needs_the_real_parquet(lib.CONFIGS[e[1]][1]):
        if not lib.SLABS_SPLITS.exists():
            pytest.skip("needs .cache/probe/splits.parquet")
        pytest.importorskip("pandas")
    spec = lib.worker_spec(e, lib.EXPERIMENTS[name][2])
    ids = [r["id"] for r in spec["prompts"]]
    order = lib.capture_order(ids, [r["id"] for r in spec["canary"]], spec["config"]["batch_size"])
    assert len(order) == len(set(order))


def test_the_full_request_is_the_canary_then_train_in_the_pilots_order_then_the_later_splits_without_it():
    if not lib.SLABS_SPLITS.exists():
        pytest.skip("needs .cache/probe/splits.parquet")
    pytest.importorskip("pandas")
    ids = [r["id"] for r in lib.worker_spec(("r1", "qwen3_4b_bf16_full"), 720)["prompts"]]
    canary = [r["id"] for r in lib.canary_prompts()]
    every = [r["id"] for r in lib.prompts_for(("slabs_all", 0))]
    n_train = 8945
    assert ids[:8] == canary
    assert ids[8 : 8 + n_train] == [r["id"] for r in lib.slabs_train_prompts(n_train, 0)]
    assert ids[8 : 8 + 1000] == [r["id"] for r in lib.prompts_for(("slabs_train", 1000, 0))]  # the pilot's
    assert ids[8 + n_train :] == [i for i in every[n_train:] if i not in canary]
    # All 8 are boundary-pairs train prompts, so they were in the later splits until moved forward.
    assert len([i for i in every[n_train:] if i in canary]) == 8
    assert len(ids) == 16332 == len(set(ids)) and set(ids) == set(every)


def test_canary_first_moves_the_canary_to_the_front_and_keeps_the_rest_in_order():
    canary = lib.canary_prompts()
    rest = [{"id": f"x/{i}", "text": f"x {i}", "label": i % 2} for i in range(5)]
    prompts = [rest[0], canary[3], rest[1], canary[0], rest[2], rest[3], rest[4]]
    assert lib.canary_first(prompts) == canary + rest


def test_canary_first_refuses_a_request_whose_copy_of_a_canary_prompt_differs():
    canary = lib.canary_prompts()
    changed = {**canary[2], "text": canary[2]["text"] + " "}
    with pytest.raises(ValueError, match=canary[2]["id"]):
        lib.canary_first([changed])


def test_the_existing_configs_still_request_their_smoke_prompts():
    expected = {"qwen3_4b_bf16": 64, "qwen3_4b_fp8": 64, "qwen3_4b_fp8_online": 64, "qwen3_8b_bf16_load": 1}
    for name, n in expected.items():
        spec = lib.worker_spec(("x", name), 300)
        assert spec["prompts"] == lib.smoke_prompts(n)
        assert spec["save"] is (n > 1)


def test_importing_smoke_lib_needs_no_pandas():
    # The container imports smoke_lib, and only the local entrypoint reads the parquet.
    import subprocess
    import sys

    code = "import sys, smoke_lib; assert 'pandas' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=lib.ROOT / "experiments" / "modal", check=True)


def test_the_timing_estimate_projects_startup_plus_2040_mean_batches_and_describes_the_spread():
    # Per-prompt times 0.1 to 0.5 s; numpy's linear percentiles put the 5th at 0.12 and the 95th at 0.48.
    timing = lib.timing_estimate(30.0, [0.8, 1.6, 2.4, 3.2, 4.0])
    assert timing["batches"] == 5 and timing["total_seconds"] == pytest.approx(12.0)
    assert timing["mean_batch_seconds"] == pytest.approx(2.4)
    assert timing["projection_seconds"] == pytest.approx(30 + 2040 * 2.4)
    spread = timing["per_prompt_seconds"]
    assert [spread[k] for k in ("median", "p5", "p95", "max")] == pytest.approx([0.3, 0.12, 0.48, 0.5])


def test_constant_batch_times_collapse_the_band_onto_the_projection():
    timing = lib.timing_estimate(12.0, [0.4] * 125)
    assert timing["mean_band"] == pytest.approx([0.4, 0.4])
    assert timing["projection_band"] == pytest.approx([12 + 816, 12 + 816])


def test_the_bootstrap_band_is_the_percentile_interval_of_resampled_means():
    # Two batch times: a resample's mean is 1, 2 or 3 with probability 1/4, 1/2, 1/4, so the 2.5th and
    # 97.5th percentiles of 10,000 resampled means are 1 and 3.
    timing = lib.timing_estimate(0.0, [1.0, 3.0])
    assert timing["mean_band"] == [1.0, 3.0]
    assert timing["bootstrap"] == {"seed": 0, "resamples": 10_000, "confidence": 0.95}


def test_the_band_is_fixed_by_its_seed_holds_the_mean_and_narrows_with_more_batches():
    times = np.random.default_rng(5).gamma(4.0, 0.1, size=500)
    few, many = lib.timing_estimate(10.0, times[:50]), lib.timing_estimate(10.0, times)
    assert lib.timing_estimate(10.0, times[:50]) == few
    assert lib.timing_estimate(10.0, times[:50], seed=1)["mean_band"] != few["mean_band"]
    assert few["mean_band"][0] < few["mean_batch_seconds"] < few["mean_band"][1]
    assert many["mean_band"][1] - many["mean_band"][0] < few["mean_band"][1] - few["mean_band"][0]


@pytest.mark.parametrize(("startup", "fits"), [(179.5, True), (180.0, False)])
def test_the_capture_fits_only_if_startup_and_the_bands_upper_end_stay_under_the_timeout(startup, fits):
    # 2040 batches of 0.5 s are 1020 s, so 180 s of startup reaches the 1200 s timeout exactly.
    assert lib.timing_estimate(startup, [0.5] * 125)["fits_timeout"] is fits


def test_common_is_unchanged_so_the_canary_batch_stays_comparable():
    assert lib.COMMON == {
        "layers": (18, 27),
        "positions": (-1,),
        "system_prompt": "You are a helpful assistant.",
        "enable_thinking": False,
        "batch_size": 8,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 1024,
        "max_num_batched_tokens": 1024,
    }


def test_a_smoke_source_can_start_partway():
    assert lib.prompts_for(("smoke", 64, 8)) == lib.smoke_prompts(64)[8:]


def test_the_pilot_runs_canary_check_then_pilot_in_one_container_for_at_most_83_cents():
    [group], single_use, timeout_s = lib.EXPERIMENTS["pilot"]
    assert [e[0] for e in group] == ["canary_check", "pilot"] and not single_use
    assert timeout_s == 20 * 60
    assert lib.worst_case_cost(len(group), timeout_s) == pytest.approx(0.83, abs=0.005)


def test_canary_check_captures_exactly_the_historical_64_prompt_batches():
    [group], _, timeout_s = lib.EXPERIMENTS["pilot"]
    spec = lib.worker_spec(group[0], timeout_s)
    assert spec["config"] == lib.CONFIGS["qwen3_4b_bf16"][0]
    assert spec["prompts"] == lib.smoke_prompts(64)[8:]
    order = lib.capture_order([r["id"] for r in spec["prompts"]], CANARY_IDS, batch_size=8)
    assert order == SMOKE_IDS


def test_the_pilot_config_adds_positions_minus_9_and_minus_10_to_bf16_and_samples_1000_slabs_train_prompts():
    [group], _, _ = lib.EXPERIMENTS["pilot"]
    fields, source = lib.CONFIGS[group[1][1]]
    bf16 = lib.CONFIGS["qwen3_4b_bf16"][0]
    assert {k for k in fields.keys() | bf16.keys() if fields.get(k) != bf16.get(k)} == {"positions"}
    assert fields["positions"] == (-1, -9, -10)
    assert source == ("slabs_train", 1000, 0)


def test_the_runner_requires_todays_spend_rather_than_assuming_none():
    # capture.py imports modal, absent here and in CI, so its entrypoint's signature is read from source.
    import ast

    tree = ast.parse((lib.ROOT / "experiments" / "modal" / "capture.py").read_text())
    [main] = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    args = main.args
    defaults = dict(zip([a.arg for a in args.args][len(args.args) - len(args.defaults) :], args.defaults, strict=True))
    assert "spent_usd" in [a.arg for a in args.args] and "spent_usd" not in defaults


def save_capture(directory, acts, ids=SMOKE_IDS):
    """Snapshots of 64 prompts at layers 18 and 27, as a recorded capture saves them."""
    import hulder

    directory.mkdir(parents=True)
    labels = np.arange(len(ids)) % 2
    for i, layer in enumerate(lib.LAYERS):
        hulder.Snapshot(ids=np.array(ids), labels=labels, activations=acts[:, i, 0]).save(directory / f"{layer}")


def test_a_saved_captures_canary_hash_equals_the_workers_hash_of_the_same_activations(tmp_path):
    acts = np.random.default_rng(3).normal(size=(64, 2, 1, 8)).astype(np.float16)
    save_capture(tmp_path / "capture", acts)
    split = lib.split_canary(SMOKE_IDS, SMOKE_IDS, CANARY_IDS, acts, [50] * 64, lib.LAYERS, (-1,), 1023)
    assert lib.recorded_canary_hash(tmp_path / "capture") == split.canary_sha256


def test_a_saved_capture_that_does_not_begin_with_the_canary_has_no_canary_hash(tmp_path):
    save_capture(tmp_path / "capture", np.zeros((64, 2, 1, 8), np.float16), ids=SMOKE_IDS[::-1])
    with pytest.raises(ValueError, match="doesn't begin with the canary"):
        lib.recorded_canary_hash(tmp_path / "capture")


def test_equal_token_ids_hash_equal_whatever_their_integer_type():
    ids = [[151644, 8948, 198], [151644, 872]]
    as_numpy = [np.array(p, dtype=np.int64) for p in ids]
    assert lib.token_ids_sha256(ids) == lib.token_ids_sha256(as_numpy)
    assert len(lib.token_ids_sha256(ids)) == 64


@pytest.mark.parametrize(
    "other",
    [
        [[151644, 8948, 199], [151644, 872]],  # one id changed
        [[151644, 8948], [198, 151644, 872]],  # the same ids with a prompt boundary moved
        [[151644, 872], [151644, 8948, 198]],  # the prompts in another order
        [[151644, 8948, 198]],  # a prompt missing
    ],
)
def test_any_change_to_the_ids_their_prompts_or_order_changes_the_token_hash(other):
    assert lib.token_ids_sha256(other) != lib.token_ids_sha256([[151644, 8948, 198], [151644, 872]])


def test_compare_batches_names_each_batch_that_differs_anywhere_at_any_layer():
    rng = np.random.default_rng(2)
    reference = {layer: rng.normal(size=(20, 4)).astype(np.float16) for layer in lib.LAYERS}
    rows = {layer: a.copy() for layer, a in reference.items()}
    rows[27][9, 3] = np.nextafter(rows[27][9, 3], np.float16(np.inf))  # one ulp, row 9: the second batch
    batches = lib.compare_batches(rows, reference, batch_size=8, first_batch=2)
    # 20 rows make batches of 8, 8 and 4, numbered from 2 as canary_check's rows are.
    assert batches == [
        {"batch": 2, "rows": 8, "identical": True},
        {"batch": 3, "rows": 8, "identical": False},
        {"batch": 4, "rows": 4, "identical": True},
    ]


def test_compare_batches_refuses_rows_of_another_shape():
    a = {layer: np.zeros((16, 4), np.float16) for layer in lib.LAYERS}
    b = {layer: np.zeros((8, 4), np.float16) for layer in lib.LAYERS}
    with pytest.raises(ValueError, match="layer 18"):
        lib.compare_batches(a, b)


def test_canary_repeat_runs_three_untraced_and_one_traced_canary_check_in_fresh_containers_for_at_most_67_cents():
    containers, single_use, timeout_s = lib.EXPERIMENTS["canary_repeat"]
    assert single_use and timeout_s == 8 * 60
    # Untraced entries carry no env, so they run under exactly the pilot's engine environment.
    assert containers == [
        [("u1_canary_check", "qwen3_4b_bf16_canary_check")],
        [("u2_canary_check", "qwen3_4b_bf16_canary_check")],
        [("u3_canary_check", "qwen3_4b_bf16_canary_check")],
        [("t1_canary_check", "qwen3_4b_bf16_canary_check", {lib.CUBLAS_LOG: "1"})],
    ]
    assert [lib.worker_spec(g[0], timeout_s)["env"] for g in containers] == [{}, {}, {}, {lib.CUBLAS_LOG: "1"}]
    assert sum(lib.worst_case_cost(len(g), timeout_s) for g in containers) == pytest.approx(0.66, abs=0.005)


def test_full_capture_runs_three_untraced_replicas_in_fresh_containers_for_at_most_75_cents():
    containers, single_use, timeout_s = lib.EXPERIMENTS["full_capture"]
    assert single_use and timeout_s == 12 * 60
    # No env entry, so each runs under exactly the pilot's engine environment.
    assert containers == [[(f"r{i}", "qwen3_4b_bf16_full")] for i in (1, 2, 3)]
    assert sum(lib.worst_case_cost(len(g), timeout_s) for g in containers) == pytest.approx(0.75, abs=0.005)


class FakeVolume:
    """The Volume as capture.py and volume_release.py use it: read a file in chunks, remove one."""

    def __init__(self, files):
        self.files, self.removed = dict(files), []

    def read(self, name):
        data = self.files[name]
        yield data[:3]
        yield data[3:]

    def remove(self, name):
        self.removed.append(name)


def volume_files():
    files = {"18_-1.npz": b"layer 18 rows", "27_-1.npz": b"layer 27 rows"}
    return files, {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}


def test_volume_files_are_fetched_and_backed_up_matching_their_hashes(tmp_path):
    files, expected = volume_files()
    status = lib.settle_volume_files(expected, FakeVolume(files).read, tmp_path / "run", tmp_path / "backup")
    assert status == {"files": expected, "fetched": True, "backed_up": True, "problem": None}
    for name, data in files.items():
        assert (tmp_path / "run" / name).read_bytes() == data == (tmp_path / "backup" / name).read_bytes()
    assert (tmp_path / "backup" / "SHA256SUMS").read_text() == "".join(f"{expected[n]}  {n}\n" for n in sorted(files))


def test_a_fetched_file_that_doesnt_match_its_hash_is_not_backed_up(tmp_path):
    files, expected = volume_files()
    volume = FakeVolume({**files, "27_-1.npz": b"layer 27 rowz"})
    status = lib.settle_volume_files(expected, volume.read, tmp_path / "run", tmp_path / "backup")
    assert (status["fetched"], status["backed_up"]) == (False, False)
    assert "27_-1.npz" in status["problem"]
    assert not (tmp_path / "backup").exists()


def test_a_backup_that_doesnt_match_its_hash_is_not_counted_as_backed_up(tmp_path):
    files, expected = volume_files()

    def corrupting_copy(src, dst):
        Path(dst).write_bytes(Path(src).read_bytes()[:-1])

    status = lib.settle_volume_files(
        expected, FakeVolume(files).read, tmp_path / "run", tmp_path / "backup", copy=corrupting_copy
    )
    assert (status["fetched"], status["backed_up"]) == (True, False)
    assert "backup" in status["problem"]
    assert not (tmp_path / "backup" / "SHA256SUMS").exists()


def test_without_a_backup_dir_the_files_are_only_fetched(tmp_path):
    files, expected = volume_files()
    status = lib.settle_volume_files(expected, FakeVolume(files).read, tmp_path / "run", None)
    assert (status["fetched"], status["backed_up"], status["problem"]) == (True, False, None)


def backed_up(tmp_path):
    files, expected = volume_files()
    status = lib.settle_volume_files(expected, FakeVolume(files).read, tmp_path / "run", tmp_path / "backup")
    return status, FakeVolume(files)


def test_a_backup_reverified_after_a_reboot_releases_its_files_from_the_volume(tmp_path):
    status, volume = backed_up(tmp_path)
    released = lib.release_volume_files(status, tmp_path / "backup", volume.remove, "boot-a", "boot-b")
    assert released == {"released": True, "removed": sorted(status["files"]), "already_gone": [], "problem": None}
    assert volume.removed == sorted(status["files"])


def test_nothing_is_released_in_the_boot_that_fetched_the_files(tmp_path):
    status, volume = backed_up(tmp_path)
    released = lib.release_volume_files(status, tmp_path / "backup", volume.remove, "boot-a", "boot-a")
    assert (released["released"], volume.removed) == (False, [])
    assert "reboot" in released["problem"]


def test_nothing_is_released_when_a_backed_up_file_no_longer_matches(tmp_path):
    status, volume = backed_up(tmp_path)
    (tmp_path / "backup" / "27_-1.npz").write_bytes(b"layer 27 rowz")
    released = lib.release_volume_files(status, tmp_path / "backup", volume.remove, "boot-a", "boot-b")
    assert (released["released"], volume.removed) == (False, [])
    assert "27_-1.npz" in released["problem"]


def test_nothing_is_released_when_sha256sums_disagrees_with_the_containers_hashes(tmp_path):
    status, volume = backed_up(tmp_path)
    (tmp_path / "backup" / "SHA256SUMS").write_text(f"{'0' * 64}  18_-1.npz\n")
    released = lib.release_volume_files(status, tmp_path / "backup", volume.remove, "boot-a", "boot-b")
    assert (released["released"], volume.removed) == (False, [])
    assert "SHA256SUMS" in released["problem"]


def test_nothing_is_released_that_was_never_backed_up(tmp_path):
    files, expected = volume_files()
    status = lib.settle_volume_files(expected, FakeVolume(files).read, tmp_path / "run", None)
    volume = FakeVolume(files)
    released = lib.release_volume_files(status, tmp_path / "backup", volume.remove, "boot-a", "boot-b")
    assert (released["released"], volume.removed) == (False, [])
    assert "backed up" in released["problem"]


def replica_digests(**arrays):
    """Each replica's per-batch digests of each array, as the report builds them one file at a time."""
    return {label: {key: lib.batch_digests(a) for key, a in by_key.items()} for label, by_key in arrays.items()}


def rows_of(n=20, seed=0):
    return np.random.default_rng(seed).standard_normal((n, 4)).astype(np.float16)


def test_batches_are_classed_as_all_agree_a_2_to_1_split_or_all_different():
    # 20 rows in batches of 8: batch 2 is rows 0 to 7, batch 3 rows 8 to 15, batch 4 rows 16 to 19.
    r1 = rows_of()
    r2, r3 = r1.copy(), r1.copy()
    r2[9] += np.float16(1)  # batch 3: r2 alone differs
    r2[18] += np.float16(1)  # batch 4: r2 and r3 differ from r1 and from each other
    r3[17] += np.float16(1)
    agreement = lib.batch_agreement(replica_digests(r1={"18_-1": r1}, r2={"18_-1": r2}, r3={"18_-1": r3}))
    assert agreement == [
        {"batch": 2, "kind": "agree", "majority": ["r1", "r2", "r3"], "outliers": []},
        {"batch": 3, "kind": "split", "majority": ["r1", "r3"], "outliers": ["r2"]},
        {"batch": 4, "kind": "no majority", "majority": [], "outliers": ["r1", "r2", "r3"]},
    ]


def test_a_replica_that_differs_in_one_array_only_is_the_outlier():
    a, b = rows_of(8, 1), rows_of(8, 2)
    b3 = b.copy()
    b3[0, 0] += np.float16(1)
    digests = replica_digests(r1={"18_-1": a, "27_-1": b}, r2={"18_-1": a, "27_-1": b}, r3={"18_-1": a, "27_-1": b3})
    [batch] = lib.batch_agreement(digests)
    assert (batch["kind"], batch["outliers"]) == ("split", ["r3"])


def test_negative_zero_is_not_zero():
    zeros = np.zeros((8, 4), np.float16)
    negative = zeros.copy()
    negative[3, 2] = np.float16(-0.0)
    [batch] = lib.batch_agreement(replica_digests(r1={"k": zeros}, r2={"k": zeros}, r3={"k": negative}))
    assert batch["outliers"] == ["r3"]


def test_replicas_with_other_arrays_or_batch_counts_are_refused():
    with pytest.raises(ValueError, match="arrays"):
        lib.batch_agreement(replica_digests(r1={"a": rows_of()}, r2={"a": rows_of()}, r3={"b": rows_of()}))
    with pytest.raises(ValueError, match="batches"):
        lib.batch_agreement(replica_digests(r1={"a": rows_of()}, r2={"a": rows_of()}, r3={"a": rows_of(8)}))


def test_the_kept_rows_are_each_batchs_majority_in_capture_order_without_the_no_majority_batches():
    r1 = rows_of()
    r2, r3 = r1.copy(), r1.copy()
    r1[2] += np.float16(1)  # batch 2: r1 is the outlier, so r2's rows are kept
    r2[18] += np.float16(1)  # batch 4: no majority, so rows 16 to 19 are dropped
    r3[17] += np.float16(1)
    arrays = {"r1": r1, "r2": r2, "r3": r3}
    agreement = lib.batch_agreement(replica_digests(**{label: {"k": a} for label, a in arrays.items()}))
    kept = lib.majority_rows(arrays, agreement)
    assert kept.tobytes() == r2[:16].tobytes()
    # The same selection applied to the prompt ids gives the kept ids.
    ids = np.array([f"p{i}" for i in range(20)])
    assert list(lib.majority_rows({label: ids for label in arrays}, agreement)) == [f"p{i}" for i in range(16)]


def test_the_expected_split_rates_for_three_independent_replicas():
    # At p = 1/35: exactly one of three deviates with 3 * 34**2 / 35**3 = 3468/42875, and two or
    # more with 1 - (34**3 + 3468) / 35**3 = 103/42875.
    rates = lib.expected_split_rates(1 / 35)
    assert rates == pytest.approx({"split": 3468 / 42875, "two_or_more_deviate": 103 / 42875}, rel=1e-12)
    assert lib.expected_split_rates(0.0) == {"split": 0.0, "two_or_more_deviate": 0.0}


def test_saved_token_ids_load_back_as_they_were_and_hash_as_the_worker_hashed_them(tmp_path):
    ids, token_ids = ["a", "b", "c"], [[151644, 8948, 198], [], [2**32 - 1]]
    lib.save_token_ids(tmp_path / "token_ids.npz", ids, token_ids)
    loaded_ids, loaded = lib.load_token_ids(tmp_path / "token_ids.npz")
    assert (loaded_ids, loaded) == (ids, token_ids)
    assert lib.token_ids_sha256(loaded) == lib.token_ids_sha256(token_ids)


def test_token_ids_must_pair_with_their_prompt_ids(tmp_path):
    with pytest.raises(ValueError, match="2 prompt ids.*1 token"):
        lib.save_token_ids(tmp_path / "token_ids.npz", ["a", "b"], [[1]])
