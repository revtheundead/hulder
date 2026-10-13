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
    assert lib.CONFIGS["qwen3_4b_fp8_online"][2] == "Qwen/Qwen3-4B"  # itself, not its card's parent


def test_every_config_pins_a_full_revision_and_thinking_off():
    for cfg, _, _ in lib.CONFIGS.values():
        assert len(cfg["revision"]) == 40
        assert cfg["enable_thinking"] is False


def test_every_experiment_names_known_configs_with_unique_labels_and_fits_the_budget():
    for containers, single_use in lib.EXPERIMENTS.values():
        labels = [label for group in containers for label, _ in group]
        assert len(labels) == len(set(labels))
        assert all(config in lib.CONFIGS for group in containers for _, config in group)
        assert max(map(len, containers)) <= lib.MAX_CONFIGS_PER_CONTAINER
        worst = sum(lib.worst_case_cost(len(group)) for group in containers)
        assert worst < lib.BUDGET_USD


def test_a_comparison_runs_in_one_container_and_the_cross_container_run_in_two():
    containers, single_use = lib.EXPERIMENTS["fp8_online"]
    assert len(containers) == 1 and not single_use
    containers, single_use = lib.EXPERIMENTS["cross_container"]
    assert single_use and [[config for _, config in group] for group in containers] == [["qwen3_4b_bf16"]] * 2


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
