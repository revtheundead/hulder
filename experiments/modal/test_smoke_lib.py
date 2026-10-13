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


def test_the_bf16_rerun_is_the_bf16_config_and_fp8_changes_only_the_checkpoint():
    bf16, rerun, fp8 = (lib.SMOKE[k][0] for k in ("qwen3_4b_bf16", "qwen3_4b_bf16_rerun", "qwen3_4b_fp8"))
    assert bf16 == rerun
    assert {k for k in bf16.keys() | fp8.keys() if bf16.get(k) != fp8.get(k)} == {"model", "revision", "quantization"}


def test_every_smoke_config_pins_a_full_revision_and_thinking_off():
    for cfg, _ in lib.SMOKE.values():
        assert len(cfg["revision"]) == 40
        assert cfg["enable_thinking"] is False


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


def test_the_stop_rules_leave_room_for_every_smoke_config():
    assert len(lib.SMOKE) * lib.worst_case_config_cost() < lib.RUN_BUDGET_USD


def test_relative_l2_is_zero_for_equal_and_scale_free():
    a = np.random.default_rng(0).normal(size=(3, 2, 8)).astype(np.float16)
    assert np.all(lib.relative_l2(a, a) == 0)
    assert lib.relative_l2(a, 1.01 * a.astype(np.float64)) == pytest.approx(0.01)


def test_compare_measures_each_run_against_bf16_per_layer(tmp_path):
    import compare

    import hulder

    rng = np.random.default_rng(0)
    ids, labels = np.array([f"p{i}" for i in range(4)]), np.array([0, 0, 1, 1])
    acts = {layer: rng.normal(size=(4, 8)).astype(np.float16) for layer in lib.LAYERS}
    moved = {"qwen3_4b_bf16": 0.0, "qwen3_4b_bf16_rerun": 0.0, "qwen3_4b_fp8": 0.5}
    for name, step in moved.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.json").write_text(json.dumps({"nvidia_smi": "NVIDIA L4, GPU-1", "log": {}}))
        for layer, a in acts.items():
            hulder.Snapshot(ids=ids, labels=labels, activations=a + np.float16(step)).save(tmp_path / name / f"{layer}")
    run = {"run_id": "r", "commit": "c", "configs": {name: {"wall_seconds": 1.0} for name in moved}}
    (tmp_path / "manifest.json").write_text(json.dumps(run))
    summary = compare.summarise(tmp_path)
    assert len(summary["shifts"]) == 2 * len(lib.LAYERS)
    rerun = summary["shifts"][f"qwen3_4b_bf16_rerun vs qwen3_4b_bf16, layer {lib.LAYERS[0]}"]
    assert rerun["bit_identical"] == "4/4"
    assert summary["shifts"][f"qwen3_4b_fp8 vs qwen3_4b_bf16, layer {lib.LAYERS[0]}"]["relative_l2_mean"] > 0
    assert summary["configs"]["qwen3_4b_fp8"]["nvidia_smi"] == "NVIDIA L4, GPU-1"
