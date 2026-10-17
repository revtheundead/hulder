"""worker.py's wiring with hulder.capture faked. The split and the budget decision are
smoke_lib.capture_order and split_canary, tested in CI; this checks the worker uses them. Imports
torch and reads vLLM's version, so it runs with --runvllm."""

import json

import numpy as np
import pytest
import smoke_lib as lib

import hulder

pytestmark = pytest.mark.vllm

PILOT = lib.smoke_prompts(64)[8:24]  # requested prompts that don't start with the canary


@pytest.fixture
def worker(monkeypatch):
    """worker.main with capture returning seeded float16 activations, and the prompt texts it got."""
    hidden_states = pytest.importorskip("hulder.capture.hidden_states")
    import worker as module

    import hulder.capture

    calls, counts = [], {"per_prompt": 50}

    def capture(prompts, cfg):
        # Prompt k's value at layer i and position j is k + 32 i + 64 j, exact in float16, and its
        # token at position p is "p<p>#<k>", so each row shows where it came from.
        calls.append(list(prompts))
        n = len(prompts)
        k, i, j = np.ix_(range(n), range(len(cfg.layers)), range(len(cfg.positions)))
        return hidden_states.Capture(
            activations=np.repeat((k + 32 * i + 64 * j)[..., None], 8, axis=-1).astype(np.float16),
            position_tokens=[[f"p{p}#{k}" for p in cfg.positions] for k in range(n)],
            gpu="NVIDIA L4",
            enable_thinking=cfg.enable_thinking,
            prompt_token_counts=[counts["per_prompt"]] * n,
            engine_seconds=41.5,
            batch_seconds=[0.25 * (b + 1) for b in range(-(-n // cfg.batch_size))],
        )

    monkeypatch.setattr(hidden_states, "capture", capture)
    monkeypatch.setattr(hulder.capture, "base_model_for", lambda *a, **k: ("Qwen/Qwen3-4B", None))
    return module.main, calls, counts


GPU = {"gpu_uuid": "GPU-1234", "gpu_driver": "580.95.05"}


def run(main, tmp_path, prompts, positions=(-1,)):
    spec = {**lib.worker_spec(("bf16", "qwen3_4b_bf16"), 300), "prompts": prompts, "extra_metadata": GPU}
    spec["config"] = {**spec["config"], "positions": positions}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    main(str(tmp_path / "spec.json"), str(tmp_path))


def test_the_worker_captures_the_canary_first_and_records_its_hash_with_exactly_the_requested_prompts(worker, tmp_path):
    main, calls, _ = worker
    run(main, tmp_path, PILOT)
    assert calls == [[r["text"] for r in lib.canary_prompts() + PILOT]]
    record = json.loads((tmp_path / "worker.json").read_text())
    assert len(record["canary_sha256"]) == 64 and record["canary_prompt_tokens"] == 8 * 50
    for layer in lib.LAYERS:
        snapshot = hulder.Snapshot.load(tmp_path / f"{layer}.npz")
        assert snapshot.ids.tolist() == [r["id"] for r in PILOT]
        assert snapshot.labels.tolist() == [r["label"] for r in PILOT]
        assert snapshot.metadata["canary_sha256"] == record["canary_sha256"]


def test_a_canary_over_the_token_budget_fails_the_worker_and_records_no_hash(worker, tmp_path):
    # 8 x 128 = 1024 tokens against a budget of 1023 (max_num_batched_tokens less one draft slot).
    main, _, counts = worker
    counts["per_prompt"] = 128
    with pytest.raises(ValueError, match="1024 prompt tokens.*budget of 1023"):
        run(main, tmp_path, PILOT)
    assert not list(tmp_path.glob("*.npz")) and not (tmp_path / "worker.json").exists()


def test_several_positions_give_one_snapshot_per_layer_and_position(worker, tmp_path):
    main, _, _ = worker
    run(main, tmp_path, PILOT, positions=(-1, -9, -10))
    assert sorted(p.name for p in tmp_path.glob("*.npz")) == sorted(
        f"{layer}_{p}.npz" for layer in lib.LAYERS for p in (-1, -9, -10)
    )
    # Layer 27 is index 1 and position -10 index 2; request row r is captured as prompt r + 8.
    snapshot = hulder.Snapshot.load(tmp_path / "27_-10.npz")
    assert snapshot.metadata["layer"] == 27 and snapshot.metadata["position"] == -10
    assert snapshot.ids.tolist() == [r["id"] for r in PILOT]
    assert snapshot.activations[:, 0].tolist() == [r + 8 + 32 + 128 for r in range(len(PILOT))]
    assert json.loads((tmp_path / "worker.json").read_text())["snapshots"] == sorted(
        f"{layer}_{p}.npz" for layer in lib.LAYERS for p in (-1, -9, -10)
    )


def test_one_position_keeps_the_layer_only_snapshot_names(worker, tmp_path):
    main, _, _ = worker
    run(main, tmp_path, PILOT)
    assert sorted(p.name for p in tmp_path.glob("*.npz")) == sorted(f"{layer}.npz" for layer in lib.LAYERS)
    snapshot = hulder.Snapshot.load(tmp_path / "18.npz")
    assert snapshot.activations[:, 0].tolist() == [r + 8 for r in range(len(PILOT))]


def test_the_worker_records_each_requested_prompts_token_at_each_position(worker, tmp_path):
    main, _, _ = worker
    run(main, tmp_path, PILOT, positions=(-1, -9, -10))
    record = json.loads((tmp_path / "worker.json").read_text())
    assert record["position_tokens"] == {str(p): [f"p{p}#{r + 8}" for r in range(len(PILOT))] for p in (-1, -9, -10)}


def test_the_worker_records_timing_the_gpu_and_the_requested_prompts_hash(worker, tmp_path):
    main, _, _ = worker
    run(main, tmp_path, PILOT)
    record = json.loads((tmp_path / "worker.json").read_text())
    assert record["engine_seconds"] == 41.5
    assert record["batch_seconds"] == [0.25, 0.5, 0.75]  # the canary's batch first, then the 16 requested
    assert (record["gpu_uuid"], record["gpu_driver"]) == ("GPU-1234", "580.95.05")
    assert record["prompt_ids_sha256"] == hulder.prompt_ids_sha256([r["id"] for r in PILOT])
