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
        calls.append(list(prompts))
        rng = np.random.default_rng(len(calls))
        n = len(prompts)
        return hidden_states.Capture(
            activations=rng.normal(size=(n, len(cfg.layers), len(cfg.positions), 8)).astype(np.float16),
            position_tokens=[["\n\n"] * len(cfg.positions)] * n,
            gpu="NVIDIA L4",
            enable_thinking=cfg.enable_thinking,
            prompt_token_counts=[counts["per_prompt"]] * n,
        )

    monkeypatch.setattr(hidden_states, "capture", capture)
    monkeypatch.setattr(hulder.capture, "base_model_for", lambda *a, **k: ("Qwen/Qwen3-4B", None))
    return module.main, calls, counts


def run(main, tmp_path, prompts):
    spec = {**lib.worker_spec(("bf16", "qwen3_4b_bf16"), 300), "prompts": prompts, "extra_metadata": {}}
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
