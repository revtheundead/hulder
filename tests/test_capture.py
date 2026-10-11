"""hulder.capture against a fake engine: what it passes to vLLM and what it records. CPU only,
but it imports vLLM, so it runs with --runvllm. Nothing heavy is imported at collection."""

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

pytestmark = pytest.mark.vllm

CONFIG = {
    "model": "org/model",
    "revision": "abc",
    "layers": (2,),
    "positions": (-1,),
    "system_prompt": "You are a helpful assistant.",
}


class FakeLLM:
    """Records each chat call and returns three-token prompts whose hidden states are their positions."""

    calls: ClassVar[list[dict]] = []

    def __init__(self, **kwargs):
        pass

    def get_tokenizer(self):
        return SimpleNamespace(convert_ids_to_tokens=lambda ids: [f"t{i}" for i in ids])

    def chat(self, convs, params, use_tqdm=False, **kwargs):
        FakeLLM.calls.append(kwargs)
        return [
            SimpleNamespace(kv_transfer_params={"hidden_states_path": f"{i}.pt"}, prompt_token_ids=[7, 8, 9])
            for i, _ in enumerate(convs)
        ]


def _fake_load(path):
    import torch

    return {"token_ids": torch.tensor([7, 8, 9]), "hidden_states": torch.arange(3.0).reshape(3, 1, 1).expand(3, 1, 4)}


@pytest.fixture
def hidden_states(monkeypatch):
    """hulder.capture.hidden_states with vLLM's engine, connector and device query faked."""
    module = pytest.importorskip("hulder.capture.hidden_states")
    FakeLLM.calls = []
    monkeypatch.setattr(module, "LLM", FakeLLM)
    monkeypatch.setattr(module.example_hidden_states_connector, "load_hidden_states", _fake_load)
    monkeypatch.setattr(module.os, "remove", lambda path: None)
    monkeypatch.setattr(module.torch.cuda, "get_device_name", lambda: "Fake RTX")
    return module


def test_enable_thinking_is_passed_to_the_chat_template_when_pinned(hidden_states):
    hidden_states.capture(["hi"], hidden_states.CaptureConfig(**CONFIG, enable_thinking=False))
    assert FakeLLM.calls == [{"chat_template_kwargs": {"enable_thinking": False}}]


def test_enable_thinking_is_left_to_the_template_by_default(hidden_states):
    config = hidden_states.CaptureConfig(**CONFIG)
    hidden_states.capture(["hi"], config)
    assert config.enable_thinking is None
    assert FakeLLM.calls == [{}]


def test_capture_records_the_gpu_and_the_requested_position(hidden_states):
    result = hidden_states.capture(["hi", "there"], hidden_states.CaptureConfig(**CONFIG))
    assert result.gpu == "Fake RTX"
    assert result.activations.shape == (2, 1, 1, 4)
    assert np.all(result.activations == 2)  # position -1 of hidden states 0, 1, 2
    assert result.position_tokens == [["t9"], ["t9"]]
