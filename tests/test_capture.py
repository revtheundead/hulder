"""hulder.capture against a fake engine: what it passes to vLLM and what it records. CPU only."""

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch

hidden_states = pytest.importorskip("hulder.capture.hidden_states")

CONFIG = hidden_states.CaptureConfig(
    model="org/model", revision="abc", layers=(2,), positions=(-1,), system_prompt="You are a helpful assistant."
)


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
    return {"token_ids": torch.tensor([7, 8, 9]), "hidden_states": torch.arange(3.0).reshape(3, 1, 1).expand(3, 1, 4)}


@pytest.fixture
def fake_engine(monkeypatch):
    FakeLLM.calls = []
    monkeypatch.setattr(hidden_states, "LLM", FakeLLM)
    monkeypatch.setattr(hidden_states.example_hidden_states_connector, "load_hidden_states", _fake_load)
    monkeypatch.setattr(hidden_states.os, "remove", lambda path: None)
    monkeypatch.setattr(hidden_states.torch.cuda, "get_device_name", lambda: "Fake RTX")


def test_enable_thinking_is_passed_to_the_chat_template_when_pinned(fake_engine):
    hidden_states.capture(["hi"], CONFIG.__class__(**{**CONFIG.__dict__, "enable_thinking": False}))
    assert FakeLLM.calls == [{"chat_template_kwargs": {"enable_thinking": False}}]


def test_enable_thinking_is_left_to_the_template_by_default(fake_engine):
    hidden_states.capture(["hi"], CONFIG)
    assert CONFIG.enable_thinking is None
    assert FakeLLM.calls == [{}]


def test_capture_records_the_gpu_and_the_requested_position(fake_engine):
    result = hidden_states.capture(["hi", "there"], CONFIG)
    assert result.gpu == "Fake RTX"
    assert result.activations.shape == (2, 1, 1, 4)
    assert np.all(result.activations == 2)  # position -1 of hidden states 0, 1, 2
    assert result.position_tokens == [["t9"], ["t9"]]
