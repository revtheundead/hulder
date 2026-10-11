"""Snapshot metadata from a capture (hulder.capture.metadata). Imports no vLLM, so it runs
in CI: the config and capture are stand-ins with the fields the helper reads."""

from types import SimpleNamespace

import pytest

from hulder.capture.metadata import snapshot_metadata

CONFIG = {
    "model": "org/model-AWQ",
    "revision": "abc",
    "layers": (20,),
    "positions": (-1,),
    "system_prompt": "You are a helpful assistant.",
    "quantization": "awq",
    "dtype": "float16",
    "batch_size": 8,
}


def _capture(**kwargs):
    return SimpleNamespace(**{"gpu": "NVIDIA GeForce RTX 2070", "enable_thinking": None, **kwargs})


def test_metadata_records_the_serving_config_the_probes_layer_and_position_and_the_gpu():
    got = snapshot_metadata(SimpleNamespace(**CONFIG), _capture(), "org/model", "0.30.0")
    assert got == {
        "base_model": "org/model",
        "model": "org/model-AWQ",
        "revision": "abc",
        "quantization": "awq",
        "dtype": "float16",
        "batch_size": 8,
        "vllm_version": "0.30.0",
        "system_prompt": "You are a helpful assistant.",
        "layer": 20,
        "position": -1,
        "gpu": "NVIDIA GeForce RTX 2070",
    }


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_a_pinned_enable_thinking_is_recorded_from_the_capture(enable_thinking):
    got = snapshot_metadata(SimpleNamespace(**CONFIG), _capture(enable_thinking=enable_thinking), "org/model", "0.30.0")
    assert got["enable_thinking"] is enable_thinking


def test_a_capture_of_several_layers_and_positions_names_the_snapshots_one():
    cfg = SimpleNamespace(**{**CONFIG, "layers": (10, 20), "positions": (-2, -1)})
    got = snapshot_metadata(cfg, _capture(), "org/model", "0.30.0", layer=10, position=-2)
    assert (got["layer"], got["position"]) == (10, -2)
    with pytest.raises(ValueError, match="layer"):
        snapshot_metadata(cfg, _capture(), "org/model", "0.30.0")
    with pytest.raises(ValueError, match="30"):
        snapshot_metadata(cfg, _capture(), "org/model", "0.30.0", layer=30, position=-1)
