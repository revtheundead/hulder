"""hulder.runtime.serve_args: the `vllm serve` arguments for in-server scoring (docs/design.md §5 D1, D3). CPU only."""

import json
import subprocess
import sys
from importlib.resources import files

import numpy as np
import pytest

import hulder
from hulder.runtime import serve_args

DEMO = files("hulder") / "data" / "probe_demo.npz"


def probe_file(tmp_path, **metadata):
    path = tmp_path / "probe.npz"
    d = 4
    hulder.Probe(np.ones(d), 0.0, np.zeros(d), np.ones(d), metadata=metadata).save(path)
    return path


def parsed(args):
    assert args[0::2] == ["--speculative-config", "--kv-transfer-config"]
    return json.loads(args[1]), json.loads(args[3])


def test_the_demo_probe_gets_zero_draft_at_its_layer_and_the_scoring_connector():
    speculative, transfer = parsed(serve_args(DEMO))
    assert speculative == {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "num_speculative_tokens_per_batch_size": [[1, 1, 0]],
        "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [20]}},
    }
    assert transfer == {
        "kv_connector": "ScoringHiddenStatesConnector",
        "kv_connector_module_path": "hulder.runtime.connector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"probe_path": str(DEMO), "use_synchronization_lock": False},
    }


def test_a_relative_probe_path_is_made_absolute_since_the_server_may_run_elsewhere(tmp_path, monkeypatch):
    probe_file(tmp_path, layer=3, position=-1, threshold=0.5)
    monkeypatch.chdir(tmp_path)
    _, transfer = parsed(serve_args("probe.npz"))
    assert transfer["kv_connector_extra_config"]["probe_path"] == str(tmp_path / "probe.npz")


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"position": -1, "threshold": 0.5}, "no layer"),
        ({"layer": 3, "threshold": 0.5}, "no position"),
        ({"layer": 3, "position": -5, "threshold": 0.5}, "position -5"),
        ({"layer": 3, "position": -1}, "no threshold"),
    ],
)
def test_a_probe_the_connector_cannot_score_correctly_is_refused(tmp_path, metadata, message):
    with pytest.raises(ValueError, match=message):
        serve_args(probe_file(tmp_path, **metadata))


def test_importing_hulder_runtime_and_building_the_arguments_loads_no_vllm_or_torch():
    code = (
        "import sys\n"
        "from hulder.runtime import serve_args\n"
        f"serve_args({str(DEMO)!r})\n"
        "print(','.join(m for m in ('vllm', 'torch') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


@pytest.mark.vllm
def test_the_named_connector_is_the_class_vllm_will_load():
    pytest.importorskip("vllm")
    import importlib

    _, transfer = parsed(serve_args(DEMO))
    module = importlib.import_module(transfer["kv_connector_module_path"])
    assert hasattr(module, transfer["kv_connector"])
