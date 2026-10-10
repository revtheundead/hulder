"""The README's code runs as written."""

import re
from pathlib import Path

import numpy as np

import hulder

README = Path(__file__).parents[1] / "README.md"


def test_the_readmes_snapshot_example_saves_a_snapshot_that_loads(tmp_path, monkeypatch):
    [code] = re.findall(r"```python\n(.*?hulder\.Snapshot\(.*?)```", README.read_text(), re.DOTALL)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "snaps").mkdir()
    labels = np.r_[np.zeros(6, int), np.ones(4, int)]
    namespace = {
        "ids": np.array([f"calib/{i}" for i in range(10)]),
        "labels": labels,
        "acts": np.random.default_rng(0).normal(size=(10, 8)),
    }
    exec(code, namespace)  # noqa: S102, the README is ours
    snapshot = hulder.Snapshot.load(tmp_path / "snaps/bf16.npz")
    assert snapshot.activations.shape == (10, 8)
    assert snapshot.metadata["layer"] == 20
