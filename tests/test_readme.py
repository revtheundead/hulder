"""The README's code runs as written."""

import re
from pathlib import Path

import numpy as np

import hulder
from hulder import cli

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


def test_the_readmes_hulder_check_output_is_what_the_cli_prints(capsys, monkeypatch):
    # README §4 shows real output on the bundled demo; it must stay in step with the CLI.
    [command, shown] = re.findall(r"```\n\$ (hulder check .*?)\n\n(.*?)\n```", README.read_text(), re.DOTALL)[0]
    monkeypatch.chdir(README.parent)
    argv = command.replace("\\\n", " ").split()[1:]
    assert cli.run(argv) == 12
    assert capsys.readouterr().out.rstrip("\n") == shown
