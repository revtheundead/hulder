"""The README's code runs as written, and its report walkthrough quotes the real quickstart."""

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


# Numbers the walkthrough uses to explain, not to quote: the CI level and the AUROC scale.
EXPLANATORY = {"95%", "1.0", "0.5"}
NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)*%?(?![\w])")


def walkthrough() -> tuple[str, str]:
    """The report shown under "How to read a report", and the prose around it."""
    section = re.search(r"^## How to read a report\n(.*?)^## ", README.read_text(), re.DOTALL | re.MULTILINE).group(1)
    [report] = re.findall(r"```\n(.*?)\n```", section, re.DOTALL)
    prose = re.sub(r"```.*?```", "", section, flags=re.DOTALL)
    return report, re.sub(r"(?m)^\d+\. ", "", prose)  # list markers aren't quoted numbers


def quickstart(capsys) -> str:
    assert cli.run(["quickstart"]) == 0
    return capsys.readouterr().out


def test_the_walkthroughs_report_is_check_2_of_the_quickstart_as_printed(capsys):
    report, _ = walkthrough()
    assert "\n" + report + "\n" in quickstart(capsys)


def test_every_number_the_walkthrough_quotes_is_in_the_quickstart_output(capsys):
    _, prose = walkthrough()
    printed = set(re.findall(r"[-+]?\d+(?:[.,]\d+)*%?", quickstart(capsys)))
    quoted = set(NUMBER.findall(prose)) - EXPLANATORY
    assert quoted, "the walkthrough quotes no numbers"
    assert sorted(quoted - printed) == []


def test_the_readmes_hulder_check_command_prints_the_walkthroughs_report(capsys, monkeypatch):
    [command] = re.findall(r"```\n\$ (hulder check .*?)\n```", README.read_text(), re.DOTALL)
    monkeypatch.chdir(README.parent)
    argv = command.replace("\\\n", " ").split()[1:]
    assert cli.run(argv) == 12
    report, _ = walkthrough()
    assert capsys.readouterr().out.rstrip("\n").replace("src/hulder/data/", "") == report
