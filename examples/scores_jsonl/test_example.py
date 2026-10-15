"""The bring-your-own-scores example: its files come from make_example.py, and the README's command
says THRESHOLD STALE with a recalibrated threshold. CPU only."""

from pathlib import Path

import make_example

from hulder import cli

HERE = Path(__file__).resolve().parent
ROLES = ("reference", "rerun", "candidate")


def test_the_committed_files_are_what_make_example_writes(tmp_path):
    make_example.write(tmp_path)
    for role in ROLES:
        assert (tmp_path / f"{role}.jsonl").read_bytes() == (HERE / f"{role}.jsonl").read_bytes()


def test_the_readmes_command_uses_the_fitted_threshold_and_says_threshold_stale(tmp_path, monkeypatch, capsys):
    threshold = make_example.write(tmp_path)
    command = (HERE / "README.md").read_text().split("```bash\n", 1)[1].split("\n```", 1)[0]
    args = command.replace("\\\n", " ").split()
    assert args[:2] == ["hulder", "check"]
    assert args[args.index("--threshold") + 1] == repr(threshold)
    monkeypatch.chdir(HERE)
    assert cli.run(args[1:]) == 10
    out = capsys.readouterr().out
    assert "Verdict: THRESHOLD STALE" in out and "Recalibrated threshold  0.37 " in out
