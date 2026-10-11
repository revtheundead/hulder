"""The live tutorial's orchestration, with a fake capture subprocess that copies the bundled
snapshots in. CPU only; the real run is recorded in docs/results/phase1/live_demo/."""

import json
import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import pytest

import hulder
from hulder.capture import demo

BUNDLED = {"fp16": "demo_fp16.npz", "fp16_rerun": "demo_fp16_run1.npz", "awq": "demo_awq.npz"}


@pytest.fixture
def fake_captures(monkeypatch):
    """Each 'subprocess' copies the bundled snapshot for its run and writes a manifest."""
    calls = []

    def run(command, check=False):
        name, out = command[command.index("--capture") + 1], Path(command[command.index("--out") + 1])
        calls.append(name)
        shutil.copy(str(files("hulder.data") / BUNDLED[name]), out / f"{name}.npz")
        (out / f"{name}.json").write_text(json.dumps({"gpu": "Fake RTX"}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(demo.subprocess, "run", run)
    return calls


def test_the_demo_captures_each_run_then_checks_and_records_the_verdict(tmp_path, capsys, fake_captures):
    assert demo.main(["--out", str(tmp_path)]) == 0
    assert fake_captures == ["fp16", "fp16_rerun", "awq"]
    printed = capsys.readouterr().out
    assert "Verdict: INCONCLUSIVE" in printed  # the bundled AWQ verdict (README §4)
    assert (tmp_path / "check.txt").read_text() in printed
    assert json.loads((tmp_path / "check.json").read_text())["report"]["verdict"] == "INCONCLUSIVE"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert (manifest["verdict"], manifest["exit_code"]) == ("INCONCLUSIVE", 12)
    assert set(manifest["runs"]) == {"fp16", "fp16_rerun", "awq"}


def test_a_rerun_of_the_demo_reuses_its_snapshots_unless_forced(tmp_path, capsys, fake_captures):
    demo.main(["--out", str(tmp_path)])
    demo.main(["--out", str(tmp_path)])
    assert len(fake_captures) == 3
    demo.main(["--out", str(tmp_path), "--force"])
    assert len(fake_captures) == 6


def test_a_failed_capture_exits_1_and_skips_the_check(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(demo.subprocess, "run", lambda command, check=False: SimpleNamespace(returncode=1))
    assert demo.main(["--out", str(tmp_path)]) == 1
    assert "the fp16 capture failed" in capsys.readouterr().err
    assert not (tmp_path / "check.txt").exists()


@pytest.mark.vllm
def test_the_demo_snapshot_metadata_is_the_bundled_snapshots_plus_the_gpu():
    hidden_states = pytest.importorskip("hulder.capture.hidden_states")
    for name, bundled in (("fp16", "demo_fp16.npz"), ("awq", "demo_awq.npz")):
        expected = hulder.Snapshot.load(str(files("hulder.data") / bundled)).metadata
        expected = {k: v for k, v in expected.items() if k not in ("splits_sha256", "prompt_ids_sha256")}
        cfg = hidden_states.CaptureConfig(**demo.RUNS[name])
        got = demo.snapshot_metadata(cfg, expected["base_model"], "Fake RTX", expected["vllm_version"])
        assert got == {**expected, "gpu": "Fake RTX"}


def test_importing_the_demo_loads_no_vllm_or_torch():
    # The parent process only orchestrates and checks; vLLM loads in each capture subprocess.
    code = "import sys, hulder.capture.demo; print(','.join(m for m in ('vllm', 'torch') if m in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == ""
