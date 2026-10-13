"""The live tutorial's orchestration, with a fake capture subprocess that saves the bundled
snapshots in. CPU only; the real runs are recorded in docs/results/phase1/live_demo/."""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import live_demo as demo
import numpy as np
import pytest

import hulder
import hulder.capture

MACHINE = {"vllm_version": "0.30.0", "gpu": "Fake RTX"}


@pytest.fixture
def fake_captures(monkeypatch):
    """Each 'subprocess' saves the bundled snapshot for its run, as captured on a fake GPU,
    and writes its manifest. The parent sees the same fake GPU and vLLM version."""
    calls = []

    def run(name, out):
        calls.append(name)
        bundled = hulder.Snapshot.load(demo.DATA / demo.BUNDLED[name])
        hulder.Snapshot(
            ids=bundled.ids,
            labels=bundled.labels,
            activations=bundled.activations,
            metadata={**bundled.metadata, "gpu": MACHINE["gpu"]},
        ).save(out / f"{name}.npz")
        (out / f"{name}.json").write_text(json.dumps({"gpu": MACHINE["gpu"], "prompts_sha256": demo.prompts_sha256()}))
        return 0

    monkeypatch.setattr(demo, "capture_subprocess", run)
    monkeypatch.setattr(demo, "this_machine", lambda: dict(MACHINE))
    return calls


def test_the_demo_captures_each_run_then_checks_and_records_the_verdict(tmp_path, capsys, fake_captures):
    assert demo.main(["--out", str(tmp_path)]) == 0
    assert fake_captures == ["fp16", "fp16_rerun", "awq"]
    printed = capsys.readouterr().out
    assert "Verdict: INCONCLUSIVE" in printed  # the bundled AWQ verdict (docs/reading-a-report.md)
    assert (tmp_path / "check.txt").read_text() in printed
    assert json.loads((tmp_path / "check.json").read_text())["report"]["verdict"] == "INCONCLUSIVE"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert (manifest["status"], manifest["verdict"], manifest["exit_code"]) == ("done", "INCONCLUSIVE", 12)
    assert set(manifest["runs"]) == {"fp16", "fp16_rerun", "awq"}


def test_the_manifest_records_the_hulder_commit_it_ran_from(tmp_path, capsys, fake_captures):
    # The tests run from the repo, a git checkout.
    demo.main(["--out", str(tmp_path)])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=Path(demo.__file__).parent).strip()
    assert manifest["git_commit"] == head
    assert isinstance(manifest["git_dirty"], bool)


@pytest.mark.parametrize("where", ["scripts/live_demo", "examples/live_demo"])
def test_no_commit_is_recorded_for_a_copy_of_the_example_inside_another_repo(tmp_path, where):
    # A copy in some other project's repo must not record that repo's commit, even at the same path.
    copy = tmp_path / where
    copy.mkdir(parents=True)
    assert demo.source_commit(copy) == {"git_commit": None, "git_dirty": None}
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert demo.source_commit(copy) == {"git_commit": None, "git_dirty": None}


def test_the_demos_report_is_what_hulder_check_prints_for_the_same_files(tmp_path, capsys, fake_captures):
    demo.main(["--out", str(tmp_path)])
    capsys.readouterr()
    check_args = json.loads((tmp_path / "manifest.json").read_text())["check_args"]
    for report, extra in (("check.txt", []), ("check.json", ["--json"])):
        run = subprocess.run(["hulder", *check_args, *extra], capture_output=True, text=True, check=False)
        assert run.returncode == 12
        assert (tmp_path / report).read_text() == run.stdout


def test_a_rerun_of_the_demo_reuses_its_snapshots_unless_forced(tmp_path, capsys, fake_captures):
    demo.main(["--out", str(tmp_path)])
    demo.main(["--out", str(tmp_path)])
    assert len(fake_captures) == 3
    demo.main(["--out", str(tmp_path), "--force"])
    assert len(fake_captures) == 6


def test_a_snapshot_without_its_manifest_is_recaptured(tmp_path, capsys, fake_captures):
    demo.main(["--out", str(tmp_path)])
    (tmp_path / "awq.json").unlink()
    assert demo.main(["--out", str(tmp_path)]) == 0
    assert fake_captures[3:] == ["awq"]


def test_a_manifest_without_its_snapshot_is_recaptured(tmp_path, capsys, fake_captures):
    demo.main(["--out", str(tmp_path)])
    (tmp_path / "fp16_rerun.npz").unlink()
    assert demo.main(["--out", str(tmp_path)]) == 0
    assert fake_captures[3:] == ["fp16_rerun"]


@pytest.mark.parametrize(
    ("change", "keys"),
    [
        ({"gpu": "NVIDIA GeForce RTX 4070"}, ["gpu"]),
        ({"vllm_version": "0.31.0"}, ["vllm_version"]),
        ({"prompts_sha256": "0" * 64}, ["prompts_sha256"]),
    ],
)
def test_a_snapshot_from_another_machine_or_prompt_set_is_refused(
    tmp_path, capsys, fake_captures, monkeypatch, change, keys
):
    demo.main(["--out", str(tmp_path)])
    capsys.readouterr()
    here = {**MACHINE, "prompts_sha256": demo.prompts_sha256(), **change}
    monkeypatch.setattr(demo, "this_machine", lambda: {k: here[k] for k in MACHINE})
    monkeypatch.setattr(demo, "prompts_sha256", lambda: here["prompts_sha256"])
    assert demo.main(["--out", str(tmp_path)]) == 1
    assert len(fake_captures) == 3
    err = capsys.readouterr().err
    assert all(f"{key}:" in err for key in keys) and "--force" in err
    assert demo.main(["--out", str(tmp_path), "--force"]) == 0
    assert len(fake_captures) == 6


def test_another_runs_snapshot_under_this_runs_name_is_refused(tmp_path, capsys, fake_captures):
    demo.main(["--out", str(tmp_path)])
    capsys.readouterr()
    for suffix in (".npz", ".json"):
        shutil.copy(tmp_path / f"awq{suffix}", tmp_path / f"fp16{suffix}")
    assert demo.main(["--out", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert all(f"{key}:" in err for key in ("model", "revision", "quantization")) and "--force" in err
    assert len(fake_captures) == 3


def test_a_damaged_snapshot_is_refused_before_anything_is_captured(tmp_path, capsys, fake_captures):
    demo.main(["--out", str(tmp_path)])
    capsys.readouterr()
    (tmp_path / "awq.npz").write_bytes((tmp_path / "awq.npz").read_bytes()[:5000])
    assert demo.main(["--out", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "awq.npz" in err and "--force" in err
    assert len(fake_captures) == 3


def test_a_failed_capture_exits_1_skips_the_check_and_records_the_failure(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(demo, "capture_subprocess", lambda name, out: 1)
    assert demo.main(["--out", str(tmp_path)]) == 1
    assert "the fp16 capture failed" in capsys.readouterr().err
    assert not (tmp_path / "check.txt").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "capture failed" and "fp16" in manifest["error"]
    assert "verdict" not in manifest


def test_a_failed_check_exits_1_prints_its_error_once_and_records_the_failure(
    tmp_path, capsys, fake_captures, monkeypatch
):
    # The awq 'capture' succeeds but its file is damaged, so the check itself fails.
    demo.main(["--out", str(tmp_path)])
    capsys.readouterr()
    run = demo.capture_subprocess

    def damaging(name, out):
        result = run(name, out)
        awq = tmp_path / "awq.npz"
        awq.write_bytes(awq.read_bytes()[:5000])
        return result

    monkeypatch.setattr(demo, "capture_subprocess", damaging)
    assert demo.main(["--out", str(tmp_path), "--force"]) == 1
    err = capsys.readouterr().err
    assert err.count("awq.npz") == 1 and "--force" in err
    assert not (tmp_path / "check.txt").exists() and not (tmp_path / "check.json").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "check failed" and "awq.npz" in manifest["error"]
    assert "verdict" not in manifest


def test_a_refused_resume_replaces_the_previous_runs_report(tmp_path, capsys, fake_captures):
    demo.main(["--out", str(tmp_path)])
    (tmp_path / "awq.npz").write_bytes(b"")
    assert demo.main(["--out", str(tmp_path)]) == 1
    assert not (tmp_path / "check.txt").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "resume refused"


# CaptureConfig's defaults for the keys the demo's runs leave unset, so the metadata test
# runs without vLLM; the --runvllm test below checks they are still CaptureConfig's.
CONFIG_DEFAULTS = {"quantization": None, "dtype": "float16", "batch_size": 8, "enable_thinking": None}


def test_the_demo_snapshot_metadata_is_the_bundled_snapshots_plus_the_gpu():
    for name, bundled in demo.BUNDLED.items():
        expected = hulder.Snapshot.load(demo.DATA / bundled).metadata
        expected = {k: v for k, v in expected.items() if k not in ("splits_sha256", "prompt_ids_sha256")}
        cfg = SimpleNamespace(**{**CONFIG_DEFAULTS, **demo.RUNS[name]})
        captured = SimpleNamespace(gpu="Fake RTX", enable_thinking=None)
        got = hulder.capture.snapshot_metadata(cfg, captured, expected["base_model"], expected["vllm_version"])
        assert got == {**expected, "gpu": "Fake RTX"}


@pytest.mark.vllm
def test_the_config_defaults_the_metadata_test_assumes_are_capture_configs():
    pytest.importorskip("vllm")
    for run in demo.RUNS.values():
        cfg = hulder.capture.CaptureConfig(**run)
        assert {key: getattr(cfg, key) for key in CONFIG_DEFAULTS} == {
            **CONFIG_DEFAULTS,
            **{key: run[key] for key in CONFIG_DEFAULTS if key in run},
        }


def test_each_capture_runs_this_script_in_a_fresh_python_process(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(
        demo.subprocess, "run", lambda command, check: commands.append(command) or SimpleNamespace(returncode=3)
    )
    assert demo.capture_subprocess("awq", tmp_path) == 3
    script = str(Path(demo.__file__).resolve())
    assert commands == [[sys.executable, script, "--capture", "awq", "--out", str(tmp_path)]]


def test_a_capture_subprocess_that_raises_exits_non_zero(tmp_path, monkeypatch):
    # The parent sees a failed capture only through the child's exit code, so the child must
    # let the exception out (python exits 1 on it) rather than return 0.
    def no_gpu(name, out):
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(demo, "capture_run", no_gpu)
    with pytest.raises(RuntimeError, match="CUDA"):
        demo.main(["--capture", "fp16", "--out", str(tmp_path)])


@pytest.fixture
def fake_engine(monkeypatch):
    """capture_run's vLLM, torch and Hugging Face calls replaced, so it runs without the
    [vllm] extra: the 'engine' returns zero activations captured on a fake GPU."""

    @dataclass(frozen=True)
    class CaptureConfig:
        model: str
        revision: str
        layers: tuple
        positions: tuple
        system_prompt: str
        quantization: str | None = None
        dtype: str = "float16"
        batch_size: int = 8
        enable_thinking: bool | None = None

    def capture(texts, cfg):
        return SimpleNamespace(
            activations=np.zeros((len(texts), 1, 1, 4), dtype=np.float16),
            position_tokens=[["\n"]] * len(texts),
            gpu="Fake RTX",
            enable_thinking=None,
        )

    fakes = {
        "torch": SimpleNamespace(version=SimpleNamespace(cuda="13.0")),
        # hulder.capture loads CaptureConfig and capture from here on first use.
        "hulder.capture.hidden_states": SimpleNamespace(CaptureConfig=CaptureConfig, capture=capture),
    }
    for name, module in fakes.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(hulder.capture, "base_model_for", lambda model, revision, quantization: (model, None))
    monkeypatch.setattr(demo, "version", lambda package: "0.30.0")


def test_capture_run_saves_the_snapshot_and_its_manifest(tmp_path, fake_engine):
    demo.capture_run("awq", tmp_path)
    snapshot = hulder.Snapshot.load(tmp_path / "awq.npz")
    assert snapshot.ids.tolist() == [r["id"] for r in demo.prompts()]
    assert (snapshot.metadata["quantization"], snapshot.metadata["gpu"]) == ("awq", "Fake RTX")
    manifest = json.loads((tmp_path / "awq.json").read_text())
    assert manifest["capture_config"]["model"] == demo.AWQ["model"]
    assert manifest["prompts_sha256"] == demo.prompts_sha256()


def test_a_capture_interrupted_before_its_manifest_lands_leaves_the_old_one_whole(tmp_path, fake_engine, monkeypatch):
    # The manifest is written atomically like the snapshot, so a killed child never leaves half a .json.
    (tmp_path / "awq.json").write_text('{"old": true}')
    replace = os.replace

    def crash_on_json(src, dst):
        if str(dst).endswith(".json"):
            raise OSError("interrupted before the rename")
        replace(src, dst)

    monkeypatch.setattr(demo.os, "replace", crash_on_json)
    with pytest.raises(OSError, match="interrupted"):
        demo.capture_run("awq", tmp_path)
    assert json.loads((tmp_path / "awq.json").read_text()) == {"old": True}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["awq.json", "awq.npz"]


def test_importing_the_demo_loads_no_vllm_or_torch():
    # The parent process only orchestrates and checks; vLLM loads in each capture subprocess.
    code = "import sys, live_demo; print(','.join(m for m in ('vllm', 'torch') if m in sys.modules))"
    here = Path(demo.__file__).parent
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=here)
    assert run.stdout.strip() == ""


def test_the_prompts_are_the_bundled_demo_snapshots_examples_in_order():
    rows = demo.prompts()
    reference = hulder.Snapshot.load(demo.DATA / "demo_fp16.npz")
    assert [r["id"] for r in rows] == list(reference.ids)
    assert [r["label"] for r in rows] == reference.labels.tolist()
    assert all(r["text"].strip() for r in rows)


def test_the_example_imports_only_hulders_public_api():
    # The public API (docs/design.md §4): hulder.__all__ and hulder.capture.__all__, and the hulder command.
    source = Path(demo.__file__).read_text()
    imports = [line.strip() for line in source.splitlines() if "hulder" in line and "import" in line]
    assert imports == [
        "import hulder",
        "from hulder.capture import CaptureConfig, base_model_for, capture, snapshot_metadata",
    ]
