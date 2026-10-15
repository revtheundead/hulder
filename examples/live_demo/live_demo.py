"""Live tutorial: capture the demo prompts with vLLM under fp16, an fp16 rerun and AWQ int4,
then check them with hulder's bundled demo probe (README.md next to this file).

From a clone of the hulder repository, at its root:

    source scripts/dev-env.sh && uv sync
    uv run python examples/live_demo/live_demo.py

Each capture runs in its own subprocess, since one process can't reliably free a vLLM
engine's GPU memory for the next. The snapshots, the report and a manifest go to --out.
A snapshot already there with its manifest is reused, so an interrupted run resumes, but
only if it was captured with this run's config, vLLM, GPU and prompts; --force recaptures.
The live verdict can differ from the bundled one (README.md says why), so any verdict
exits 0. Exit 1 is a failed capture or check.

Uses only hulder's public API: hulder.Snapshot, hulder.capture and the hulder check command.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

import hulder

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "demo_prompts.jsonl"
# The demo probe and snapshots that ship with hulder, read from this checkout's source tree.
DATA = HERE.parents[1] / "src" / "hulder" / "data"
PROBE = DATA / "probe_demo.npz"
COMMAND = "python examples/live_demo/live_demo.py"
# The configs of the bundled demo snapshots (experiments/phase1/probe/extract.py), extracting
# only the probe's layer and position.
FP16 = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "layers": (20,),
    "positions": (-1,),
    "system_prompt": "You are a helpful assistant.",
}
AWQ = {
    **FP16,
    "model": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    "revision": "3ecffa0ceb27851800f45519bab9c457a04405e1",
    "quantization": "awq",
}
# The rerun is the fp16 config again in a fresh engine, for the noise floor.
RUNS = {"fp16": FP16, "fp16_rerun": FP16, "awq": AWQ}
# The bundled demo snapshot each run reproduces, up to batch composition (README.md).
BUNDLED = {"fp16": "demo_fp16.npz", "fp16_rerun": "demo_fp16_run1.npz", "awq": "demo_awq.npz"}
# What the parent writes to --out besides the snapshots.
REPORTS = ("check.txt", "check.json", "manifest.json")
# What a snapshot in --out must match to be reused, besides this_machine() and the prompts.
RESUME_KEYS = ("model", "revision", "quantization")
# hulder check's exit codes for a verdict, PASS to INCONCLUSIVE (docs/design.md §4). Anything else failed.
VERDICT_CODES = (0, 10, 11, 12)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=COMMAND, description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="hulder-demo", help="directory for snapshots, report and manifest")
    parser.add_argument("--force", action="store_true", help="recapture snapshots already in --out")
    parser.add_argument("--capture", choices=RUNS, help=argparse.SUPPRESS)  # one subprocess's run
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.capture:
        capture_run(args.capture, out)
        return 0
    print(f"{shutil.disk_usage(out).free / 1e9:.0f} GB free in {out.resolve()}", flush=True)
    for report in REPORTS:  # a previous run's, which this run replaces whatever happens
        (out / report).unlink(missing_ok=True)
    try:
        todo = [name for name in RUNS if args.force or not reusable(out, name)]
    except ValueError as e:
        return failed(out, "resume refused", str(e))
    n_prompts = len(prompts())
    for name in RUNS:
        if name not in todo:
            print(f"{name}: reusing {out / f'{name}.npz'}", flush=True)
            continue
        print(f"{name}: capturing {n_prompts} prompts in a fresh vLLM engine", flush=True)
        if (code := capture_subprocess(name, out)) != 0:
            return failed(
                out, "capture failed", f"the {name} capture failed with exit code {code}; its output is above"
            )
    return check(out)


def capture_subprocess(name: str, out: Path) -> int:
    """Run one capture in a fresh Python process, whose exit frees its engine's GPU memory."""
    command = [sys.executable, str(Path(__file__).resolve()), "--capture", name, "--out", str(out)]
    return subprocess.run(command, check=False).returncode


def reusable(out: Path, name: str) -> bool:
    """Whether out holds a complete capture of this run, the snapshot and its manifest. Raises
    ValueError, suggesting --force, if they are damaged or were captured with anything else."""
    snapshot, manifest = out / f"{name}.npz", out / f"{name}.json"
    if not (snapshot.exists() and manifest.exists()):
        return False
    try:
        recorded = {
            **hulder.Snapshot.load(snapshot).metadata,
            "prompts_sha256": json.loads(manifest.read_text()).get("prompts_sha256"),
        }
    except ValueError as e:
        raise ValueError(f"{e}. Rerun with --force to recapture it") from e
    expected = {
        **{key: RUNS[name].get(key) for key in RESUME_KEYS},
        **this_machine(),
        "prompts_sha256": prompts_sha256(),
    }
    differing = [
        f"  {k}: recorded {recorded.get(k)!r}, this run {v!r}" for k, v in expected.items() if recorded.get(k) != v
    ]
    if differing:
        raise ValueError(
            "\n".join([f"{snapshot} was captured with something else, so it can't be reused:", *differing])
            + "\nRerun with --force to recapture it, or pick another --out."
        )
    return True


def this_machine() -> dict:
    """The vLLM version and GPU a capture here would record. The GPU is queried in a
    subprocess, so this process never imports torch."""
    query = "import torch; print(torch.cuda.get_device_name())"
    gpu = subprocess.run([sys.executable, "-c", query], capture_output=True, text=True, check=True).stdout.strip()
    return {"vllm_version": version("vllm"), "gpu": gpu}


def prompts() -> list[dict]:
    return [json.loads(line) for line in PROMPTS.read_text().splitlines()]


def prompts_sha256() -> str:
    return hashlib.sha256(PROMPTS.read_bytes()).hexdigest()


def capture_run(name: str, out: Path) -> None:
    """Capture one config and save it as a Snapshot plus a JSON manifest. Imports vLLM."""
    import torch

    from hulder.capture import CaptureConfig, base_model_for, capture, snapshot_metadata

    cfg = CaptureConfig(**RUNS[name])
    rows = prompts()
    start = time.time()
    captured = capture([r["text"] for r in rows], cfg)
    seconds = time.time() - start
    base_model, warning = base_model_for(cfg.model, cfg.revision, cfg.quantization)
    if warning:  # what it assumed; check warns again if no base_model was recorded
        print(f"warning: {warning}", file=sys.stderr, flush=True)
    hulder.Snapshot(
        ids=np.array([r["id"] for r in rows]),
        labels=np.array([r["label"] for r in rows]),
        activations=captured.activations[:, 0, 0].astype(np.float16),  # the config's only layer and position
        metadata=snapshot_metadata(cfg, captured, base_model, version("vllm")),
    ).save(out / f"{name}.npz")
    manifest = {
        "capture_config": asdict(cfg),
        "gpu": captured.gpu,
        "seconds": round(seconds, 1),
        "last_tokens": sorted({tokens[0] for tokens in captured.position_tokens}),
        "packages": {p: version(p) for p in ("hulder", "vllm", "torch", "numpy")},
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "prompts_sha256": prompts_sha256(),
    }
    write_text_atomically(out / f"{name}.json", json.dumps(manifest, indent=2) + "\n")


def check(out: Path) -> int:
    """hulder check on the captures with the bundled demo probe, run twice: as printed, for
    check.txt, and with --json, for check.json. Writes manifest.json too."""
    check_args = ["check", "--probe", str(PROBE), "--reference", str(out / "fp16.npz")]
    check_args += ["--candidate", str(out / "awq.npz"), "--noise-floor", str(out / "fp16_rerun.npz")]
    runs = []
    for extra in ([], ["--json"]):
        run = hulder_cli([*check_args, *extra])
        if run.returncode not in VERDICT_CODES:
            print(run.stderr, end="", file=sys.stderr)
            print("If a snapshot in --out is damaged, rerun with --force to recapture it.", file=sys.stderr)
            record(out, status="check failed", error=run.stderr.strip(), check_args=check_args)
            return 1
        runs.append(run)
    text, as_json = runs
    write_text_atomically(out / "check.txt", text.stdout)
    write_text_atomically(out / "check.json", as_json.stdout)
    print(text.stdout, end="")
    verdict = json.loads(as_json.stdout)["verdict"]
    record(out, status="done", check_args=check_args, exit_code=as_json.returncode, verdict=verdict)
    print(f"\nThe live verdict is {verdict}. Snapshots, report and manifest are in {out}.")
    return 0


def hulder_cli(args: list[str]) -> subprocess.CompletedProcess:
    """The hulder command installed with this interpreter's hulder, so both are the same version."""
    executable = Path(sysconfig.get_path("scripts")) / "hulder"
    return subprocess.run([str(executable), *args], capture_output=True, text=True, check=False)


def failed(out: Path, status: str, error: str) -> int:
    print(f"hulder: error: {error}", file=sys.stderr)
    record(out, status=status, error=error)
    return 1


def record(out: Path, status: str, **fields) -> None:
    """manifest.json: how this run ended ("done" or what failed) and each capture's manifest."""
    manifest = {
        "command": COMMAND,
        "status": status,
        **fields,
        **source_commit(),
        "runs": {
            name: json.loads((out / f"{name}.json").read_text()) for name in RUNS if (out / f"{name}.json").exists()
        },
    }
    write_text_atomically(out / "manifest.json", json.dumps(manifest, indent=2) + "\n")


def source_commit(here: Path = HERE) -> dict:
    """The git commit of the hulder checkout this example runs from, and whether tracked files
    differ from it. None for both unless here is examples/live_demo/ of a checkout that also
    holds src/hulder/, so a copy inside another project's repo doesn't record that repo's commit."""

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(here), *args], text=True, stderr=subprocess.DEVNULL).strip()

    try:
        top = Path(git("rev-parse", "--show-toplevel"))
        if top / "examples" / "live_demo" != here or not (top / "src" / "hulder").is_dir():
            raise ValueError("not hulder's own checkout")
        return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "-uno"))}
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {"git_commit": None, "git_dirty": None}


def write_text_atomically(path: Path, text: str) -> None:
    """Write text to path whole or not at all: to a temporary file in path's directory, then
    renamed over path, so an interrupted run never leaves half a file."""
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    sys.exit(main())
