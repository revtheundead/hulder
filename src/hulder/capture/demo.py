"""Live tutorial: capture the bundled demo prompts with vLLM under fp16, an fp16 rerun and
AWQ int4, then check them with the bundled demo probe (docs/tutorial-live.md).

    pip install "hulder[vllm]"
    python -m hulder.capture.demo

Each capture runs in its own subprocess, since one process can't reliably free a vLLM
engine's GPU memory for the next. The snapshots, the report and a manifest go to --out.
A snapshot already there with its manifest is reused, so an interrupted run resumes, but
only if it was captured with this run's config, vLLM, GPU and prompts; --force recaptures.
The live verdict can differ from the bundled one (docs/tutorial-live.md says why), so any
verdict exits 0. Exit 1 is a failed capture or check.
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import numpy as np

from hulder import cli
from hulder.capture.metadata import snapshot_metadata
from hulder.gate.atomic import write_text_atomically
from hulder.gate.probe import Probe
from hulder.gate.snapshot import Snapshot

DATA = files("hulder.data")
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
# What the parent writes to --out besides the snapshots.
REPORTS = ("check.txt", "check.json", "manifest.json")
# What a snapshot in --out must match to be reused, besides this_machine() and the prompts.
RESUME_KEYS = ("model", "revision", "quantization")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hulder.capture.demo", description=__doc__.split("\n\n")[0])
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
    for name in RUNS:
        if name not in todo:
            print(f"{name}: reusing {out / f'{name}.npz'}", flush=True)
            continue
        print(f"{name}: capturing {len(prompts())} prompts in a fresh vLLM engine", flush=True)
        command = [sys.executable, "-m", "hulder.capture.demo", "--capture", name, "--out", str(out)]
        if (code := subprocess.run(command, check=False).returncode) != 0:
            return failed(
                out, "capture failed", f"the {name} capture failed with exit code {code}; its output is above"
            )
    return check(out)


def reusable(out: Path, name: str) -> bool:
    """Whether out holds a complete capture of this run, the snapshot and its manifest. Raises
    ValueError, suggesting --force, if they are damaged or were captured with anything else."""
    snapshot, manifest = out / f"{name}.npz", out / f"{name}.json"
    if not (snapshot.exists() and manifest.exists()):
        return False
    try:
        recorded = {
            **Snapshot.load(snapshot).metadata,
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
    return [json.loads(line) for line in (DATA / "demo_prompts.jsonl").read_text().splitlines()]


def prompts_sha256() -> str:
    return hashlib.sha256((DATA / "demo_prompts.jsonl").read_bytes()).hexdigest()


def capture_run(name: str, out: Path) -> None:
    """Capture one config and save it as a Snapshot plus a JSON manifest. Imports vLLM."""
    import torch

    from hulder.capture.base_model import base_model_for
    from hulder.capture.hidden_states import CaptureConfig, capture

    cfg = CaptureConfig(**RUNS[name])
    rows = prompts()
    start = time.time()
    captured = capture([r["text"] for r in rows], cfg)
    seconds = time.time() - start
    base_model, warning = base_model_for(cfg.model, cfg.revision, cfg.quantization)
    if warning:
        raise SystemExit(warning)
    Snapshot(
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
    """hulder check on the captures with the bundled probe, through the same code as the CLI;
    writes check.txt, check.json and manifest.json to out."""
    probe_path = str(DATA / "probe_demo.npz")
    files = {"reference": out / "fp16.npz", "candidate": out / "awq.npz", "rerun": out / "fp16_rerun.npz"}
    check_args = ["check", "--probe", probe_path, "--reference", str(files["reference"])]
    check_args += ["--candidate", str(files["candidate"]), "--noise-floor", str(files["rerun"])]
    try:
        probe = Probe.load(probe_path)
        threshold, target_fpr = probe.metadata["threshold"], probe.metadata["target_fpr"]
        report = cli.check_files(files["reference"], files["candidate"], files["rerun"], probe, threshold, target_fpr)
    except cli.TOOL_FAILURES as e:
        cli.fail_tool(e)
        print("If a snapshot in --out is damaged, rerun with --force to recapture it.", file=sys.stderr)
        record(out, status="check failed", error=str(e), check_args=check_args)
        return 1
    inputs = cli.report_inputs(*(str(path) for path in files.values()), probe_path)
    text = cli.render(report, inputs) + "\n"
    write_text_atomically(out / "check.txt", text)
    write_text_atomically(out / "check.json", cli.to_json(report, inputs) + "\n")
    print(text, end="")
    record(
        out,
        status="done",
        check_args=check_args,
        exit_code=report.verdict.exit_code,
        verdict=report.verdict.value,
    )
    print(f"\nThe live verdict is {report.verdict.value}. Snapshots, report and manifest are in {out}.")
    return 0


def failed(out: Path, status: str, error: str) -> int:
    print(f"hulder: error: {error}", file=sys.stderr)
    record(out, status=status, error=error)
    return 1


def record(out: Path, status: str, **fields) -> None:
    """manifest.json: how this run ended ("done" or what failed) and each capture's manifest."""
    manifest = {
        "command": "python -m hulder.capture.demo",
        "status": status,
        **fields,
        "runs": {
            name: json.loads((out / f"{name}.json").read_text()) for name in RUNS if (out / f"{name}.json").exists()
        },
    }
    write_text_atomically(out / "manifest.json", json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
