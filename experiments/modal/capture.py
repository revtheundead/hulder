"""Activation captures on a Modal L4 (Ada, 24 GB), one container per serving config.

From the repository root, after `modal token new` (the token stays in .cache/, see
scripts/dev-env.sh):

    source scripts/dev-env.sh
    uv run --with modal==1.6.1 modal run experiments/modal/capture.py
    uv run python experiments/modal/compare.py .cache/modal-runs/<run id>

The image is this repository's uv.lock with the vllm extra (vLLM 0.30.0) on Python 3.12.14.
Model weights download once, on a CPU container, into the hulder-hf Volume, which also keeps
vLLM's compile caches. Each config then runs in its own L4 container with the Hub offline.

Stop rules: each config's container times out after 20 minutes, and no config starts if the
estimated cost so far plus that config's worst case would pass $3. Estimates use Modal's published
rates (smoke_lib.py). The actual charge is on Modal's usage page.

Writes, per config, the snapshots (one per layer), the full log and a manifest to
--out/<run id>/<config>/, and the run's manifest to --out/<run id>/manifest.json. The tracked tree
must be clean, so the manifest's commit is what ran.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import modal
import smoke_lib as lib

LOCAL = modal.is_local()
ROOT = Path(__file__).resolve().parents[2] if LOCAL else Path("/")
HERE = Path(__file__).resolve().parent
CACHE, RUNS = "/vol/hf", "/vol/runs"
# Set at runtime, never in the image: Modal refuses to mount a Volume on a non-empty path, and
# with these in the image's environment the mount path was already populated at /cache and /vol/hf.
CACHE_ENV = {
    "HF_HOME": f"{CACHE}/hf",
    "VLLM_CACHE_ROOT": f"{CACHE}/vllm",
    "TRITON_CACHE_DIR": f"{CACHE}/triton",
    "XDG_CACHE_HOME": f"{CACHE}/xdg",
    "FLASHINFER_WORKSPACE_BASE": CACHE,
}
# Engine settings for every capture, recorded in each manifest. On Ada, vLLM 0.30.0's warmup samples
# through FlashInfer's top-k/top-p kernel, which JIT-compiles with nvcc, absent from this image. Sampling
# happens after the prefill the activations come from, so it can't change them. Turing (the local
# RTX 2070) falls back from that sampler anyway.
ENGINE_ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
UV_VERSION = "0.12.21"  # the uv that locked uv.lock


def build_wheel() -> Path:
    """hulder as a wheel built from this checkout, so the image has its package metadata too
    (hulder.__version__ reads it). The tree is clean when main runs, so it is the recorded commit."""
    dist = ROOT / ".cache" / "modal" / "dist"
    shutil.rmtree(dist, ignore_errors=True)
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(dist)], cwd=ROOT, check=True, capture_output=True)
    [wheel] = dist.glob("hulder-*.whl")
    return wheel


# Inside a container the image is already built, so the wheel is never read there.
WHEEL = build_wheel() if LOCAL else Path("/root/hulder.whl")

image = (
    modal.Image.debian_slim(python_version="3.12.14")  # .python-version
    .apt_install("gcc")  # Triton compiles a C launcher at runtime
    .uv_sync(
        str(ROOT),
        extras=["vllm"],
        # The project pins a uv-managed interpreter for local work; here the image's Python serves.
        extra_options="--no-default-groups --python-preference=system",
        uv_version=UV_VERSION,
    )
    .add_local_file(str(WHEEL), f"/wheels/{WHEEL.name}", copy=True)
    .run_commands(f"/.uv/uv pip install --python /.uv/.venv/bin/python --no-deps /wheels/{WHEEL.name}")
    .add_local_file(str(HERE / "smoke_lib.py"), "/root/smoke_lib.py")
    .add_local_file(str(HERE / "worker.py"), "/root/worker.py")
)
app = modal.App("hulder-capture", image=image)
hf = modal.Volume.from_name("hulder-hf", create_if_missing=True)
runs = modal.Volume.from_name("hulder-runs", create_if_missing=True)


@app.function(
    cpu=lib.DOWNLOAD_CPU,
    memory=lib.DOWNLOAD_MEMORY_GIB * 1024,
    volumes={CACHE: hf},
    timeout=30 * 60,
    retries=0,
    scaledown_window=2,
)
def download(models: list[tuple[str, str]]) -> dict:
    """Fetch each model at its pinned revision into the Volume's HF cache, on CPU."""
    os.environ.update(CACHE_ENV)  # before huggingface_hub reads HF_HOME
    from huggingface_hub import snapshot_download

    start = time.time()
    for model, revision in models:
        snapshot_download(model, revision=revision)
    hf.commit()
    return {"seconds": round(time.time() - start, 1)}


@app.function(
    gpu="L4",
    cpu=lib.GPU_CPU,
    memory=lib.GPU_MEMORY_GIB * 1024,
    volumes={CACHE: hf, RUNS: runs},
    timeout=lib.CONFIG_TIMEOUT_S,
    retries=0,
    scaledown_window=2,
)
def capture_config(name: str, spec: dict, run_id: str) -> dict:
    """Run worker.py on one config in a subprocess, Hub offline, and return its files."""
    start = time.time()
    out = Path(RUNS) / run_id / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    query = "--query-gpu=name,uuid,driver_version,memory.total"
    gpu = subprocess.run(
        ["nvidia-smi", query, "--format=csv,noheader"], capture_output=True, text=True, check=False
    ).stdout
    with open(out / "capture.log", "w") as log:
        worker = subprocess.run(
            [sys.executable, "/root/worker.py", str(out / "spec.json"), str(out)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **CACHE_ENV, **ENGINE_ENV, "HF_HUB_OFFLINE": "1"},
            check=False,
        )
    text = (out / "capture.log").read_text(errors="replace")
    manifest = {
        "config": name,
        "returncode": worker.returncode,
        "nvidia_smi": gpu.strip(),
        "engine_env": ENGINE_ENV,
        "function_seconds": round(time.time() - start, 1),
        "log": lib.log_facts(text),
    }
    if (out / "worker.json").exists():
        manifest["worker"] = json.loads((out / "worker.json").read_text())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    runs.commit()
    hf.commit()  # compile caches, for the next container
    keep = [p for p in out.iterdir() if p.suffix == ".npz" or p.name in ("capture.log", "manifest.json")]
    return {"manifest": manifest, "files": {p.name: p.read_bytes() for p in keep}}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


@app.local_entrypoint()
def main(configs: str = ",".join(lib.SMOKE), out: str = ".cache/modal-runs") -> None:
    names = configs.split(",")
    if unknown := [n for n in names if n not in lib.SMOKE]:
        raise SystemExit(f"unknown configs {unknown}; known: {list(lib.SMOKE)}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("the tracked tree has changes; commit them so the manifest's commit is what ran")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / out / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "commit": git("rev-parse", "HEAD"),
        "command": "uv run --with modal==1.6.1 modal run experiments/modal/capture.py --configs " + configs,
        "modal": modal.__version__,
        "rates_usd_per_s": {"L4": lib.L4_PER_S, "cpu_core": lib.CPU_CORE_PER_S, "memory_gib": lib.MEMORY_GIB_PER_S},
        "configs": {},
    }

    def record() -> None:
        manifest["estimated_usd"] = round(sum(c.get("estimated_usd", 0) for c in manifest["configs"].values()), 4)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    models = sorted({(lib.SMOKE[n][0]["model"], lib.SMOKE[n][0]["revision"]) for n in names})
    start = time.time()
    fetched = download.remote(models)
    wall = time.time() - start
    cost = lib.estimate_cost(wall, gpu=False, cpu=lib.DOWNLOAD_CPU, memory_gib=lib.DOWNLOAD_MEMORY_GIB)
    manifest["download"] = {
        "models": models,
        **fetched,
        "wall_seconds": round(wall, 1),
        "estimated_usd": round(cost, 4),
    }
    spent = cost
    record()
    for name in names:
        if spent + lib.worst_case_config_cost() > lib.RUN_BUDGET_USD:
            manifest["stopped"] = (
                f"before {name}: ${spent:.2f} spent, a config can cost ${lib.worst_case_config_cost():.2f}"
            )
            break
        fields, n = lib.SMOKE[name]
        spec = {
            "config": fields,
            "prompts": lib.smoke_prompts(n),
            "save": n > 1,
            # The FP8 card lists its base model as a one-element list, which base_model_for doesn't read.
            "base_model": {"Qwen/Qwen3-4B-FP8": "Qwen/Qwen3-4B"}.get(fields["model"]),
        }
        print(f"{name}: {n} prompts on an L4", flush=True)
        start = time.time()
        try:
            result = capture_config.remote(name, spec, run_id)
        except modal.exception.FunctionTimeoutError:
            result = {"manifest": {"config": name, "error": f"timed out after {lib.CONFIG_TIMEOUT_S} s"}, "files": {}}
        wall = time.time() - start
        cost = lib.estimate_cost(wall, gpu=True, cpu=lib.GPU_CPU, memory_gib=lib.GPU_MEMORY_GIB)
        spent += cost
        (run_dir / name).mkdir()
        for filename, data in result["files"].items():
            (run_dir / name / filename).write_bytes(data)
        manifest["configs"][name] = {
            "wall_seconds": round(wall, 1),  # includes container start, so an upper bound on billed time
            "estimated_usd": round(cost, 4),
            **{k: result["manifest"][k] for k in ("returncode", "error") if k in result["manifest"]},
        }
        record()
        if result["manifest"].get("returncode") != 0:
            manifest["stopped"] = f"{name} failed; its log is in {run_dir / name}"
            break
    record()
    print(json.dumps({k: manifest[k] for k in ("configs", "estimated_usd")}, indent=2))
    print(f"Run {run_id} in {run_dir}. Check the actual charge on Modal's usage page.")
