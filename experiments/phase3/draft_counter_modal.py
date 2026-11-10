"""draft_counter.py's check on one Modal L4 container.

From the repository root, after `modal token new` (the token stays in .cache/, see
scripts/dev-env.sh):

    source scripts/dev-env.sh
    uv run --with modal==1.6.1 modal run experiments/phase3/draft_counter_modal.py --spent-usd 0.0

The image is d1_modal.py's (uv.lock with the vllm extra, on Python 3.12.14) without vllm-lens, and
hulder is a wheel built from this checkout, so the server loads hulder.runtime as installed. The
weights come from the hulder-hf Volume at d1_local's revision, with the Hub offline.

Stop rules: the call has 15 minutes once its container runs, and each server 6 minutes to answer
/health (the 1.5B overhead run's slowest cold start was 316 s). Waiting for a GPU, container
startup and the call together get WALL_LIMIT_S, after which the call is cancelled with its
container. Nothing retries. The run doesn't start if today's spend (--spent-usd, read from
Modal's usage page) plus the worst case would pass $3. The worst case and the estimate both use
Modal's published rates (experiments/modal/smoke_lib.py) over wall time, queue included, so the
estimate can't pass the guarded worst case and runs high. The manifest also records the wait
before the function ran and an estimate from the container's own time, closer to the charge.

Writes draft_counter.json, both server logs, a manifest with the estimated cost and SHA256SUMS
to --out/<run id>/.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

LOCAL = modal.is_local()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if LOCAL else Path("/root")
PHASE3 = "/root/experiments/phase3"
sys.path.insert(0, str(ROOT / "experiments" / "modal") if LOCAL else "/root")
sys.path.insert(0, str(HERE) if LOCAL else PHASE3)
import d1_local
import draft_counter
import smoke_lib as lib

CACHE = "/vol/hf"
CACHE_ENV = {
    "HF_HOME": f"{CACHE}/hf",
    "VLLM_CACHE_ROOT": f"{CACHE}/vllm",
    "TRITON_CACHE_DIR": f"{CACHE}/triton",
    "XDG_CACHE_HOME": f"{CACHE}/xdg",
    "FLASHINFER_WORKSPACE_BASE": CACHE,
}
ENGINE_ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HUB_OFFLINE": "1"}
UV_VERSION = "0.12.21"
CALL_TIMEOUT_S = 15 * 60
STARTUP_TIMEOUT_S = 6 * 60
# Run 20261110T152540Z waited about 20 minutes for a GPU, so its 1,652 s of wall time passed the
# worst case its guard had counted. The guard now counts this much queue, and the call is cancelled
# when the whole wall time would pass it.
QUEUE_LIMIT_S = 30 * 60
WALL_LIMIT_S = QUEUE_LIMIT_S + lib.CONTAINER_STARTUP_S + CALL_TIMEOUT_S


def build_wheel() -> Path:
    """hulder as a wheel from this checkout, as d1_modal.py builds it."""
    dist = ROOT / ".cache" / "modal" / "dist"
    shutil.rmtree(dist, ignore_errors=True)
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(dist)], cwd=ROOT, check=True, capture_output=True)
    [wheel] = dist.glob("hulder-*.whl")
    return wheel


WHEEL = build_wheel() if LOCAL else Path("/root/hulder.whl")

image = (
    modal.Image.debian_slim(python_version="3.12.14")
    .apt_install("gcc")
    .uv_sync(
        str(ROOT),
        extras=["vllm"],
        extra_options="--no-default-groups --python-preference=system",
        uv_version=UV_VERSION,
    )
    .add_local_file(str(WHEEL), f"/wheels/{WHEEL.name}", copy=True)
    .run_commands(f"/.uv/uv pip install --python /.uv/.venv/bin/python --no-deps /wheels/{WHEEL.name}")
    .add_local_file(str(ROOT / "experiments" / "modal" / "smoke_lib.py"), "/root/smoke_lib.py")
    # At their checkout paths, since d1_local.py finds the repository root two levels up.
    .add_local_file(str(HERE / "d1_local.py"), f"{PHASE3}/d1_local.py")
    .add_local_file(str(HERE / "draft_counter.py"), f"{PHASE3}/draft_counter.py")
    .add_local_file(str(d1_local.PROMPTS), "/root/examples/live_demo/demo_prompts.jsonl")
)
app = modal.App("hulder-draft-counter", image=image)
hf = modal.Volume.from_name("hulder-hf", create_if_missing=True)


def nvidia_smi() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


@app.function(
    gpu="L4",
    cpu=lib.GPU_CPU,
    memory=lib.GPU_MEMORY_GIB * 1024,
    volumes={CACHE: hf},
    timeout=CALL_TIMEOUT_S,
    retries=0,
    scaledown_window=2,
)
def check() -> dict:
    started = time.time()
    os.environ.update({**CACHE_ENV, **ENGINE_ENV})
    from importlib.resources import as_file, files

    from huggingface_hub import snapshot_download

    gpu = nvidia_smi()
    # Fails at once if the weights aren't on the Volume, before any server starts.
    snapshot_download(d1_local.MODEL, revision=d1_local.REVISION, local_files_only=True)
    out = Path("/tmp/draft-counter")
    with as_file(files("hulder") / "data" / "probe_demo.npz") as probe:
        results = draft_counter.run(probe, out, STARTUP_TIMEOUT_S)
    logs = {path.name: path.read_bytes() for path in sorted(out.glob("*.log"))}
    return {"results": results, "logs": logs, "nvidia_smi": gpu, "container_started": started}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


@app.local_entrypoint()
def main(spent_usd: float, out: str = ".cache/modal-runs") -> None:
    worst = lib.estimate_cost(WALL_LIMIT_S, gpu=True, cpu=lib.GPU_CPU, memory_gib=lib.GPU_MEMORY_GIB)
    if spent_usd + worst > lib.BUDGET_USD:
        raise SystemExit(f"today's ${spent_usd} plus the worst case ${worst:.2f} would pass ${lib.BUDGET_USD}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("commit first: the manifest records the commit the image was built from")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / out / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "experiment": "draft_counter",
        "commit": git("rev-parse", "HEAD"),
        "command": f"uv run --with modal==1.6.1 modal run experiments/phase3/draft_counter_modal.py --spent-usd {spent_usd}",
        "modal": modal.__version__,
        "rates_usd_per_s": {"L4": lib.L4_PER_S, "cpu_core": lib.CPU_CORE_PER_S, "memory_gib": lib.MEMORY_GIB_PER_S},
        "spent_before_usd": spent_usd,
        "worst_case_usd": round(worst, 4),
        "timeouts_s": {"call": CALL_TIMEOUT_S, "startup": STARTUP_TIMEOUT_S, "wall": WALL_LIMIT_S},
        "model": d1_local.MODEL,
        "revision": d1_local.REVISION,
        "workload": {
            "requests": draft_counter.N_REQUESTS,
            "concurrency": draft_counter.CONCURRENCY,
            "max_tokens": draft_counter.MAX_TOKENS,
        },
    }

    def record() -> None:
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    record()
    start = time.time()
    call = check.spawn()
    try:
        returned = call.get(timeout=WALL_LIMIT_S)
    except (TimeoutError, modal.exception.TimeoutError):
        call.cancel(terminate_containers=True)
        manifest["cancelled"] = f"no result within {WALL_LIMIT_S} s of wall time, queue included"
        raise
    finally:
        wall = time.time() - start
        manifest["wall_seconds"] = round(wall, 1)
        manifest["estimated_usd"] = round(
            lib.estimate_cost(wall, gpu=True, cpu=lib.GPU_CPU, memory_gib=lib.GPU_MEMORY_GIB), 4
        )
        record()
    for name, data in returned["logs"].items():
        (run_dir / name).write_bytes(data)
    (run_dir / "draft_counter.json").write_text(json.dumps(returned["results"], indent=2) + "\n")
    manifest["nvidia_smi"] = returned["nvidia_smi"]
    # The wait for a GPU plus the container's boot, up to the function's first line by the
    # container's clock, so a few seconds of skew either way. Modal bills the boot but not the
    # queue, so the closer estimate adds back the most boot time measured.
    waited = min(max(returned["container_started"] - start, 0.0), wall)
    manifest["waited_before_function_seconds"] = round(waited, 1)
    manifest["estimated_container_usd"] = round(
        lib.estimate_cost(
            min(wall - waited + lib.CONTAINER_STARTUP_S, wall),
            gpu=True,
            cpu=lib.GPU_CPU,
            memory_gib=lib.GPU_MEMORY_GIB,
        ),
        4,
    )
    record()
    sums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in sorted(run_dir.iterdir())
        if path.name != "SHA256SUMS"
    ]
    (run_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps(returned["results"]["verdict"]))
    print(
        f"Run {run_id} in {run_dir}, estimated ${manifest['estimated_usd']}. Check the actual charge on Modal's usage page."
    )
