"""D1's cost measurement on a Modal L4 (d1_l4.py), every arm-run in one container.

From the repository root, after `modal token new` (the token stays in .cache/, see
scripts/dev-env.sh):

    source scripts/dev-env.sh
    uv run --with modal==1.6.1 modal run experiments/phase3/d1_modal.py --spent-usd 0.0

Qwen3-4B in bf16 at 1cfa9a72, the probe layer 18, served with chunked prefill on, prefix caching
off and vLLM's defaults otherwise. The bench prompts are the 2,101 S-Labs calibration prompts
(.cache/probe/splits.parquet), sorted by id. Each arm-run: greedy replies to the first 16, 64
tokens each, a saturation bench of 300 prompts and a fixed-rate bench of up to 90 s, each
request 128 output tokens with EOS ignored. 4 arms over 3 rounds.

The image is capture.py's (uv.lock with the vllm extra, on Python 3.12.14) with vllm-lens 1.3.0
added, and the weights come from the hulder-hf Volume, so the Hub is offline in the container.

Stop rules: each server has 8 minutes to answer /health in round 0, which may compile cold, and
4.5 minutes in later rounds, which reuse its compile cache. Measurement then has 4 minutes, and
only lens may run out of it without stopping the run, since that is its result. The container
stops when all 12 arm-runs could have run out both timeouts, and the run doesn't start if today's
spend (--spent-usd, read from Modal's usage page) plus the worst case would pass $3. Inside, the
run stops at the first stop rule in d1_l4.py. The estimates use Modal's published rates
(experiments/modal/smoke_lib.py) and have run high against the actual charge on the usage page.

Writes the results, every server and bench log, the image's packages and a manifest with the
estimated cost to --out/<run id>/. A failed call still records its wall time and estimate. The
estimate counts the call's whole wall time, which includes any wait for a GPU, so it runs high.
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

LOCAL = modal.is_local()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if LOCAL else Path("/")
PHASE3 = "/root/experiments/phase3"  # the phase 3 files in the container
sys.path.insert(0, str(ROOT / "experiments" / "modal") if LOCAL else "/root")
sys.path.insert(0, str(HERE) if LOCAL else PHASE3)
import d1_l4
import smoke_lib as lib

CACHE = "/vol/hf"
CACHE_ENV = {
    "HF_HOME": f"{CACHE}/hf",
    "VLLM_CACHE_ROOT": f"{CACHE}/vllm",
    "TRITON_CACHE_DIR": f"{CACHE}/triton",
    "XDG_CACHE_HOME": f"{CACHE}/xdg",
    "FLASHINFER_WORKSPACE_BASE": CACHE,
}
# As capture.py: FlashInfer's sampler JIT-compiles with nvcc, absent from this image.
ENGINE_ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HUB_OFFLINE": "1"}
UV_VERSION = "0.12.21"
VLLM_LENS = "vllm-lens==1.3.0"
MODEL, REVISION = "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"
CFG = {
    "model": MODEL,
    "hidden_size": 2560,
    "layer": 18,
    "serve": [
        "--revision", REVISION, "--dtype", "bfloat16", "--max-model-len", "2048",
        "--enable-chunked-prefill", "--no-enable-prefix-caching", "--seed", "0",
    ],
}  # fmt: skip
WORKLOAD = {"gen_tokens": 64, "saturation_prompts": 300, "fixed_seconds": 90, "output_len": 128}
N_GENERATE = 16
ROUNDS = 3
# From run 20261030T191447Z's baseline: a cold start of 386.6 s, 174 s of it compiling, and about
# 50 s of greedy replies, 16 s of saturation and 90 s at the fixed rate at this workload.
TIMEOUTS = d1_l4.Timeouts(cold_startup=8 * 60, warm_startup=270, measure=4 * 60)
# Outside the arm-runs: the GPU query, the comparisons and committing the compile cache.
MEASURE_SLACK_S = 5 * 60
MEASURE_TIMEOUT_S = d1_l4.worst_case_seconds(TIMEOUTS, ROUNDS) + MEASURE_SLACK_S
# The weights are on the Volume already, so the download only checks them; 8 GB fetched fresh fits too.
DOWNLOAD_TIMEOUT_S = 10 * 60
# The pre-flight: CPU only, on the same image and Volume, before any GPU is asked for.
PREFLIGHT_CPU, PREFLIGHT_MEMORY_GIB, PREFLIGHT_TIMEOUT_S = 2.0, 8, 10 * 60


def build_wheel() -> Path:
    """hulder as a wheel from this checkout, as capture.py builds it."""
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
    .run_commands(
        f"/.uv/uv pip install --python /.uv/.venv/bin/python --no-deps /wheels/{WHEEL.name}",
        f"/.uv/uv pip install --python /.uv/.venv/bin/python {VLLM_LENS}",
    )
    .add_local_file(str(ROOT / "experiments" / "modal" / "smoke_lib.py"), "/root/smoke_lib.py")
    # At their checkout paths, since d1_local.py finds the repository root two levels up.
    .add_local_file(str(HERE / "d1_local.py"), f"{PHASE3}/d1_local.py")
    .add_local_file(str(HERE / "d1_l4.py"), f"{PHASE3}/d1_l4.py")
    .add_local_file(str(HERE / "no_write_connector.py"), f"{PHASE3}/no_write_connector.py")
)
app = modal.App("hulder-d1", image=image)
hf = modal.Volume.from_name("hulder-hf", create_if_missing=True)


@app.function(
    cpu=lib.DOWNLOAD_CPU,
    memory=lib.DOWNLOAD_MEMORY_GIB * 1024,
    volumes={CACHE: hf},
    timeout=DOWNLOAD_TIMEOUT_S,
    retries=0,
)
def download() -> dict:
    """The model at its pinned revision in the Volume's HF cache, fetched if it isn't there yet."""
    os.environ.update(CACHE_ENV)
    from huggingface_hub import snapshot_download

    start = time.time()
    snapshot_download(MODEL, revision=REVISION)
    hf.commit()
    return {"seconds": round(time.time() - start, 1)}


@app.function(
    cpu=PREFLIGHT_CPU,
    memory=PREFLIGHT_MEMORY_GIB * 1024,
    volumes={CACHE: hf},
    timeout=PREFLIGHT_TIMEOUT_S,
    retries=0,
)
def preflight() -> dict:
    """d1_l4.preflight on this image with the Volume's HF cache and the Hub offline, as measure
    runs, and the image's package list."""
    os.environ.update({**CACHE_ENV, **ENGINE_ENV})
    return {"facts": d1_l4.preflight(MODEL, REVISION), "packages": d1_l4.installed_packages()}


@app.function(
    gpu="L4",
    cpu=lib.GPU_CPU,
    memory=lib.GPU_MEMORY_GIB * 1024,
    volumes={CACHE: hf},
    timeout=int(MEASURE_TIMEOUT_S),
    retries=0,
    scaledown_window=2,
)
def measure(workload: dict) -> dict:
    """Every arm-run of d1_l4's schedule, then the results and every log."""
    os.environ.update({**CACHE_ENV, **ENGINE_ENV})
    from huggingface_hub import snapshot_download

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    cfg = {**CFG, "tokenizer": snapshot_download(MODEL, revision=REVISION, local_files_only=True)}
    out = Path("/tmp/d1")
    out.mkdir(parents=True)
    (out / "packages.txt").write_text(d1_l4.installed_packages())
    results = d1_l4.run_all(cfg, workload, out, rounds=ROUNDS, timeouts=TIMEOUTS)
    hf.commit()  # compile caches, for later runs
    files = {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    return {"results": results, "nvidia_smi": gpu, "files": files}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def workload() -> dict:
    import pandas as pd

    rows = pd.read_parquet(lib.SLABS_SPLITS, columns=["id", "split", "text"])
    calibration = rows[rows["split"] == "calibration"].sort_values("id", kind="stable")
    texts = [str(t) for t in calibration["text"]]
    return {**WORKLOAD, "gen_prompts": texts[:N_GENERATE], "bench_prompts": texts}


def download_worst_case_usd() -> float:
    return lib.estimate_cost(
        DOWNLOAD_TIMEOUT_S + lib.CONTAINER_STARTUP_S,
        gpu=False,
        cpu=lib.DOWNLOAD_CPU,
        memory_gib=lib.DOWNLOAD_MEMORY_GIB,
    )


def preflight_worst_case_usd() -> float:
    return lib.estimate_cost(
        PREFLIGHT_TIMEOUT_S + lib.CONTAINER_STARTUP_S, gpu=False, cpu=PREFLIGHT_CPU, memory_gib=PREFLIGHT_MEMORY_GIB
    )


def worst_case_usd() -> float:
    return lib.estimate_cost(
        MEASURE_TIMEOUT_S + lib.CONTAINER_STARTUP_S, gpu=True, cpu=lib.GPU_CPU, memory_gib=lib.GPU_MEMORY_GIB
    )


@app.local_entrypoint()
def main(spent_usd: float, out: str = ".cache/modal-runs", preflight_only: bool = False) -> None:
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("the tracked tree has changes; commit them so the manifest's commit is what ran")
    worst = download_worst_case_usd() + preflight_worst_case_usd() + (0 if preflight_only else worst_case_usd())
    if spent_usd + worst > lib.BUDGET_USD:
        raise SystemExit(f"today's ${spent_usd} plus the worst case ${worst:.2f} would pass ${lib.BUDGET_USD}")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / out / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "experiment": "d1_cost_preflight" if preflight_only else "d1_cost",
        "commit": git("rev-parse", "HEAD"),
        "command": "uv run --with modal==1.6.1 modal run experiments/phase3/d1_modal.py "
        f"--spent-usd {spent_usd}" + (" --preflight-only" if preflight_only else ""),
        "modal": modal.__version__,
        "rates_usd_per_s": {"L4": lib.L4_PER_S, "cpu_core": lib.CPU_CORE_PER_S, "memory_gib": lib.MEMORY_GIB_PER_S},
        "spent_before_usd": spent_usd,
        "worst_case_usd": round(worst, 4),
        "cfg": CFG,
        "workload": WORKLOAD,
        "rounds": ROUNDS,
        "timeouts_s": {**TIMEOUTS._asdict(), "stop_grace": d1_l4.STOP_GRACE_S, "measure_call": MEASURE_TIMEOUT_S},
    }

    def record() -> None:
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    record()
    start = time.time()
    manifest["download"] = download.remote()
    wall = time.time() - start
    manifest["download"]["estimated_usd"] = round(
        lib.estimate_cost(wall, gpu=False, cpu=lib.DOWNLOAD_CPU, memory_gib=lib.DOWNLOAD_MEMORY_GIB), 4
    )
    record()
    start = time.time()
    checked = preflight.remote()
    wall = time.time() - start
    manifest["preflight"] = {
        **checked["facts"],
        "wall_seconds": round(wall, 1),
        "estimated_usd": round(
            lib.estimate_cost(wall, gpu=False, cpu=PREFLIGHT_CPU, memory_gib=PREFLIGHT_MEMORY_GIB), 4
        ),
    }
    manifest["estimated_usd"] = round(manifest["download"]["estimated_usd"] + manifest["preflight"]["estimated_usd"], 4)
    (run_dir / "packages.txt").write_text(checked["packages"])
    record()
    if preflight_only:
        print(json.dumps(manifest["preflight"], indent=2))
        print(f"Pre-flight {run_id} in {run_dir}, estimated ${manifest['estimated_usd']}.")
        return
    start = time.time()
    try:
        returned = measure.remote(workload())
    except Exception as error:
        # Recorded before re-raising, so a failed call still leaves its wall time and estimate.
        wall = time.time() - start
        manifest["measure"] = {
            "wall_seconds": round(wall, 1),
            "estimated_usd": round(
                lib.estimate_cost(wall, gpu=True, cpu=lib.GPU_CPU, memory_gib=lib.GPU_MEMORY_GIB), 4
            ),
            "error": f"{type(error).__name__}: {error}",
        }
        manifest["estimated_usd"] = round(
            manifest["download"]["estimated_usd"]
            + manifest["preflight"]["estimated_usd"]
            + manifest["measure"]["estimated_usd"],
            4,
        )
        record()
        raise
    wall = time.time() - start
    manifest["measure"] = {
        "wall_seconds": round(wall, 1),
        "estimated_usd": round(lib.estimate_cost(wall, gpu=True, cpu=lib.GPU_CPU, memory_gib=lib.GPU_MEMORY_GIB), 4),
        "nvidia_smi": returned["nvidia_smi"],
    }
    manifest["estimated_usd"] = round(
        manifest["download"]["estimated_usd"]
        + manifest["preflight"]["estimated_usd"]
        + manifest["measure"]["estimated_usd"],
        4,
    )
    for name, data in returned["files"].items():
        (run_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / name).write_bytes(data)
    (run_dir / "d1_l4.json").write_text(json.dumps(returned["results"], indent=2, default=str) + "\n")
    record()
    print(json.dumps({k: returned["results"][k] for k in ("stopped", "verdicts", "divergences")}, indent=2))
    print(
        f"Run {run_id} in {run_dir}, estimated ${manifest['estimated_usd']}. Check the actual charge on Modal's usage page."
    )
