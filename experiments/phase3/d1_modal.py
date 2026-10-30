"""D1's cost measurement on a Modal L4 (d1_l4.py), every arm-run in one container.

From the repository root, after `modal token new` (the token stays in .cache/, see
scripts/dev-env.sh):

    source scripts/dev-env.sh
    uv run --with modal==1.6.1 modal run experiments/phase3/d1_modal.py --spent-usd 0.0

Qwen3-4B in bf16 at 1cfa9a72, the probe layer 18, served with chunked prefill on, prefix caching
off and vLLM's defaults otherwise. The bench prompts are the 2,101 S-Labs calibration prompts
(.cache/probe/splits.parquet), sorted by id. Each arm-run: greedy replies to the first 32, 64
tokens each, a saturation bench of 300 prompts and a fixed-rate bench of up to 120 s, each
request 128 output tokens with EOS ignored. 4 arms over 3 rounds.

The image is capture.py's (uv.lock with the vllm extra, on Python 3.12.14) with vllm-lens 1.3.0
added, and the weights come from the hulder-hf Volume, so the Hub is offline in the container.

Stop rules: each arm-run stops after 10 minutes, the container after all 12, and the run doesn't
start if today's spend (--spent-usd, read from Modal's usage page) plus the container's worst
case would pass $3. Inside, the run stops at the first stop rule in d1_l4.py. The estimates use
Modal's published rates (experiments/modal/smoke_lib.py) and have run high against the actual
charge on the usage page.

Writes the results, every server and bench log, and a manifest with the estimated cost to
--out/<run id>/.
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
sys.path.insert(0, str(ROOT / "experiments" / "modal") if LOCAL else "/root")
import smoke_lib as lib

CACHE = "/vol/hf"
PHASE3 = "/root/experiments/phase3"  # the phase 3 files in the container
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
WORKLOAD = {"gen_tokens": 64, "saturation_prompts": 300, "fixed_seconds": 120, "output_len": 128}
N_GENERATE = 32
ROUNDS = 3
RUN_TIMEOUT_S = 10 * 60
N_RUNS = ROUNDS * 4


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
    cpu=lib.DOWNLOAD_CPU, memory=lib.DOWNLOAD_MEMORY_GIB * 1024, volumes={CACHE: hf}, timeout=30 * 60, retries=0
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
    gpu="L4",
    cpu=lib.GPU_CPU,
    memory=lib.GPU_MEMORY_GIB * 1024,
    volumes={CACHE: hf},
    timeout=N_RUNS * RUN_TIMEOUT_S + 10 * 60,
    retries=0,
    scaledown_window=2,
)
def measure(workload: dict) -> dict:
    """Every arm-run of d1_l4's schedule, then the results and every log."""
    os.environ.update({**CACHE_ENV, **ENGINE_ENV})
    sys.path.insert(0, PHASE3)
    import d1_l4
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
    freeze = subprocess.run(
        ["/.uv/uv", "pip", "freeze", "--python", sys.executable], capture_output=True, text=True, check=True
    )
    (out / "pip_freeze.txt").write_text(freeze.stdout)
    results = d1_l4.run_all(cfg, workload, out, rounds=ROUNDS, timeout_s=RUN_TIMEOUT_S)
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


def worst_case_usd() -> float:
    return lib.estimate_cost(
        N_RUNS * RUN_TIMEOUT_S + lib.CONTAINER_STARTUP_S, gpu=True, cpu=lib.GPU_CPU, memory_gib=lib.GPU_MEMORY_GIB
    )


@app.local_entrypoint()
def main(spent_usd: float, out: str = ".cache/modal-runs") -> None:
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("the tracked tree has changes; commit them so the manifest's commit is what ran")
    worst = worst_case_usd()
    if spent_usd + worst > lib.BUDGET_USD:
        raise SystemExit(f"today's ${spent_usd} plus the worst case ${worst:.2f} would pass ${lib.BUDGET_USD}")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / out / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "experiment": "d1_cost",
        "commit": git("rev-parse", "HEAD"),
        "command": f"uv run --with modal==1.6.1 modal run experiments/phase3/d1_modal.py --spent-usd {spent_usd}",
        "modal": modal.__version__,
        "rates_usd_per_s": {"L4": lib.L4_PER_S, "cpu_core": lib.CPU_CORE_PER_S, "memory_gib": lib.MEMORY_GIB_PER_S},
        "spent_before_usd": spent_usd,
        "worst_case_usd": round(worst, 4),
        "cfg": CFG,
        "workload": WORKLOAD,
        "rounds": ROUNDS,
        "run_timeout_s": RUN_TIMEOUT_S,
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
    returned = measure.remote(workload())
    wall = time.time() - start
    manifest["measure"] = {
        "wall_seconds": round(wall, 1),
        "estimated_usd": round(lib.estimate_cost(wall, gpu=True, cpu=lib.GPU_CPU, memory_gib=lib.GPU_MEMORY_GIB), 4),
        "nvidia_smi": returned["nvidia_smi"],
    }
    manifest["estimated_usd"] = round(manifest["download"]["estimated_usd"] + manifest["measure"]["estimated_usd"], 4)
    for name, data in returned["files"].items():
        (run_dir / name).parent.mkdir(parents=True, exist_ok=True)
        (run_dir / name).write_bytes(data)
    (run_dir / "d1_l4.json").write_text(json.dumps(returned["results"], indent=2, default=str) + "\n")
    record()
    print(json.dumps({k: returned["results"][k] for k in ("stopped", "verdicts")}, indent=2))
    print(
        f"Run {run_id} in {run_dir}, estimated ${manifest['estimated_usd']}. Check the actual charge on Modal's usage page."
    )
