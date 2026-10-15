"""Activation captures on a Modal L4 (Ada, 24 GB), one container per comparison.

From the repository root, after `modal token new` (the token stays in .cache/, see
scripts/dev-env.sh):

    source scripts/dev-env.sh
    uv run --with modal==1.6.1 modal run experiments/modal/capture.py --experiment fp8_online --spent-usd 0.6
    uv run python experiments/modal/compare.py .cache/modal-runs/<run id>

The image is this repository's uv.lock with the vllm extra (vLLM 0.30.0) on Python 3.12.14.
Model weights download once, on a CPU container, into the hulder-hf Volume, which also keeps
vLLM's compile caches. An experiment (smoke_lib.EXPERIMENTS) is one or more containers. Each
container is one Modal call that runs its configs in order, each in a fresh engine process with
the Hub offline, so a reference, its rerun and a candidate share one GPU and driver. A single_use
experiment instead runs its containers at once, each serving one call, to put runs on different GPUs.

Each snapshot records the GPU's UUID and driver, so runs from different GPUs warn when checked.

Stop rules: each config stops after 20 minutes, and no container starts if the estimate of what
was spent today (--spent-usd, plus this run so far) and that container's worst case would pass
$3. Estimates use Modal's published rates (smoke_lib.py) and have run high against the actual
charge on Modal's usage page.

Writes, per run, the snapshots (one per layer), the full log and a manifest to
--out/<run id>/<label>/, and the run's manifest to --out/<run id>/manifest.json. The tracked tree
must be clean, so the manifest's commit is what ran.
"""

import json
import os
import shutil
import signal
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


def run_group(group: list[tuple[str, dict]], run_id: str) -> dict:
    """Run worker.py on each (label, spec) in order, each in its own process with the Hub offline,
    and return their manifests and files."""
    query = "--query-gpu=name,uuid,driver_version,memory.total"
    gpu = subprocess.run(
        ["nvidia-smi", query, "--format=csv,noheader"], capture_output=True, text=True, check=False
    ).stdout.strip()
    results = {}
    for label, spec in group:
        start = time.time()
        out = Path(RUNS) / run_id / label
        out.mkdir(parents=True, exist_ok=True)
        spec = {**spec, "extra_metadata": lib.gpu_metadata(gpu)}
        (out / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
        env = dict(spec.get("env", {}))
        log_cublas = env.pop(lib.CUBLAS_LOG, None)
        if log_cublas:
            env.update(lib.cublas_log_env(str(out)))
        timeout_s = spec.get("timeout_s", lib.CONFIG_TIMEOUT_S)
        error = None
        with open(out / "capture.log", "w") as log:
            # Its own session, so a timeout kills the engine-core process too and frees the GPU.
            worker = subprocess.Popen(
                [sys.executable, "/root/worker.py", str(out / "spec.json"), str(out)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, **CACHE_ENV, **ENGINE_ENV, **env, "HF_HUB_OFFLINE": "1"},
                start_new_session=True,
            )
            try:
                returncode = worker.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                returncode, error = worker.wait(), f"timed out after {timeout_s} s"
        text = (out / "capture.log").read_text(errors="replace")
        if log_cublas:
            raw = "".join(p.read_text(errors="replace") for p in sorted(out.glob("cublaslt*.log")))
            (out / "cublas_fingerprint.txt").write_text("\n".join(lib.cublas_fingerprint(raw)) + "\n")
            for p in out.glob("cublaslt*.log"):
                p.unlink()
        manifest = {
            "label": label,
            "returncode": returncode,
            **({"error": error} if error else {}),
            "nvidia_smi": gpu,
            "engine_env": {**ENGINE_ENV, **env},
            "seconds": round(time.time() - start, 1),
            "log": lib.log_facts(text),
        }
        if (out / "worker.json").exists():
            manifest["worker"] = json.loads((out / "worker.json").read_text())
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        runs.commit()
        hf.commit()  # compile caches, for the next engine
        names = ("capture.log", "manifest.json", "cublas_fingerprint.txt")
        keep = [p for p in out.iterdir() if p.suffix == ".npz" or p.name in names]
        results[label] = {"manifest": manifest, "files": {p.name: p.read_bytes() for p in keep}}
        if returncode != 0:
            break
    return results


GPU_FUNCTION = {
    "gpu": "L4",
    "cpu": lib.GPU_CPU,
    "memory": lib.GPU_MEMORY_GIB * 1024,
    "volumes": {CACHE: hf, RUNS: runs},
    "timeout": lib.MAX_CONFIGS_PER_CONTAINER * lib.CONFIG_TIMEOUT_S,
    "retries": 0,
    "scaledown_window": 2,
}


@app.function(**GPU_FUNCTION)
def capture_group(group: list[tuple[str, dict]], run_id: str) -> dict:
    return run_group(group, run_id)


@app.function(**GPU_FUNCTION, single_use_containers=True)
def capture_group_single_use(group: list[tuple[str, dict]], run_id: str) -> dict:
    return run_group(group, run_id)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def spec_for(entry: tuple, timeout_s: int) -> dict:
    """The worker spec for one (label, config[, env]) entry of an experiment."""
    fields, n = lib.CONFIGS[entry[1]]
    env = entry[2] if len(entry) > 2 else {}
    return {
        "config": fields,
        "prompts": lib.smoke_prompts(n),
        "save": n > 1,
        "config_name": entry[1],
        "base_model": lib.BASE_MODEL_OVERRIDES.get(entry[1]),
        "env": env,
        "timeout_s": timeout_s,
    }


@app.local_entrypoint()
def main(experiment: str, spent_usd: float = 0.0, out: str = ".cache/modal-runs") -> None:
    if experiment not in lib.EXPERIMENTS:
        raise SystemExit(f"unknown experiment {experiment!r}; known: {list(lib.EXPERIMENTS)}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("the tracked tree has changes; commit them so the manifest's commit is what ran")
    containers, single_use, timeout_s = lib.EXPERIMENTS[experiment]
    if max(map(len, containers)) > lib.MAX_CONFIGS_PER_CONTAINER:
        raise SystemExit(f"a container runs at most {lib.MAX_CONFIGS_PER_CONTAINER} configs")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / out / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "experiment": experiment,
        "single_use": single_use,
        "commit": git("rev-parse", "HEAD"),
        "command": f"uv run --with modal==1.6.1 modal run experiments/modal/capture.py --experiment {experiment}",
        "modal": modal.__version__,
        "rates_usd_per_s": {"L4": lib.L4_PER_S, "cpu_core": lib.CPU_CORE_PER_S, "memory_gib": lib.MEMORY_GIB_PER_S},
        "spent_before_usd": spent_usd,
        "containers": [],
        "configs": {},
    }

    def record() -> None:
        run_cost = manifest.get("download", {}).get("estimated_usd", 0)
        run_cost += sum(c["estimated_usd"] for c in manifest["containers"])
        manifest["estimated_usd"] = round(run_cost, 4)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def affordable(n_configs: int) -> bool:
        worst = lib.worst_case_cost(n_configs, timeout_s)
        return spent_usd + manifest.get("estimated_usd", 0) + worst <= lib.BUDGET_USD

    start = time.time()
    fetched = download.remote(lib.downloads(experiment))
    wall = time.time() - start
    cost = lib.estimate_cost(wall, gpu=False, cpu=lib.DOWNLOAD_CPU, memory_gib=lib.DOWNLOAD_MEMORY_GIB)
    manifest["download"] = {**fetched, "wall_seconds": round(wall, 1), "estimated_usd": round(cost, 4)}
    record()

    def collect(i: int, group: list, results: dict, wall: float) -> bool:
        cost = lib.estimate_cost(wall, gpu=True, cpu=lib.GPU_CPU, memory_gib=lib.GPU_MEMORY_GIB)
        manifest["containers"].append(
            {"labels": [e[0] for e in group], "wall_seconds": round(wall, 1), "estimated_usd": round(cost, 4)}
        )
        ok = len(results) == len(group)
        for label, config, *_ in group:
            if label not in results:
                continue
            (run_dir / label).mkdir()
            for filename, data in results[label]["files"].items():
                (run_dir / label / filename).write_bytes(data)
            m = results[label]["manifest"]
            manifest["configs"][label] = {"container": i, "config": config, "returncode": m["returncode"]}
            ok = ok and m["returncode"] == 0
        record()
        return ok

    groups = [[(e[0], spec_for(e, timeout_s)) for e in group] for group in containers]
    if single_use:
        # All at once, so each container holds its own GPU; the budget covers all of them up front.
        if not affordable(sum(map(len, containers))):
            manifest["stopped"] = "the containers' worst case would pass the day's budget"
        else:
            start = time.time()
            calls = [capture_group_single_use.spawn(group, run_id) for group in groups]
            for i, (group, call) in enumerate(zip(containers, calls, strict=True)):
                results = call.get()
                if not collect(i, group, results, time.time() - start):
                    manifest["stopped"] = f"container {i} failed; its logs are in {run_dir}"
    else:
        for i, (group, specs) in enumerate(zip(containers, groups, strict=True)):
            if not affordable(len(group)):
                manifest["stopped"] = f"before container {i}: its worst case would pass the day's budget"
                break
            print(f"container {i}: {', '.join(e[0] for e in group)} on an L4", flush=True)
            start = time.time()
            results = capture_group.remote(specs, run_id)
            if not collect(i, group, results, time.time() - start):
                manifest["stopped"] = f"container {i} failed; its logs are in {run_dir}"
                break
    record()
    print(json.dumps({k: manifest.get(k) for k in ("configs", "estimated_usd", "stopped")}, indent=2))
    print(f"Run {run_id} in {run_dir}. Check the actual charge on Modal's usage page.")
