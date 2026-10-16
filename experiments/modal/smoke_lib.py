"""The parts of the Modal capture runner that need neither Modal nor a GPU: prompt selection,
the configs and experiments, log facts, cost estimates and activation comparison. Tested in CI."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# .parent rather than .parents[2], which fails inside a container where this file is /root/smoke_lib.py.
ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS = ROOT / "examples" / "live_demo" / "demo_prompts.jsonl"

# Modal's published rates in USD per second, from https://modal.com/pricing (checked 2026-10-13).
# Billed per second of container time, CPU and memory at least at the requested amount.
L4_PER_S = 0.000222
CPU_CORE_PER_S = 0.0000131  # one physical core is 2 vCPUs
MEMORY_GIB_PER_S = 0.00000222

# What each capture container requests, and what the download container requests.
GPU_CPU, GPU_MEMORY_GIB = 4.0, 32
DOWNLOAD_CPU, DOWNLOAD_MEMORY_GIB = 2.0, 4

# Stop rules: no config beyond 20 minutes, and no more than $3 (estimated) spent in a day, counting
# what was spent before the run (--spent-usd).
CONFIG_TIMEOUT_S = 20 * 60
BUDGET_USD = 3.0
MAX_CONFIGS_PER_CONTAINER = 6

SYSTEM_PROMPT = "You are a helpful assistant."
# Qwen3 has 36 decoder layers at both sizes. Two depths, half and three quarters.
LAYERS = (18, 27)
QWEN3_4B = {
    "model": "Qwen/Qwen3-4B",
    "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    "dtype": "bfloat16",
}
QWEN3_4B_FP8 = {
    "model": "Qwen/Qwen3-4B-FP8",
    "revision": "96b30dc13593a244a5e59e84687309f53c375cfa",
    "dtype": "bfloat16",
    "quantization": "fp8",
}
QWEN3_8B = {
    "model": "Qwen/Qwen3-8B",
    "revision": "b968826d9c46dd6066d109eabc6255188de91218",
    "dtype": "bfloat16",
}
COMMON = {
    "layers": LAYERS,
    "positions": (-1,),
    "system_prompt": SYSTEM_PROMPT,
    "enable_thinking": False,  # pinned: Qwen3's template, and so the last token, depends on it
    "batch_size": 8,
    "gpu_memory_utilization": 0.9,
    "max_model_len": 1024,
    "max_num_batched_tokens": 1024,
}
# name -> (CaptureConfig fields, prompts captured). base_model_for finds every base model itself:
# Qwen3-4B-FP8's card names it, and online FP8 of Qwen3-4B is its own base. The 8B run captures one
# prompt, only to read the KV-cache headroom bf16 leaves on 24 GB; its activations are dropped.
CONFIGS = {
    "qwen3_4b_bf16": ({**COMMON, **QWEN3_4B}, 64),
    "qwen3_4b_fp8": ({**COMMON, **QWEN3_4B_FP8}, 64),
    # vLLM's online FP8: bf16 weights quantized at load, per-tensor scales.
    "qwen3_4b_fp8_online": ({**COMMON, **QWEN3_4B, "quantization": "fp8"}, 64),
    "qwen3_8b_bf16_load": ({**COMMON, **QWEN3_8B}, 1),
}
# config name -> base model, passed to base_model_for as override=. Only for a checkpoint whose
# base model it can't find; the worker stops with a warning naming the entry to add.
BASE_MODEL_OVERRIDES: dict[str, str] = {}


def base_model_error(config: str, warning: str) -> str:
    """Why the worker stops on a base_model warning: a snapshot without a base model can't be
    checked against its reference, so a run here stays strict where the examples carry on."""
    return (
        f"{warning}\nThis runner stops rather than record no base model. To resolve it, add "
        f'"{config}": "<base model>" to BASE_MODEL_OVERRIDES in experiments/modal/smoke_lib.py.'
    )


# Per-run environment for the batch-invariance experiment. Batch-invariant mode on sm_8x swaps
# mm/addmm/matmul/linear for Triton persistent kernels, sets FlashAttention to one split and turns
# off AOT compile loading (vLLM 0.30.0, model_executor/determinism/batch_invariant.py).
BATCH_INVARIANT = {"VLLM_BATCH_INVARIANT": "1"}
# cuBLASLt tracing, so a GEMM algorithm switch between runs shows in a diff (cublas_fingerprint).
CUBLAS_LOG = "cublas"

# name -> (containers, single_use, per-config timeout in s). Each container is one Modal call that
# runs its (label, config[, env]) entries in order, each in a fresh engine process, so a comparison's
# runs share a GPU and driver. single_use experiments run their containers at once in containers
# that serve one call each, so each lands on its own GPU.
EXPERIMENTS = {
    # As the 2026-10-13 smoke run actually ran: four calls that Modal served from one warm container.
    "smoke": (
        [
            [
                ("qwen3_4b_bf16", "qwen3_4b_bf16"),
                ("qwen3_4b_bf16_rerun", "qwen3_4b_bf16"),
                ("qwen3_4b_fp8", "qwen3_4b_fp8"),
                ("qwen3_8b_bf16_load", "qwen3_8b_bf16_load"),
            ]
        ],
        False,
        CONFIG_TIMEOUT_S,
    ),
    "fp8_online": (
        [[("qwen3_4b_bf16", "qwen3_4b_bf16"), ("qwen3_4b_fp8_online", "qwen3_4b_fp8_online")]],
        False,
        CONFIG_TIMEOUT_S,
    ),
    "cross_container": ([[("bf16_a", "qwen3_4b_bf16")], [("bf16_b", "qwen3_4b_bf16")]], True, CONFIG_TIMEOUT_S),
    # Three default engines with cuBLAS logging, then three batch-invariant ones, in one container.
    # 10 minutes per config keeps six within the day's budget.
    "batch_invariant": (
        [
            [
                *((f"default_{i}", "qwen3_4b_bf16", {CUBLAS_LOG: "1"}) for i in (1, 2, 3)),
                *((f"invariant_{i}", "qwen3_4b_bf16", BATCH_INVARIANT) for i in (1, 2, 3)),
            ]
        ],
        False,
        10 * 60,
    ),
    # The cause of the two bf16 paths (docs/design.md §10). Ten default engines, traced, so at a minority
    # rate near 2 in 8 at least one lands on the other path with probability 1 - 0.75**10 = 94%, and its
    # cuBLASLt algorithms can be diffed against the majority's. Each container ends with one batch-invariant
    # engine, to test that mode's path across containers against the 2026-10-13 run. Single-use, so two
    # fresh containers. 8 minutes per config: the slowest traced engine so far took about 210 s.
    "two_path_census": (
        [
            [
                *((f"{c}_default_{i}", "qwen3_4b_bf16", {CUBLAS_LOG: "1"}) for i in range(1, 6)),
                (f"{c}_invariant", "qwen3_4b_bf16", BATCH_INVARIANT),
            ]
            for c in "ab"
        ],
        True,
        8 * 60,
    ),
}


def downloads(experiment: str) -> list[tuple[str, str]]:
    """Every model revision an experiment needs."""
    containers = EXPERIMENTS[experiment][0]
    return sorted({(CONFIGS[e[1]][0]["model"], CONFIGS[e[1]][0]["revision"]) for group in containers for e in group})


# vLLM 0.30.0 log lines worth keeping in a manifest, matched as substrings of each line.
LOG_FACTS = {
    "kv_cache": ("Available KV cache memory", "KV cache size:", "Maximum concurrency for"),
    "kernels": ("Selected ", "attention backend", "FlashInfer"),
    "timings": ("Loading weights took", "Model loading took", "Graph capturing finished", "init engine (profile"),
    # Batch-invariant mode turns AOT loading off, so this line's absence shows the mode took effect.
    "compile": ("Directly load AOT compilation", "torch.compile took"),
}


def smoke_prompts(n: int, path: Path = PROMPTS) -> list[dict]:
    """The first n // 2 benign and the first n - n // 2 attack prompts in file order, interleaved
    as they appear. n = 1 gives the first attack prompt alone."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    want = {0: n // 2, 1: n - n // 2}
    chosen = []
    for row in rows:
        if want[row["label"]] > 0:
            chosen.append(row)
            want[row["label"]] -= 1
    if any(want.values()):
        raise ValueError(f"{path} has too few prompts of a label for n={n}")
    return chosen


def canary_prompts() -> list[dict]:
    """The path canary: the first batch of smoke_prompts(64), which was batch 1 of every Qwen3-4B
    capture since 2026-10-13, so a new capture's canary hash compares with the historical ones. It
    runs as the first batch of each engine, with the captured prompts after it."""
    return smoke_prompts(64)[: COMMON["batch_size"]]


def canary_sha256(activations: np.ndarray, layers: tuple[int, ...], positions: tuple[int, ...]) -> str:
    """SHA-256 of the canary's activations at layers 18 and 27 and position -1, as (prompt, layer,
    dim) little-endian float16 behind a header naming the format and shape. activations is laid out
    as hulder.capture.Capture's, (prompts, layers, positions, dims), with the config's layers and
    positions, so captures that record more layers or positions hash the same as the historical
    ones. Equal hashes mean bit-identical canaries. A hash not seen before means an unknown path,
    not that the path switched."""
    if activations.dtype.newbyteorder("<") != np.dtype("<f2"):  # either byte order
        raise TypeError(f"the canary hashes float16 as stored in snapshots, got {activations.dtype}")
    for layer in LAYERS:
        if layer not in layers:
            raise ValueError(f"the canary needs layer {layer}, the capture has {layers}")
    if -1 not in positions:
        raise ValueError(f"the canary needs position -1, the capture has {positions}")
    picked = activations[:, [layers.index(layer) for layer in LAYERS], positions.index(-1)]
    data = np.ascontiguousarray(picked, dtype="<f2")
    header = f"hulder-canary-v1 float16le {data.shape}\n".encode()
    return hashlib.sha256(header + data.tobytes()).hexdigest()


def check_canary_budget(token_counts: list[int], budget: int) -> None:
    """Raise if the canary's prompts, as vLLM tokenized them, total more than the scheduler's token
    budget for one step. With chunked prefill off, vLLM 0.30.0 then prefills them over more than one
    forward pass (v1/core/sched/scheduler.py:1072-1078), so the canary's batch differs from the
    historical one and its hash would mislead. The counts exist only after capture, so this detects
    a split after the fact rather than preventing it."""
    total = sum(token_counts)
    if total > budget:
        raise ValueError(
            f"the canary's {total} prompt tokens exceed the scheduler's per-step budget of {budget}, so it "
            "ran over more than one forward pass and its hash isn't comparable with the historical ones"
        )


def capture_order(request_ids: list[str], canary_ids: list[str], batch_size: int) -> list[str]:
    """The prompt ids an engine captures, in order: the canary as its first batch, then the request.
    A request whose first ids are exactly the canary's, in order, is captured as it is, so the
    canary runs once, as in every historical capture. Any other canary id in the request raises,
    since the canary's batch would then differ or its prompts run twice."""
    if len(canary_ids) != batch_size:
        raise ValueError(
            f"the canary must fill exactly one batch: {len(canary_ids)} prompts at batch size {batch_size}"
        )
    begins = request_ids[: len(canary_ids)] == canary_ids
    rest = request_ids[len(canary_ids) :] if begins else request_ids
    stray = [(i, x) for i, x in enumerate(rest, len(request_ids) - len(rest)) if x in set(canary_ids)]
    if stray:
        i, x = stray[0]
        raise ValueError(
            f"canary prompt {x} is at position {i} of the request. The request must begin with all "
            f"{len(canary_ids)} canary prompts in order, or hold none of them"
        )
    return request_ids if begins else canary_ids + request_ids


@dataclass(frozen=True)
class CanarySplit:
    canary_sha256: str
    canary_prompt_tokens: int
    activations: np.ndarray  # the requested prompts' rows, in request order


def split_canary(
    captured_ids: list[str],
    request_ids: list[str],
    canary_ids: list[str],
    activations: np.ndarray,
    token_counts: list[int],
    layers: tuple[int, ...],
    positions: tuple[int, ...],
    budget: int,
) -> CanarySplit:
    """Split a capture in capture_order into the canary's hash and the requested prompts' rows.
    Raises, so nothing gets recorded, if the canary's tokens exceed the per-step budget
    (check_canary_budget) or the capture isn't in capture_order."""
    if captured_ids != capture_order(request_ids, canary_ids, len(canary_ids)):
        raise ValueError("the captured prompts aren't in capture order, canary first then the request")
    n = len(canary_ids)
    check_canary_budget(token_counts[:n], budget)
    return CanarySplit(
        canary_sha256=canary_sha256(activations[:n], layers, positions),
        canary_prompt_tokens=sum(token_counts[:n]),
        activations=activations[len(captured_ids) - len(request_ids) :],
    )


def worker_spec(entry: tuple, timeout_s: int) -> dict:
    """The worker spec for one (label, config[, env]) entry of an experiment. The worker runs the
    canary as the engine's first batch, then the requested prompts."""
    fields, n = CONFIGS[entry[1]]
    return {
        "config": fields,
        "canary": canary_prompts(),
        "prompts": smoke_prompts(n),
        "save": n > 1,
        "config_name": entry[1],
        "base_model": BASE_MODEL_OVERRIDES.get(entry[1]),
        "env": entry[2] if len(entry) > 2 else {},
        "timeout_s": timeout_s,
    }


def log_facts(log: str) -> dict[str, list[str]]:
    """The log lines of each LOG_FACTS kind, stripped, in order, each line once."""
    lines = [line.strip() for line in log.splitlines()]
    return {
        kind: list(dict.fromkeys(line for line in lines if any(s in line for s in needles)))
        for kind, needles in LOG_FACTS.items()
    }


def estimate_cost(seconds: float, *, gpu: bool, cpu: float, memory_gib: float) -> float:
    """Modal's charge in USD for one container running this long, at the published rates."""
    per_s = (L4_PER_S if gpu else 0.0) + cpu * CPU_CORE_PER_S + memory_gib * MEMORY_GIB_PER_S
    return seconds * per_s


def worst_case_cost(n_configs: int, timeout_s: float = CONFIG_TIMEOUT_S) -> float:
    """What a capture container running n_configs can cost before their timeouts stop it."""
    return estimate_cost(n_configs * timeout_s, gpu=True, cpu=GPU_CPU, memory_gib=GPU_MEMORY_GIB)


def cublas_log_env(out_dir: str) -> dict:
    """Environment that traces cuBLASLt calls to files in out_dir (%i is the process id). The trace
    records each matmul's shapes and chosen algorithm. The legacy cuBLAS log adds only pointers."""
    return {"CUBLASLT_LOG_LEVEL": "2", "CUBLASLT_LOG_FILE": f"{out_dir}/cublaslt.%i.log"}


def _normalised_matmul(line: str) -> str:
    """A cuBLASLt matmul line from "[cublasLt]" on, with process id and pointers taken out."""
    line = line[line.index("[cublasLt]") :]
    line = re.sub(r"\[\d+\]", "[N]", line)  # process id
    line = re.sub(r"\b([A-D]|workSpace|stream)=0x[0-9a-f]+\s*", "", line, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", line).strip()


def matmul_signatures(text: str) -> set[str]:
    """The distinct cuBLASLt matmul calls (shapes, compute type, chosen algorithm) in a raw trace, a
    cublas_fingerprint file or a 2026-10-13 cublas_summary.txt, without counts. Two runs whose sets
    differ chose different GEMM algorithms somewhere."""
    return {_normalised_matmul(line) for line in text.splitlines() if "Matmul]" in line and "[cublasLt]" in line}


def cublas_fingerprint(log: str) -> list[str]:
    """Each distinct cuBLASLt matmul call (operand shapes, compute type and chosen algorithm) with
    pointers and process ids taken out, prefixed by how often it occurred. Two runs that chose
    different GEMM algorithms differ in this list. CUDA-graph replays make no calls, so it covers
    the eager GEMMs only."""
    counts: dict[str, int] = {}
    for line in log.splitlines():
        if "Matmul]" not in line or "[cublasLt]" not in line:
            continue
        line = _normalised_matmul(line)
        counts[line] = counts.get(line, 0) + 1
    return [f"{n:6d} {line}" for line, n in sorted(counts.items())]


def gpu_metadata(nvidia_smi: str) -> dict:
    """The GPU UUID and driver from nvidia-smi's name,uuid,driver_version,memory.total CSV line, for
    snapshot metadata: any difference between runs then warns (docs/design.md §6, metadata rule 2)."""
    _, uuid, driver, _ = (field.strip() for field in nvidia_smi.strip().splitlines()[0].split(","))
    return {"gpu_uuid": uuid, "gpu_driver": driver}


def relative_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-example ||a - b|| / ||a|| over the last axis, in float64."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(a, axis=-1)
