"""The parts of the Modal capture runner that need neither Modal nor a GPU: prompt selection,
the configs and experiments, log facts, cost estimates and activation comparison. Tested in CI."""

import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# .parent rather than .parents[2], which fails inside a container where this file is /root/smoke_lib.py.
ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS = ROOT / "examples" / "live_demo" / "demo_prompts.jsonl"
# The Phase 1 splits of S-Labs (docs/design.md §7), local only: the prompts travel in the worker spec.
SLABS_SPLITS = ROOT / ".cache" / "probe" / "splits.parquet"

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
# name -> (CaptureConfig fields, prompt source), where the source is ("smoke", n[, start]) or
# ("slabs_train", n, seed) (prompts_for). base_model_for finds every base model itself:
# Qwen3-4B-FP8's card names it, and online FP8 of Qwen3-4B is its own base. The 8B run captures one
# prompt, only to read the KV-cache headroom bf16 leaves on 24 GB; its activations are dropped.
CONFIGS = {
    "qwen3_4b_bf16": ({**COMMON, **QWEN3_4B}, ("smoke", 64)),
    "qwen3_4b_fp8": ({**COMMON, **QWEN3_4B_FP8}, ("smoke", 64)),
    # vLLM's online FP8: bf16 weights quantized at load, per-tensor scales.
    "qwen3_4b_fp8_online": ({**COMMON, **QWEN3_4B, "quantization": "fp8"}, ("smoke", 64)),
    "qwen3_8b_bf16_load": ({**COMMON, **QWEN3_8B}, ("smoke", 1)),
    # The timing pilot. With the canary in front, canary_check's batches are exactly the historical
    # 64-prompt captures', so its rows compare bit for bit with the recorded cluster its hash names.
    "qwen3_4b_bf16_canary_check": ({**COMMON, **QWEN3_4B}, ("smoke", 64, 8)),
    # Positions -9 and -10 for the D2 ablation (docs/design.md §5): <|im_end|> and the last content token.
    "qwen3_4b_bf16_pilot": ({**COMMON, **QWEN3_4B, "positions": (-1, -9, -10)}, ("slabs_train", 1000, 0)),
    # The full capture: the pilot's config on every split's prompts, the pilot's 1,000 first.
    "qwen3_4b_bf16_full": ({**COMMON, **QWEN3_4B, "positions": (-1, -9, -10)}, ("slabs_all", 0)),
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
    # Sizes the full Qwen3-4B capture from 1,000 S-Labs train prompts, and checks that a capture with
    # the canary in front reproduces the historical batches. Two fresh engines in one container.
    "pilot": (
        [[("canary_check", "qwen3_4b_bf16_canary_check"), ("pilot", "qwen3_4b_bf16_pilot")]],
        False,
        CONFIG_TIMEOUT_S,
    ),
    # Whether the pilot's canary_check result reproduces: it matched the majority bf16 path on
    # batches 1 to 7 and on no recorded capture on batch 8. Three untraced canary_checks, under the
    # pilot's engine environment, and one traced, each in its own fresh container, compared batch by
    # batch. 8 minutes per config, as the census; the timeout never reaches the engine.
    "canary_repeat": (
        [
            *([(f"u{i}_canary_check", "qwen3_4b_bf16_canary_check")] for i in (1, 2, 3)),
            [("t1_canary_check", "qwen3_4b_bf16_canary_check", {CUBLAS_LOG: "1"})],
        ],
        True,
        8 * 60,
    ),
    # The full capture as three replicas, each in its own fresh container on its own GPU, untraced,
    # so the kept dataset can be each batch's majority (full_capture_report.py). 12 minutes per
    # config is about 3 times the pilot's projection of engine startup and 2,043 batches.
    "full_capture": ([[(f"r{i}", "qwen3_4b_bf16_full")] for i in (1, 2, 3)], True, 12 * 60),
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


def slabs_train_prompts(n: int, seed: int, path: Path = SLABS_SPLITS) -> list[dict]:
    """n distinct S-Labs train prompts: the train rows sorted by id, so the parquet's row order
    doesn't matter, then the first n of numpy's default_rng(seed) permutation, in that order.
    Imports pandas only here, since the container imports this module but never reads the parquet."""
    import pandas as pd

    rows = pd.read_parquet(path, columns=["id", "split", "text", "label"])
    train = rows[rows["split"] == "train"].sort_values("id", kind="stable").reset_index(drop=True)
    if n > len(train):
        raise ValueError(f"{path} has {len(train)} train rows, fewer than n={n}")
    picked = train.iloc[np.random.default_rng(seed).permutation(len(train))[:n]]
    return [{"id": str(r.id), "text": str(r.text), "label": int(r.label)} for r in picked.itertuples()]


# The splits after train in a full capture, in capture order.
LATER_SPLITS = ("validation", "calibration", "test", "bpairs_train", "bpairs_validation", "bpairs_test")


def slabs_all_prompts(seed: int, path: Path = SLABS_SPLITS) -> list[dict]:
    """Every row of every split: train first, as slabs_train_prompts orders it, so any sample of it
    with this seed is a prefix, then each of LATER_SPLITS whole, its rows sorted by id and then in
    numpy's default_rng(seed) permutation. Raises on a split it doesn't know, so none is dropped."""
    import pandas as pd

    rows = pd.read_parquet(path, columns=["id", "split", "text", "label"])
    unknown = sorted(set(rows["split"]) - {"train", *LATER_SPLITS})
    if unknown:
        raise ValueError(f"{path} has splits {unknown} that a full capture doesn't place")
    prompts = slabs_train_prompts(int((rows["split"] == "train").sum()), seed, path)
    for split in LATER_SPLITS:
        part = rows[rows["split"] == split].sort_values("id", kind="stable").reset_index(drop=True)
        picked = part.iloc[np.random.default_rng(seed).permutation(len(part))]
        prompts += [{"id": str(r.id), "text": str(r.text), "label": int(r.label)} for r in picked.itertuples()]
    return prompts


def prompts_for(source: tuple) -> list[dict]:
    """The prompts a CONFIGS source names: ("smoke", n[, start]), smoke_prompts(n)[start:],
    ("slabs_train", n, seed) or ("slabs_all", seed)."""
    kind, *args = source
    if kind == "smoke":
        n, start = (*args, 0)[:2]
        return smoke_prompts(n)[start:]
    if kind == "slabs_train":
        return slabs_train_prompts(*args)
    if kind == "slabs_all":
        return slabs_all_prompts(*args)
    raise ValueError(f"unknown prompt source {source!r}")


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


def recorded_canary_hash(capture: Path) -> str:
    """The canary hash of a saved capture whose snapshots (<layer>.npz at layers 18 and 27,
    position -1) begin with the canary, as every Qwen3-4B capture since 2026-10-13 does. Equal to
    the hash the worker records for the same activations (split_canary). Raises if the snapshots
    don't begin with the canary."""
    import hulder

    snapshots = [hulder.Snapshot.load(capture / f"{layer}.npz") for layer in LAYERS]
    canary = [r["id"] for r in canary_prompts()]
    if any([str(i) for i in s.ids[: len(canary)]] != canary for s in snapshots):
        raise ValueError(f"{capture} doesn't begin with the canary")
    acts = np.stack([s.activations[: len(canary)] for s in snapshots], axis=1)[:, :, None]
    return canary_sha256(acts, LAYERS, (-1,))


def token_ids_sha256(token_ids: list) -> str:
    """SHA-256 of prompts' token ids in order, each prompt as its length then its ids, as
    little-endian uint32 behind a format header, so moving a prompt boundary changes the hash.
    Equal hashes mean the engine was fed the same tokens in the same prompts."""
    parts = [b"hulder-token-ids-v1\n"]
    for prompt in token_ids:
        ids = np.asarray(prompt, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= 2**32):
            raise ValueError("token ids must fit uint32")
        parts.append(np.array([len(ids)], dtype="<u4").tobytes() + ids.astype("<u4").tobytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()


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


# The full Qwen3-4B capture, about 16,300 prompts at batch 8 with the canary in front.
FULL_CAPTURE_BATCHES = 2040


def timing_estimate(
    engine_seconds: float,
    batch_seconds: list[float],
    *,
    batch_size: int = 8,
    n_batches: int = FULL_CAPTURE_BATCHES,
    timeout_s: float = CONFIG_TIMEOUT_S,
    seed: int = 0,
    resamples: int = 10_000,
) -> dict:
    """How long a capture of n_batches would take: engine_seconds + n_batches x the mean batch time,
    with a 95% percentile bootstrap band on the mean from resampling batch_seconds with
    default_rng(seed). It fits the timeout if the band's upper end, with startup, stays under it.
    The per-prompt median, 5th to 95th percentile and maximum describe the spread only. The band
    covers noise between batches in one engine, not between containers or GPUs."""
    times = np.asarray(batch_seconds, dtype=np.float64)
    resampled = times[np.random.default_rng(seed).integers(0, len(times), (resamples, len(times)))].mean(axis=1)
    band = [float(b) for b in np.percentile(resampled, [2.5, 97.5])]
    per_prompt = times / batch_size
    projection_band = [engine_seconds + n_batches * b for b in band]
    return {
        "batches": len(times),
        "total_seconds": float(times.sum()),
        "mean_batch_seconds": float(times.mean()),
        "mean_band": band,
        "per_prompt_seconds": {
            "median": float(np.median(per_prompt)),
            "p5": float(np.percentile(per_prompt, 5)),
            "p95": float(np.percentile(per_prompt, 95)),
            "max": float(per_prompt.max()),
        },
        "engine_seconds": engine_seconds,
        "n_batches": n_batches,
        "projection_seconds": engine_seconds + n_batches * float(times.mean()),
        "projection_band": projection_band,
        "timeout_s": timeout_s,
        "fits_timeout": bool(projection_band[1] < timeout_s),
        "bootstrap": {"seed": seed, "resamples": resamples, "confidence": 0.95},
    }


def worker_spec(entry: tuple, timeout_s: int) -> dict:
    """The worker spec for one (label, config[, env]) entry of an experiment. The worker runs the
    canary as the engine's first batch, then the requested prompts."""
    fields, source = CONFIGS[entry[1]]
    prompts = prompts_for(source)
    return {
        "config": fields,
        "canary": canary_prompts(),
        "prompts": prompts,
        "save": len(prompts) > 1,
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


def compare_batches(
    rows: dict[int, np.ndarray], reference: dict[int, np.ndarray], batch_size: int = 8, first_batch: int = 1
) -> list[dict]:
    """Per batch of rows, numbered from first_batch, whether it equals the reference bit for bit at
    every layer. A capture can match its path on most batches and leave it on one."""
    for layer in LAYERS:
        if rows[layer].shape != reference[layer].shape:
            raise ValueError(f"layer {layer}: rows of shape {rows[layer].shape} against {reference[layer].shape}")
    n = len(rows[LAYERS[0]])
    out = []
    for b, start in enumerate(range(0, n, batch_size), first_batch):
        span = slice(start, min(start + batch_size, n))
        same = all(np.array_equal(rows[la][span].view(np.uint16), reference[la][span].view(np.uint16)) for la in LAYERS)
        out.append({"batch": b, "rows": span.stop - span.start, "identical": same})
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _write_synced(path: Path, chunks: Iterable[bytes]) -> None:
    """Write through a .part file, fsynced, then rename, so a crash leaves no half-written file."""
    part = path.with_name(path.name + ".part")
    with open(part, "wb") as f:
        f.writelines(chunks)
        f.flush()
        os.fsync(f.fileno())
    part.replace(path)


def settle_volume_files(
    expected: dict[str, str],
    read: Callable[[str], Iterable[bytes]],
    remove: Callable[[str], None],
    local_dir: Path,
    backup_dir: Path | None,
    copy: Callable[[Path, Path], object] = shutil.copyfile,
) -> dict:
    """Fetch a config's files from the Volume into local_dir, check each against the sha256 the
    container recorded, copy them to backup_dir with a SHA256SUMS and check the copies, and only
    then remove them from the Volume. Any mismatch stops before the next step, so nothing is removed
    that isn't held twice, verified. Without backup_dir the files stay on the Volume. Re-reading a
    copy can be served from the page cache, so it shows the copy as written, not as stored."""
    status = {"files": expected, "fetched": False, "backed_up": False, "deleted_from_volume": False, "problem": None}
    local_dir.mkdir(parents=True, exist_ok=True)
    for name, sha in sorted(expected.items()):
        _write_synced(local_dir / name, read(name))
        if sha256_file(local_dir / name) != sha:
            status["problem"] = f"{name} as fetched doesn't match the container's sha256"
            return status
    status["fetched"] = True
    if backup_dir is None:
        return status
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name, sha in sorted(expected.items()):
        copy(local_dir / name, backup_dir / name)
        with open(backup_dir / name, "rb+") as f:
            os.fsync(f.fileno())
        if sha256_file(backup_dir / name) != sha:
            status["problem"] = f"the backup of {name} doesn't match the container's sha256"
            return status
    _write_synced(backup_dir / "SHA256SUMS", [f"{expected[n]}  {n}\n".encode() for n in sorted(expected)])
    status["backed_up"] = True
    for name in sorted(expected):
        remove(name)
    status["deleted_from_volume"] = True
    return status


def batch_digests(rows: np.ndarray, batch_size: int = 8) -> list[str]:
    """SHA-256 of each batch of rows' raw bytes, behind its dtype and shape, so equal digests mean
    bit-identical batches (-0.0 differs from 0.0). The last batch may be short."""
    out = []
    for start in range(0, len(rows), batch_size):
        batch = np.ascontiguousarray(rows[start : start + batch_size])
        header = f"{batch.dtype.str} {batch.shape}\n".encode()
        out.append(hashlib.sha256(header + batch.tobytes()).hexdigest())
    return out


def batch_agreement(digests: dict[str, dict[str, list[str]]], first_batch: int = 2) -> list[dict]:
    """Per batch, numbered from first_batch, which replicas agree bit for bit on every array, from
    each replica's batch_digests per array. "agree" when all do, "split" when more than half do
    (the majority) and the rest are outliers, else "no majority"."""
    labels = sorted(digests)
    keys = {label: sorted(digests[label]) for label in labels}
    if any(keys[label] != keys[labels[0]] for label in labels):
        raise ValueError(f"the replicas hold other arrays: {keys}")
    counts = {len(d) for label in labels for d in digests[label].values()}
    if len(counts) != 1:
        raise ValueError(f"the replicas hold other numbers of batches: {sorted(counts)}")
    out = []
    for b in range(counts.pop()):
        classes: dict[tuple, list[str]] = {}
        for label in labels:
            classes.setdefault(tuple(digests[label][k][b] for k in keys[label]), []).append(label)
        largest = max(classes.values(), key=len)
        if len(largest) == len(labels):
            kind, majority = "agree", largest
        elif 2 * len(largest) > len(labels):
            kind, majority = "split", largest
        else:
            kind, majority = "no majority", []
        outliers = [label for label in labels if label not in majority]
        out.append({"batch": b + first_batch, "kind": kind, "majority": majority, "outliers": outliers})
    return out


def majority_rows(
    arrays: dict[str, np.ndarray], agreement: list[dict], batch_size: int = 8, first_batch: int = 2
) -> np.ndarray:
    """Each batch's rows from a member of its majority, in capture order, leaving out the batches
    with no majority. arrays holds one array per replica, laid out as batch_agreement saw them."""
    parts = []
    for entry in agreement:
        if entry["majority"]:
            start = (entry["batch"] - first_batch) * batch_size
            parts.append(arrays[entry["majority"][0]][start : start + batch_size])
    first = next(iter(arrays.values()))
    return np.concatenate(parts) if parts else first[:0]


def expected_split_rates(p: float) -> dict[str, float]:
    """For three replicas that each leave the kept path on a batch with probability p,
    independently: the share of batches where exactly one does, a 2-to-1 split, and where two or
    more do. The second is an upper bound on batches with no majority, and also covers two
    replicas deviating alike, which would give an off-path majority nothing can detect."""
    split = 3 * p * (1 - p) ** 2
    return {"split": split, "two_or_more_deviate": 1 - (1 - p) ** 3 - split}


def relative_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-example ||a - b|| / ||a|| over the last axis, in float64."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(a, axis=-1)
