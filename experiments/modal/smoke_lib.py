"""The parts of the Modal capture runner that need neither Modal nor a GPU: prompt selection,
the smoke configs, log facts, cost estimates and activation comparison. Tested in CI."""

import json
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

# Stop rules agreed for the smoke run: no config beyond 20 minutes, no run beyond $3 (estimated).
CONFIG_TIMEOUT_S = 20 * 60
RUN_BUDGET_USD = 3.0

SYSTEM_PROMPT = "You are a helpful assistant."
# Qwen3 has 36 decoder layers at both sizes. Two depths, half and three quarters, for the smoke run.
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
# name -> (CaptureConfig fields, prompts captured). The bf16 rerun is the same config in its own
# container, to measure the noise floor across containers. The 8B run only loads the engine and
# captures one prompt, to read the KV-cache headroom bf16 leaves on 24 GB; its activations are dropped.
SMOKE = {
    "qwen3_4b_bf16": ({**COMMON, **QWEN3_4B}, 64),
    "qwen3_4b_bf16_rerun": ({**COMMON, **QWEN3_4B}, 64),
    "qwen3_4b_fp8": ({**COMMON, **QWEN3_4B_FP8}, 64),
    "qwen3_8b_bf16_load": ({**COMMON, **QWEN3_8B}, 1),
}
# Every model revision the smoke run downloads.
DOWNLOADS = sorted({(cfg["model"], cfg["revision"]) for cfg, _ in SMOKE.values()})

# vLLM 0.30.0 log lines worth keeping in a manifest, matched as substrings of each line.
LOG_FACTS = {
    "kv_cache": ("Available KV cache memory", "KV cache size:", "Maximum concurrency for"),
    "kernels": ("Selected ", "attention backend", "FlashInfer"),
    "timings": ("Loading weights took", "Model loading took", "Graph capturing finished", "init engine (profile"),
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


def worst_case_config_cost() -> float:
    """What one capture container can cost before its timeout stops it."""
    return estimate_cost(CONFIG_TIMEOUT_S, gpu=True, cpu=GPU_CPU, memory_gib=GPU_MEMORY_GIB)


def relative_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-example ||a - b|| / ||a|| over the last axis, in float64."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(a, axis=-1)
