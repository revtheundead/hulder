"""The parts of the Modal capture runner that need neither Modal nor a GPU: prompt selection,
the configs and experiments, log facts, cost estimates and activation comparison. Tested in CI."""

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

# Stop rules: no config beyond 20 minutes, and no more than $3 (estimated) spent in a day, counting
# what was spent before the run (--spent-usd).
CONFIG_TIMEOUT_S = 20 * 60
BUDGET_USD = 3.0
MAX_CONFIGS_PER_CONTAINER = 4

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
# name -> (CaptureConfig fields, prompts captured, base_model override). The overrides are needed
# because base_model_for reads a quantized model's card: Qwen3-4B-FP8's lists its base as a
# one-element list it doesn't read, and online quantization of Qwen3-4B itself would read Qwen3-4B's
# card, which names its pretrained parent. The 8B run captures one prompt, only to read the KV-cache
# headroom bf16 leaves on 24 GB; its activations are dropped.
CONFIGS = {
    "qwen3_4b_bf16": ({**COMMON, **QWEN3_4B}, 64, None),
    "qwen3_4b_fp8": ({**COMMON, **QWEN3_4B_FP8}, 64, "Qwen/Qwen3-4B"),
    # vLLM's online FP8: bf16 weights quantized at load, per-tensor scales.
    "qwen3_4b_fp8_online": ({**COMMON, **QWEN3_4B, "quantization": "fp8"}, 64, "Qwen/Qwen3-4B"),
    "qwen3_8b_bf16_load": ({**COMMON, **QWEN3_8B}, 1, None),
}
# name -> (containers, single_use). Each container is one Modal call that runs its (label, config)
# pairs in order, each in a fresh engine process, so a comparison's runs share a GPU and driver.
# single_use experiments run their containers at once in containers that serve one call each, so
# each lands on its own GPU.
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
    ),
    "fp8_online": ([[("qwen3_4b_bf16", "qwen3_4b_bf16"), ("qwen3_4b_fp8_online", "qwen3_4b_fp8_online")]], False),
    "cross_container": ([[("bf16_a", "qwen3_4b_bf16")], [("bf16_b", "qwen3_4b_bf16")]], True),
}


def downloads(experiment: str) -> list[tuple[str, str]]:
    """Every model revision an experiment needs."""
    containers, _ = EXPERIMENTS[experiment]
    return sorted({(CONFIGS[c][0]["model"], CONFIGS[c][0]["revision"]) for group in containers for _, c in group})


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


def worst_case_cost(n_configs: int) -> float:
    """What a capture container running n_configs can cost before their timeouts stop it."""
    return estimate_cost(n_configs * CONFIG_TIMEOUT_S, gpu=True, cpu=GPU_CPU, memory_gib=GPU_MEMORY_GIB)


def gpu_metadata(nvidia_smi: str) -> dict:
    """The GPU UUID and driver from nvidia-smi's name,uuid,driver_version,memory.total CSV line, for
    snapshot metadata: any difference between runs then warns (docs/design.md §6, metadata rule 2)."""
    _, uuid, driver, _ = (field.strip() for field in nvidia_smi.strip().splitlines()[0].split(","))
    return {"gpu_uuid": uuid, "gpu_driver": driver}


def relative_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-example ||a - b|| / ||a|| over the last axis, in float64."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(a, axis=-1)
