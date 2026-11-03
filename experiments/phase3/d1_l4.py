"""D1's cost measurement (docs/design.md §5 D1, docs/results/phase3/d1_local.md): four arms served
in turn on one GPU, interleaved over rounds, each arm-run in a fresh `vllm serve`.

- baseline: no probe
- b_stock: vLLM's hidden-state connector, writing every prompt's hidden states to disk
- b_noop: the same connector with its write removed (no_write_connector.py)
- lens: vllm-lens 1.3.0 in eager mode, with a persistent hook scoring a linear probe

A second experiment, ZERO_DRAFT, compares the baseline with b_zero_draft, B with a draft schedule of
zero tokens at every batch size on Model Runner V2. Its first round is a calibration round, each arm
running its greedy check and saturation bench only, and the fixed rate is 70% of the slower arm's
saturation. Later rounds run all three, so throughput pairs one round more than latency. Its stop
rules add a b_zero_draft arm-run off V2, where the setting doesn't work, or with any draft tokens.

A third, CONFIRMATION, adds b_stock to that, so the two B arms' cost is paired by round in one
container. After the fixed-rate bench in every later round, both B arms run a score stage (score_stage),
and a stage stops the run on other tokens than expected, too few rows, a non-finite activation or its
timeout. A b_stock arm-run with no drafts also stops it. d1_scores_report.py reads the stages.

Each arm-run records the GPU's free memory before the server starts, the server's KV-cache size
and maximum concurrency, a greedy generation check, a saturation bench (throughput), a fixed-rate
bench at 70% of round 0's baseline saturation rate (TTFT and TPOT), the peak running requests and
KV-cache use during the benches, and the spec-decode counters.

The cost rule (verdicts): an arm passes when the worst round's paired overhead against that round's
baseline is at most 5% for saturation throughput and 10% for fixed-rate p99 TTFT and p99 TPOT. A
metric fails when its smallest overhead across rounds exceeds the threshold plus the baseline's own
range across rounds. Otherwise it is inconclusive when that range exceeds half its threshold. An arm
fails on any failing metric, and an inconclusive one leaves it undecided only if none fails. Both
fail-first rules were decided after the 2026-10-31 L4 run (docs/results/phase3/d1_l4.md).

Stop rules, checked after every arm-run: a server that doesn't answer /health within its startup
timeout, a baseline or B measurement past its own timeout, a bench with failed requests, or an arm whose greedy replies diverge from round 0's baseline at more
non-tie tokens than a later baseline does (generation_fails). Each prompt counts once, at its
first divergence, and every first divergence is reported with the baseline's margin there
(divergence_table). A lens measurement past its timeout is a lens result instead, recorded with
how far it got (measurement_timeout), and the remaining arm-runs go on. Every arm except lens runs with VLLM_LENS_DISABLE=1, which left greedy
replies unchanged in the local check.

The L4 run goes through d1_modal.py. A smoke run of the whole path on the local GPU, with the
small model and a tiny workload, needs vllm-lens layered on:

    source scripts/dev-env.sh
    uv run --with vllm-lens==1.3.0 python experiments/phase3/d1_l4.py --smoke --out .cache/d1-l4-smoke

--smoke --overhead and --smoke --overhead-lens do the same for the Phase 3 overhead table's two
experiments, and overhead_report.py scores their stages. --smoke --zero-draft does the same for the zero-draft experiment, with a calibration round and one
more, and without vllm-lens. --smoke --confirmation does the same for the confirmation run, and
d1_scores_report.py --smoke scores its stages.
"""

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import NamedTuple

import d1_local
import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ("baseline", "b_stock", "b_noop", "lens")
BASELINE = "baseline"
LENS = "lens"
STOCK_ARM = "b_stock"
ZERO_DRAFT_ARM = "b_zero_draft"
SCORING_ARM = "scoring"
CACHING_ARM = "scoring_caching"
SCORING_ARMS = (SCORING_ARM, CACHING_ARM)
THRESHOLDS = {"throughput": 0.05, "p99_ttft": 0.10, "p99_tpot": 0.10}
FIXED_RATE_FRACTION = 0.7
# A greedy divergence starting where the reference's top two candidates are this close is a tie
# broken by rounding, not a change. The top two logprobs differ by exactly the difference of their
# logits, so the gap moves in steps of the logits' rounding: 2^-6 at |logit| 16 to 32 in fp16, as
# measured locally, and 0.125 to 0.25 in bf16 at |logit| 16 to 64. So two bf16 steps at 16 to 32,
# inclusive, with a tolerance for the float32 rounding of the logprobs themselves (the L4's prompt 13
# came back at 0.24999994).
BF16_LOGIT_STEP = 2**-3
TIE_STEPS = 2
TIE_TOLERANCE = 1e-6
TIE_BOUND = TIE_STEPS * BF16_LOGIT_STEP
PORT = 8766
STOP_GRACE_S = 30  # from SIGTERM to SIGKILL when a server is stopped
# A fixed-rate bench keeping up has completed every request that arrived this long ago: about four
# 128-token replies at an eager 60 ms a token. One that hasn't did not sustain the rate.
SUSTAIN_GRACE_S = 30
UNSUSTAINED = "did not sustain the fixed rate"
# Three times D1 confirmation's highest zero-draft stage against kept/ (mean |Δscore|): far enough past
# normal that the connector is more likely reading the wrong thing than a server having moved.
SCORE_SHIFT_STOP = 3 * 0.001951430574990809
MAX_MISSING_SCORES = 0.01  # the share of a stage's 16-token replies that may come back without a score
SCORE_OUTPUT_TOKENS = 16  # as the 2070's check, so decode steps share batches with prefills
COLLECT_ENV_FILE = "collect_env.txt"
IMAGE_UV_DIR = "/.uv"  # where Modal's uv_sync leaves uv, not on PATH
# Outside the arm-runs: the GPU query, collect_env, the comparisons and committing the compile cache.
MEASURE_SLACK_S = 5 * 60
COLLECT_ENV_TIMEOUT_S = 60  # it took 23 s on the 2070, and this comes out of MEASURE_SLACK_S


class Experiment(NamedTuple):
    """The arms, the rounds counting any calibration rounds, how many of those there are, and the
    arms that run a score stage in every other round."""

    arms: tuple[str, ...]
    rounds: int
    calibration_rounds: int = 0
    scored_arms: tuple[str, ...] = ()


COST = Experiment(ARMS, 3)
ZERO_DRAFT = Experiment((BASELINE, ZERO_DRAFT_ARM), 4, calibration_rounds=1)
CONFIRMATION = Experiment(
    (BASELINE, STOCK_ARM, ZERO_DRAFT_ARM), 4, calibration_rounds=1, scored_arms=(STOCK_ARM, ZERO_DRAFT_ARM)
)

# The Phase 3 overhead table. Qwen3-4B leaves lens to its 2026-10-31 result in d1_l4.md, cross-day but
# a 75% effect, so its three arms fit three measured rounds under the daily guard. The small model has
# no L4 lens data, so it keeps lens, and four arms fit two measured rounds.
OVERHEAD_QWEN3 = Experiment((BASELINE, *SCORING_ARMS), 4, calibration_rounds=1, scored_arms=SCORING_ARMS)
OVERHEAD_15B = Experiment((BASELINE, *SCORING_ARMS, LENS), 3, calibration_rounds=1, scored_arms=SCORING_ARMS)

# Every timed bench was 128-token replies, and a local check found a short-reply cost at the extraction
# path (docs/results/phase3/d3_drop.md). The fixed-rate bench gives up 15 s for the short bench's roughly
# 30 s, which leaves about 35 s of a Qwen3-4B arm-run's 240 s measure limit.
OVERHEAD_WORKLOAD = {
    "gen_tokens": 64,
    "saturation_prompts": 300,
    "fixed_seconds": 60,
    "output_len": 128,
    "short_output_len": 16,
}


def score_stages(experiment: Experiment) -> int:
    return (experiment.rounds - experiment.calibration_rounds) * len(experiment.scored_arms)


def schedule(arms: tuple[str, ...], rounds: int) -> list[tuple[int, str]]:
    """(round, arm) in run order: every arm once a round, the order rotated by one each round so no
    arm always follows another, and round 0 starting with the baseline."""
    return [(r, arms[(r + i) % len(arms)]) for r in range(rounds) for i in range(len(arms))]


class Timeouts(NamedTuple):
    """An arm-run's timeouts in seconds. Startup runs from launch until /health answers, and
    measurement from then to the end of the last bench. Round 0 runs every arm first in the
    container, so its servers may compile cold; later rounds reuse the compile cache it wrote."""

    cold_startup: float
    warm_startup: float
    measure: float
    score: float = 120

    def startup(self, rnd: int) -> float:
        return self.cold_startup if rnd == 0 else self.warm_startup


def worst_case_seconds(timeouts: Timeouts, rounds: int, arms: tuple[str, ...] = ARMS) -> float:
    """The longest the schedule can run: every arm-run at both its timeouts, then stopped."""
    return sum(timeouts.startup(rnd) + timeouts.measure + STOP_GRACE_S for rnd, _ in schedule(arms, rounds))


# The compile caches for all three arms are on the Volume, and at 480 s of cold start the guard total
# is $3.00 and leaves no room. A cold start would stop the run in round 0 for about $0.22.
CONFIRMATION_TIMEOUTS = Timeouts(cold_startup=270, warm_startup=270, measure=240, score=120)
# Qwen3-4B's zero-draft compile is on the Volume, though whether caching on reuses it is unverified. A
# cold start would stop the run in round 0, as above.
OVERHEAD_QWEN3_TIMEOUTS = Timeouts(cold_startup=270, warm_startup=270, measure=240, score=120)
# Nothing is compiled for the small model yet. Qwen3-4B's cold start took 386.6 s, and 420 s keeps the
# guard total below $3.
OVERHEAD_15B_TIMEOUTS = Timeouts(cold_startup=420, warm_startup=270, measure=240, score=120)


def measure_call_seconds(experiment: Experiment, timeouts: Timeouts) -> float:
    """The measure call's timeout: every arm-run and score stage at its timeouts, and the slack."""
    arm_runs = worst_case_seconds(timeouts, experiment.rounds, experiment.arms)
    return arm_runs + score_stages(experiment) * timeouts.score + MEASURE_SLACK_S


def fixed_rate(saturations: list[float]) -> float:
    """The fixed-rate bench's request rate, below every arm's saturation so none is measured past it."""
    return FIXED_RATE_FRACTION * min(saturations)


def zero_draft_stop(run: dict) -> str | None:
    """Why a b_zero_draft arm-run stops the run, or None. The setting only works on V2, and on any
    other runner, or if it stopped working, it would be measuring B under another name."""
    if run["arm"] not in (ZERO_DRAFT_ARM, *SCORING_ARMS):
        return None
    if run["startup"]["model_runner"] != "V2":
        return f"its startup log isn't V2: {run['startup']['model_runner']}"
    if run["arm"] in SCORING_ARMS and run["startup"]["async_scheduling"] is False:
        return "async scheduling is off, where zero-draft drafts again"
    if drafts := run["counters"]["draft_tokens"]:
        return f"{drafts} draft tokens"
    if drafts := run.get("score", {}).get("counters", {}).get("draft_tokens"):
        return f"{drafts} draft tokens in its score stage"
    return None


def stock_stop(run: dict) -> str | None:
    """Why a scored b_stock arm-run stops the run, or None. With no drafts it would no longer be B."""
    if run["arm"] == STOCK_ARM and not run["counters"]["draft_tokens"]:
        return "b_stock drafted no tokens"
    return None


def verdicts(runs: dict, baseline: str = BASELINE, measurement_timeouts: dict | None = None) -> dict:
    """The cost rule for every arm with a baseline in the same round. runs maps (round, arm) to
    {"saturation": bench result, "fixed": bench result}, a calibration round having no "fixed", so
    each metric reads the rounds that measured it. measurement_timeouts maps an arm to its rounds'
    measurement_timeout results. An arm that didn't sustain the fixed rate in any round
    fails, and one that ran out of measurement time can't pass."""
    measurement_timeouts = measurement_timeouts or {}
    metrics = {
        "throughput": ("saturation", "request_throughput", lambda arm, base: 1 - arm / base),
        "p99_ttft": ("fixed", "p99_ttft_ms", lambda arm, base: arm / base - 1),
        "p99_tpot": ("fixed", "p99_tpot_ms", lambda arm, base: arm / base - 1),
    }
    rounds = sorted({r for r, a in runs if a == baseline})
    out = {}
    for arm in sorted({a for _, a in runs if a != baseline} | set(measurement_timeouts)):
        paired = [r for r in rounds if (r, arm) in runs]
        timeouts = measurement_timeouts.get(arm, [])
        if not paired and not timeouts:
            continue
        result = {}
        for name, (bench, key, overhead) in metrics.items() if paired else ():
            base = [runs[(r, baseline)][bench][key] for r in rounds if bench in runs[(r, baseline)]]
            measured = [r for r in paired if bench in runs[(r, arm)] and bench in runs[(r, baseline)]]
            if not measured:  # a run stopped before this bench has nothing to say about this metric
                continue
            spread = (max(base) - min(base)) / (sum(base) / len(base))
            overheads = [overhead(runs[(r, arm)][bench][key], runs[(r, baseline)][bench][key]) for r in measured]
            threshold = THRESHOLDS[name]
            if min(overheads) > threshold + spread:
                verdict = "fail"  # past the threshold in every round by more than the baseline moves
            elif spread > threshold / 2:
                verdict = "inconclusive"
            else:
                verdict = "fail" if max(overheads) > threshold else "pass"
            result[name] = {
                "overheads": overheads,
                "upper": max(overheads),
                "threshold": threshold,
                "baseline_spread": spread,
                "verdict": verdict,
            }
        settled = [m["verdict"] for m in result.values()]
        passes = (
            False if "fail" in settled else None if "inconclusive" in settled else all(v == "pass" for v in settled)
        )
        if len(settled) < len(metrics) and passes:
            passes = None  # not every metric was measured, so nothing says it would pass
        if timeouts:
            result["measurement_timeouts"] = timeouts
            passes = False if any(t["outcome"] == UNSUSTAINED for t in timeouts) else passes and None
        result["passes"] = passes
        out[arm] = result
    return out


def short_reply_overheads(runs: list[dict]) -> dict:
    """Each arm's short-reply saturation loss against the same round's baseline. Reported beside the
    verdicts and not in them, so an arm's verdict stays comparable with D1's."""
    throughput = {
        (run["round"], run["arm"]): run["short_saturation"]["request_throughput"]
        for run in runs
        if "short_saturation" in run
    }
    out = {}
    for arm in sorted({a for _, a in throughput} - {BASELINE}):
        overheads = {
            rnd: 1 - throughput[(rnd, a)] / throughput[(rnd, BASELINE)]
            for rnd, a in sorted(throughput)
            if a == arm and (rnd, BASELINE) in throughput
        }
        threshold = THRESHOLDS["throughput"]
        out[arm] = {
            "overheads": overheads,
            "threshold": threshold,
            "past_threshold": max(overheads.values(), default=0) > threshold,
        }
    return out


def stock_against_zero_draft(runs: list[dict]) -> list[dict]:
    """b_stock's saturation throughput against b_zero_draft's, paired by round: the one comparison
    the two earlier runs could only make across days. Reported, with no pass mark."""
    throughput = {
        (run["round"], run["arm"]): run["saturation"]["request_throughput"] for run in runs if "saturation" in run
    }
    return [
        {
            "round": rnd,
            STOCK_ARM: throughput[(rnd, STOCK_ARM)],
            ZERO_DRAFT_ARM: throughput[(rnd, ZERO_DRAFT_ARM)],
            "stock_drop": 1 - throughput[(rnd, STOCK_ARM)] / throughput[(rnd, ZERO_DRAFT_ARM)],
        }
        for rnd in sorted({r for r, a in throughput if a == STOCK_ARM})
        if (rnd, ZERO_DRAFT_ARM) in throughput
    ]


def divergence_class(reference_margin: float | None) -> str:
    """A first divergence by the reference's top-two margin there. None means a reply ended early."""
    if reference_margin == 0:
        return "exact tie"
    if reference_margin is not None and reference_margin <= TIE_BOUND + TIE_TOLERANCE:
        return "within bound"
    return "change"


def non_tie_divergences(comparison: dict) -> int:
    """Prompts whose first divergence isn't a tie. compare_generations records only each prompt's
    first divergence, since everything after it differs anyway, so a prompt counts once."""
    return sum(divergence_class(d["margins"][0]) == "change" for d in comparison["diverged"])


def generation_fails(arm_vs_plain: dict, plain_vs_plain: list[dict]) -> bool | None:
    """Whether an arm's greedy replies diverge from the baseline's at more non-tie tokens than a
    fresh baseline does (d1_local.compare_generations results). None until a baseline has been
    compared with another."""
    if not plain_vs_plain:
        return None
    return non_tie_divergences(arm_vs_plain) > max(map(non_tie_divergences, plain_vs_plain))


def divergence_table(generation: dict) -> dict:
    """Every run's first divergences from round 0's baseline, each with the reference's margin and
    class, and the counts per class, so it shows whether TIE_BOUND excused any ("within bound")."""
    classes = ("exact tie", "within bound", "change")
    rows = [
        {
            "run": run,
            "prompt": d["prompt"],
            "token": d["token"],
            "tokens": d["tokens"],
            "reference_margin": d["margins"][0],
            "arm_margin": d["margins"][1],
            "class": divergence_class(d["margins"][0]),
        }
        for run, comparison in sorted(generation.items())
        for d in comparison["diverged"]
    ]
    counts = {run: dict.fromkeys(classes, 0) for run in sorted(generation)}
    for row in rows:
        counts[row["run"]][row["class"]] += 1
    tie_bound = {"steps": TIE_STEPS, "step": BF16_LOGIT_STEP, "tolerance": TIE_TOLERANCE}
    return {"tie_bound": tie_bound, "rows": rows, "counts": counts}


def measurement_timeout(stage: str, stage_seconds: float, rate: float | None, bench_log: str | None) -> dict:
    """A measurement timeout as a result: the stage it hit, how long that stage had run, the bench's
    last progress (its own seconds and requests completed of its total, None outside a bench) and
    the outcome. A fixed-rate bench that had completed fewer requests than had arrived
    SUSTAIN_GRACE_S before did not sustain the rate."""
    progress = re.findall(r"(\d+)/(\d+) \[(\d+):(\d+)", bench_log or "")
    completed, of, minutes, seconds = (int(n) for n in progress[-1]) if progress else (None,) * 4
    bench_seconds = None if minutes is None else 60 * minutes + seconds
    behind = (
        stage == "fixed" and bench_seconds is not None and completed < min(of, rate * (bench_seconds - SUSTAIN_GRACE_S))
    )
    return {
        "stage": stage,
        "stage_seconds": stage_seconds,
        "bench_seconds": bench_seconds,
        "completed": completed,
        "of": of,
        "outcome": UNSUSTAINED if behind else f"ran out of measurement time in {stage}",
    }


def startup_facts(log: str) -> dict:
    """The KV-cache size in tokens, the maximum concurrency, the model runner, the scheduled-token
    budget and async scheduling a server logged at startup. vLLM 0.30.0 logs "Using V2 Model Runner"
    (gpu_worker.py:441) and nothing for V1, so a server that sized its KV cache without that line ran
    V1. Before then the runner is unknown. It logs the budget only under a speculative config
    (config/vllm.py:2133). It logs no line for the resolved async scheduling, which is on unless the
    flag or a warning says otherwise (config/vllm.py:1438-1487)."""
    kv = re.search(r"GPU KV cache size: ([\d,]+) tokens", log)
    concurrency = re.search(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", log)
    runner = "V2" if "Using V2 Model Runner" in log else "V1" if kv else None
    budget = re.search(r"max_num_scheduled_tokens is set to (\d+)", log)
    flag = re.search(r"'async_scheduling': (True|False)", log)
    disabled = re.search(r"Async scheduling\b.*\bdisabled", log)
    asynchronous = flag.group(1) == "True" if flag else False if disabled else True if kv else None
    return {
        "kv_cache_tokens": int(kv.group(1).replace(",", "")) if kv else None,
        "max_concurrency": float(concurrency.group(1)) if concurrency else None,
        "model_runner": runner,
        "scheduled_token_budget": int(budget.group(1)) if budget else None,
        "async_scheduling": asynchronous,
    }


def model_runners(results: dict) -> dict:
    """Each arm's model runners across its arm-runs, for the manifest."""
    runners: dict[str, set] = {}
    for run in results["runs"]:
        if runner := run.get("startup", {}).get("model_runner"):
            runners.setdefault(run["arm"], set()).add(runner)
    return {arm: sorted(found) for arm, found in runners.items()}


def async_scheduling(results: dict) -> dict:
    """Each arm-run's async scheduling by name, for the manifest. None for one that never started."""
    return {f"r{run['round']}_{run['arm']}": run.get("startup", {}).get("async_scheduling") for run in results["runs"]}


def installed_packages() -> str:
    """Every installed distribution as name==version, one per line, sorted, read from package
    metadata as worker.py reads it, so recording the image's packages needs no subprocess."""
    from importlib.metadata import distributions

    lines = {f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata["Name"]}
    return "\n".join(sorted(lines, key=str.lower)) + "\n"


def score_workload(ids: list[str], texts: list[str], tokens_by_id: dict, concurrency: int = 64) -> dict:
    """A score stage's prompts with the tokens the offline capture recorded for each."""
    return {
        "score_ids": list(ids),
        "score_prompts": list(texts),
        "score_tokens": [list(tokens_by_id[i]) for i in ids],
        "score_output_tokens": SCORE_OUTPUT_TOKENS,
        "score_concurrency": concurrency,
    }


def check_templated(tokenizer, texts: list[str], expected: list[list[int]]) -> int:
    """How many prompts template to their expected tokens, as a score stage's requests do. Raises
    RuntimeError at the first that doesn't, so a template mismatch costs no GPU time."""
    for i, (text, tokens) in enumerate(zip(texts, expected, strict=True)):
        templated = tokenizer.apply_chat_template(
            d1_local.messages(text), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if tokenizer(templated, add_special_tokens=False)["input_ids"] != list(tokens):
            raise RuntimeError(f"prompt {i} templates to other tokens than expected")
    return len(texts)


def preflight(model: str, revision: str, score: dict | None = None) -> dict:
    """Everything an arm-run needs except a GPU, checked on CPU: the package list, the tokenizer
    read offline from the HF cache, the imports of the no-op connector and vllm-lens, the bench
    command, and any score stage's prompts against their expected tokens. Raises RuntimeError on the
    first that fails."""
    import importlib

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    started = time.time()
    try:
        path = snapshot_download(model, revision=revision, local_files_only=True)
    except LocalEntryNotFoundError as error:
        raise RuntimeError(f"{model} at {revision[:12]} isn't in the HF cache: {error}") from error
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}], tokenize=False, add_generation_prompt=True
    )
    templated = tokenizer(text, add_special_tokens=False)["input_ids"]
    sys.path.insert(0, str(HERE))
    for module in (
        "no_write_connector",
        "scoring_connector",
        "d3_spike",
        "vllm_lens",
        "vllm_lens.client",
        "prometheus_client.parser",
        "httpx",
    ):
        try:
            importlib.import_module(module)
        except ImportError as error:
            raise RuntimeError(f"can't import {module}: {error}") from error
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve", "--help"]
    bench_help = subprocess.run(command, capture_output=True, text=True, check=False)
    if bench_help.returncode != 0:
        raise RuntimeError(f"`vllm bench serve --help` exited {bench_help.returncode}: {bench_help.stderr[-500:]}")
    packages = installed_packages()
    scored = check_templated(tokenizer, score["score_prompts"], score["score_tokens"]) if score else None
    return {
        "score_prompts_templated": scored,
        "tokenizer_path": path,
        "tokenizer_vocab": len(tokenizer),
        "templated_tokens": len(templated),
        "packages": len(packages.splitlines()),
        "versions": {line.split("==")[0]: line.split("==")[1] for line in packages.splitlines()
                     if line.split("==")[0].lower() in ("vllm", "vllm-lens", "hulder", "torch", "transformers")},
        "seconds": round(time.time() - started, 1),
    }  # fmt: skip


def free_gpu_memory() -> dict:
    line = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    used, free = (int(x) for x in line.splitlines()[0].split(","))
    return {"used_mib": used, "free_mib": free}


def arm_command(arm: str, cfg: dict, storage: Path) -> tuple[list[str], dict]:
    """The `vllm serve` command and extra environment for one arm."""
    serve_flags = ["--enable-prefix-caching" if f == "--no-enable-prefix-caching" else f for f in cfg["serve"]]
    serve_flags = serve_flags if arm == CACHING_ARM else cfg["serve"]
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", cfg["model"], *serve_flags]
    command += ["--port", str(PORT)]
    env = {} if arm == "lens" else {"VLLM_LENS_DISABLE": "1"}
    if arm in (STOCK_ARM, "b_noop", ZERO_DRAFT_ARM, *SCORING_ARMS):
        speculative = {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [cfg["layer"]]}},
        }
        if arm in (ZERO_DRAFT_ARM, *SCORING_ARMS):
            speculative["num_speculative_tokens_per_batch_size"] = [[1, 1, 0]]
        transfer = {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": str(storage)},
        }
        if arm == "b_noop":
            transfer["kv_connector"] = "NoWriteHiddenStatesConnector"
            transfer["kv_connector_module_path"] = "no_write_connector"
            transfer["kv_connector_extra_config"]["use_synchronization_lock"] = False
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(HERE), os.environ.get("PYTHONPATH")]))
        if arm in SCORING_ARMS:
            transfer["kv_connector"] = "ScoringHiddenStatesConnector"
            transfer["kv_connector_module_path"] = "scoring_connector"
            transfer["kv_connector_extra_config"]["probe_path"] = str(cfg["probe_path"])
            transfer["kv_connector_extra_config"]["use_synchronization_lock"] = False
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(HERE), os.environ.get("PYTHONPATH")]))
        command += ["--speculative-config", json.dumps(speculative), "--kv-transfer-config", json.dumps(transfer)]
    return command, env


@contextlib.contextmanager
def serve(command: list[str], env: dict, log_path: Path, startup_timeout_s: float):
    """A server in its own process group, yielding its URL once /health answers, then stopped."""
    import httpx

    log = log_path.open("w")
    log.write(" ".join(command) + "\n")
    log.flush()
    proc = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **env}, start_new_session=True
    )
    url = f"http://127.0.0.1:{PORT}"
    deadline = time.time() + startup_timeout_s
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"the server exited with {proc.returncode}")
            with contextlib.suppress(httpx.HTTPError):
                if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                    break
            if time.time() > deadline:
                raise TimeoutError(
                    f"the server didn't answer /health within its {startup_timeout_s:g} s startup timeout"
                )
            time.sleep(2)
        yield url
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        log.close()


GAUGES = {"running": "vllm:num_requests_running", "kv_cache_usage": "vllm:kv_cache_usage_perc"}


class Peaks(threading.Thread):
    """Polls /metrics once a second for the peak running requests and KV-cache use."""

    def __init__(self, url: str):
        super().__init__(daemon=True)
        self.url, self.stop, self.peaks = url, threading.Event(), dict.fromkeys(GAUGES, 0.0)

    def run(self) -> None:
        import httpx
        from prometheus_client.parser import text_string_to_metric_families

        while not self.stop.wait(1.0):
            with contextlib.suppress(httpx.HTTPError):
                for family in text_string_to_metric_families(httpx.get(f"{self.url}/metrics", timeout=2).text):
                    for sample in family.samples:
                        for key, name in GAUGES.items():
                            if sample.name == name:
                                self.peaks[key] = max(self.peaks[key], sample.value)


def bench(url: str, cfg: dict, workload: dict, dataset: Path, out: Path, name: str, rate: float | None, n: int,
          deadline: float) -> dict:  # fmt: skip
    """One `vllm bench serve` run of n prompts, at rate requests per second or all at once."""
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--backend", "openai-chat", "--endpoint", "/v1/chat/completions", "--base-url", url,
        "--model", cfg["model"], "--tokenizer", cfg["tokenizer"],
        "--dataset-name", "custom", "--dataset-path", str(dataset), "--skip-chat-template",
        "--custom-output-len", str(workload["output_len"]), "--ignore-eos", "--temperature", "0",
        "--num-prompts", str(n), "--request-rate", "inf" if rate is None else f"{rate:.3f}", "--seed", "0",
        "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,99",
        "--save-result", "--result-dir", str(out), "--result-filename", f"{name}.json",
    ]  # fmt: skip
    with (out / f"{name}.log").open("w") as log:
        subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=max(1, deadline - time.time())
        )
    result = json.loads((out / f"{name}.json").read_text())
    keep = ("duration", "completed", "failed", "request_throughput", "output_throughput", "total_token_throughput",
            "max_concurrent_requests", "mean_ttft_ms", "p50_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p50_tpot_ms",
            "p99_tpot_ms", "p50_itl_ms", "p99_itl_ms", "p99_e2el_ms")  # fmt: skip
    return {k: result.get(k) for k in keep}


def make_probe_hook(weights, bias: float, layer: int):
    """A vllm-lens persistent hook that scores w·h + b at the last position of each request's
    first forward pass, and keeps the score on the GPU, so the hook adds no host sync."""
    from vllm_lens import Hook

    on_device = {}  # the weights, copied to the GPU once per process

    def score(ctx, h):
        if "score" not in ctx.saved:
            if (w := on_device.get(h.device)) is None:
                w = on_device[h.device] = weights.to(h.device, h.dtype)
            ctx.saved["score"] = h[-1] @ w + bias

    return Hook(fn=score, layer_indices=[layer])


def score_rows(order: np.ndarray, responses: list[dict], acts: np.ndarray, expected: list[list[int]]) -> np.ndarray:
    """A score stage's activations back in prompt order, once every prompt has answered with the
    tokens kept/ recorded for it and a finite activation. Raises RuntimeError, a stop rule, if not."""
    if len(responses) != len(expected) or len(acts) != len(expected):
        raise RuntimeError(f"the score stage returned {len(responses)} of {len(expected)} rows")
    for i, response in zip(order, responses, strict=True):
        if list(response["prompt_token_ids"]) != list(expected[i]):
            raise RuntimeError(f"prompt {i}'s saved token ids don't match the expected tokens")
    if not np.isfinite(acts).all():
        raise RuntimeError(f"{int((~np.isfinite(acts)).any(axis=1).sum())} rows hold a NaN or infinite activation")
    rows = np.empty_like(acts)
    rows[order] = acts
    return rows


def score_stage(url: str, cfg: dict, workload: dict, rnd: int, run_dir: Path, timeout_s: float) -> dict:
    """The calibration prompts through the server in a default_rng(round) order, as the offline
    capture templates them, so decode steps share batches with other prompts' prefills. Only each
    prompt's last-position hidden state stays, as score.npz, and the connector's files are deleted
    as they're read. Raises TimeoutError past timeout_s, and RuntimeError on any other stop rule."""
    started = time.time()
    texts = workload["score_prompts"]
    order = np.random.default_rng(rnd).permutation(len(texts))
    tokens = workload["score_output_tokens"]
    params = {
        "model": cfg["model"],
        "max_tokens": tokens,
        "min_tokens": tokens,
        "ignore_eos": True,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    replies = d1_local.chat_all(url, [texts[i] for i in order], workload["score_concurrency"], **params)
    responses = d1_local.asyncio.run(d1_local.asyncio.wait_for(replies, timeout_s))
    try:
        acts, _ = d1_local.connector_acts(responses)
    except ValueError as error:  # a connector file holding other tokens than its response
        raise RuntimeError(f"a saved hidden state doesn't match its prompt: {error}") from error
    rows = score_rows(order, responses, acts, workload["score_tokens"])
    if time.time() - started > timeout_s:
        raise TimeoutError(f"the score stage passed its {timeout_s:g} s timeout reading hidden states")
    np.savez(run_dir / "score.npz", ids=np.array(workload["score_ids"]), activations=rows, order=order)
    return {"rows": len(rows), "concurrency": workload["score_concurrency"], "seconds": round(time.time() - started, 1)}


def reply_scores(order: np.ndarray, responses: list[dict], expected: list[list[int]]) -> np.ndarray:
    """A stage's scores from its replies, in prompt order, NaN where a reply came back without one.
    Raises RuntimeError, a stop rule, on too few rows, other tokens or a score that isn't a probability."""
    if len(responses) != len(expected):
        raise RuntimeError(f"the score stage returned {len(responses)} of {len(expected)} rows")
    scores = np.full(len(expected), np.nan)
    for i, response in zip(order, responses, strict=True):
        if list(response["prompt_token_ids"]) != list(expected[i]):
            raise RuntimeError(f"prompt {i}'s token ids don't match the expected tokens")
        score = response["kv_transfer_params"]["hulder_score"]
        if score is None:
            continue
        if not 0 <= score <= 1:
            raise RuntimeError(f"prompt {i}'s score {score} isn't a probability")
        scores[i] = score
    return scores


def stage_summary(scores: np.ndarray, metrics_scores: int, reference: np.ndarray | None = None,
                  shift_stop: float | None = None, allowed_missing: float = MAX_MISSING_SCORES) -> dict:  # fmt: skip
    """A stage's counts and any shift against a reference. Raises RuntimeError, a stop rule, on too many
    replies without a score, a score /metrics didn't count, or a shift past shift_stop."""
    present = ~np.isnan(scores)
    missing = int((~present).sum())
    if missing > allowed_missing * len(scores):
        raise RuntimeError(f"{missing} of {len(scores)} replies came back without a score")
    if metrics_scores != len(scores):
        raise RuntimeError(f"/metrics has {metrics_scores} scores for {len(scores)} requests")
    summary = {"without_score": missing}
    if reference is not None:
        summary["mean_abs_dscore"] = float(np.abs(scores[present] - reference[present]).mean())
        if shift_stop is not None and summary["mean_abs_dscore"] > shift_stop:
            raise RuntimeError(
                f"mean |Δscore| {summary['mean_abs_dscore']:.6f} is above its stop rule, {shift_stop:.6f}"
            )
    return summary


def shift_stop_for(arm: str, workload: dict) -> float | None:
    """Only the caching-off arm stops on shift. With caching on, the shift is what's being measured."""
    return workload.get("score_shift_stop") if arm == SCORING_ARM else None


CACHE_COUNTERS = {"queried_tokens": "vllm:prefix_cache_queries_total", "hit_tokens": "vllm:prefix_cache_hits_total"}


def cache_counters(before: str, after: str) -> dict:
    """Prefix-cache tokens queried and hit between two /metrics scrapes."""
    from prometheus_client.parser import text_string_to_metric_families

    def read(text: str) -> dict:
        totals = dict.fromkeys(CACHE_COUNTERS.values(), 0.0)
        for family in text_string_to_metric_families(text):
            for sample in family.samples:
                if sample.name in totals:
                    totals[sample.name] += sample.value
        return totals

    b, a = read(before), read(after)
    return {key: round(a[name] - b[name]) for key, name in CACHE_COUNTERS.items()}


def reply_score_stage(url: str, cfg: dict, workload: dict, rnd: int, run_dir: Path, timeout_s: float, arm: str) -> dict:
    """score_stage for the scoring connector: the same prompts and order, with each score read from its
    reply and /metrics, and no hidden states. Saves score.npz. Raises TimeoutError past timeout_s, and
    RuntimeError on any other stop rule."""
    import d3_spike
    import httpx

    started = time.time()
    texts = workload["score_prompts"]
    order = np.random.default_rng(rnd).permutation(len(texts))
    tokens = workload["score_output_tokens"]
    params = {
        "model": cfg["model"],
        "max_tokens": tokens,
        "min_tokens": tokens,
        "ignore_eos": True,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    before = httpx.get(f"{url}/metrics").text
    replies = d1_local.chat_all(url, [texts[i] for i in order], workload["score_concurrency"], **params)
    responses = d1_local.asyncio.run(d1_local.asyncio.wait_for(replies, timeout_s))
    scores = reply_scores(order, responses, workload["score_tokens"])
    metrics = d3_spike.settled_metrics(url, before, len(texts))
    reference = np.array(workload["score_reference"]) if "score_reference" in workload else None
    summary = stage_summary(scores, metrics["scores"], reference, shift_stop_for(arm, workload))
    if time.time() - started > timeout_s:
        raise TimeoutError(f"the score stage passed its {timeout_s:g} s timeout waiting for /metrics")
    np.savez(run_dir / "score.npz", ids=np.array(workload["score_ids"]), scores=scores, order=order)
    return {
        "rows": len(scores),
        "concurrency": workload["score_concurrency"],
        "metrics": metrics,
        "seconds": round(time.time() - started, 1),
        **summary,
    }


def run_arm(arm: str, rnd: int, cfg: dict, workload: dict, out: Path, rate: float | None,
            timeouts: Timeouts, calibrate: bool = False, score: bool = False) -> dict:  # fmt: skip
    """One arm-run. rate is the fixed-rate bench's, None in round 0's baseline, which sets it. A
    calibration arm-run stops after its saturation bench. score adds a score stage after the bench
    counters are read, with its own timeout and counters, so it can't change what the benches report."""
    import httpx

    run_dir = out / f"r{rnd}_{arm}"
    run_dir.mkdir(parents=True, exist_ok=True)
    storage = run_dir / "hidden_states"
    command, env = arm_command(arm, cfg, storage)
    result = {"arm": arm, "round": rnd, "free_gpu_before": free_gpu_memory(), "command": command, "env": env}
    started = time.time()
    stage, peaks = None, None
    try:
        with serve(command, env, run_dir / "server.log", timeouts.startup(rnd)) as url:
            result["startup_seconds"] = round(time.time() - started, 1)
            deadline = time.time() + timeouts.measure
            stage, stage_started = "generate", time.time()
            if arm == "lens":
                import torch
                from vllm_lens.client import VLLMLensClient

                generator = torch.Generator().manual_seed(0)
                weights = torch.randn(cfg["hidden_size"], generator=generator) / math.sqrt(cfg["hidden_size"])
                # vllm-lens's layer index is one below hulder's layer id (d1_local.md, check 5).
                VLLMLensClient(url).register_hooks([make_probe_hook(weights, 0.0, cfg["layer"] - 1)])
            params = {"model": cfg["model"], "max_tokens": workload["gen_tokens"], "logprobs": True, "top_logprobs": 2}
            replies = d1_local.chat_all(url, workload["gen_prompts"], 1, **params)
            result["generate"] = d1_local.asyncio.run(d1_local.asyncio.wait_for(replies, deadline - time.time()))
            dataset = run_dir / "bench.jsonl"
            dataset.write_text("".join(json.dumps({"prompt": p}) + "\n" for p in workload["bench_prompts"]))
            before = httpx.get(f"{url}/metrics").text
            peaks = Peaks(url)
            peaks.start()
            stage, stage_started = "saturation", time.time()
            n_sat = workload["saturation_prompts"]
            result["saturation"] = bench(url, cfg, workload, dataset, run_dir, "saturation", None, n_sat, deadline)
            if "short_output_len" in workload:
                short = {**workload, "output_len": workload["short_output_len"]}
                scrape = httpx.get(f"{url}/metrics").text
                stage, stage_started = "short_saturation", time.time()
                result["short_saturation"] = bench(url, cfg, short, dataset, run_dir, stage, None, n_sat, deadline)
                if arm == CACHING_ARM:
                    result["short_cache"] = cache_counters(scrape, httpx.get(f"{url}/metrics").text)
            if not calibrate:
                rate = rate or fixed_rate([result["saturation"]["request_throughput"]])
                n_fixed = min(len(workload["bench_prompts"]), max(1, round(rate * workload["fixed_seconds"])))
                result["fixed_rate"] = rate
                stage, stage_started = "fixed", time.time()
                result["fixed"] = bench(url, cfg, workload, dataset, run_dir, "fixed", rate, n_fixed, deadline)
            after_benches = httpx.get(f"{url}/metrics").text
            result["counters"] = d1_local.step_tokens(before, after_benches)
            if arm == CACHING_ARM:
                result["cache"] = cache_counters(before, after_benches)
            if arm == "lens":
                result["hook_results"] = len(VLLMLensClient(url).collect_hook_results())
            if score:
                peaks.stop.set()  # the peaks are the benches'
                before = httpx.get(f"{url}/metrics").text
                stage, stage_started = "score", time.time()
                if arm in SCORING_ARMS:
                    result["score"] = reply_score_stage(url, cfg, workload, rnd, run_dir, timeouts.score, arm)
                else:
                    result["score"] = score_stage(url, cfg, workload, rnd, run_dir, timeouts.score)
                after_stage = httpx.get(f"{url}/metrics").text
                result["score"]["counters"] = d1_local.step_tokens(before, after_stage)
                if arm == CACHING_ARM:
                    result["score"]["cache"] = cache_counters(before, after_stage)
    except (RuntimeError, TimeoutError, subprocess.SubprocessError, httpx.HTTPError) as error:
        result["error"] = f"{type(error).__name__}: {str(error) or 'the measurement timeout passed'}"
        if stage and isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
            log = run_dir / f"{stage}.log"
            result["timeout"] = measurement_timeout(
                stage,
                round(time.time() - stage_started, 1),
                result.get("fixed_rate"),
                log.read_text(errors="replace") if log.exists() else None,
            )
    finally:
        if peaks:
            peaks.stop.set()
            peaks.join()
            result["peaks"] = peaks.peaks
        shutil.rmtree(storage, ignore_errors=True)
    result["seconds"] = round(time.time() - started, 1)
    result["startup"] = startup_facts((run_dir / "server.log").read_text(errors="replace"))
    return result


def run_all(cfg: dict, workload: dict, out: Path, experiment: Experiment, timeouts: Timeouts) -> dict:
    """Every arm-run of the schedule in order, stopping at the first stop rule that fires."""
    runs, replies, lens_timeouts, plan = {}, {}, [], schedule(experiment.arms, experiment.rounds)
    results = {"schedule": plan, "runs": [], "stopped": None}
    rate = None
    for rnd, arm in plan:
        print(f"round {rnd}: {arm}", flush=True)
        calibrate = rnd < experiment.calibration_rounds
        score = arm in experiment.scored_arms and not calibrate
        run = run_arm(arm, rnd, cfg, workload, out, rate, timeouts, calibrate=calibrate, score=score)
        generated = run.pop("generate", None)
        results["runs"].append(run)
        # Lens runs eager, so it may not keep up with the baseline's fixed rate. That is its result.
        lens_timed_out = arm == LENS and "timeout" in run
        if "error" in run and not lens_timed_out:
            results["stopped"] = f"round {rnd} {arm}: {run['error']}"
            break
        if failed := sum(run[b]["failed"] for b in ("saturation", "short_saturation", "fixed") if b in run):
            results["stopped"] = f"round {rnd} {arm}: {failed} failed requests"
            break
        if reason := zero_draft_stop(run) or (experiment.scored_arms and stock_stop(run)):
            results["stopped"] = f"round {rnd} {arm}: {reason}"
            break
        if lens_timed_out:
            lens_timeouts.append({"round": rnd, **run["timeout"]})
        else:
            runs[(rnd, arm)] = {b: run[b] for b in ("saturation", "fixed") if b in run}
        if generated is not None:
            replies[(rnd, arm)] = generated
        if experiment.calibration_rounds == 0 and rnd == 0 and arm == BASELINE:
            rate = run["fixed_rate"]
        if calibrate and len(results["runs"]) == experiment.calibration_rounds * len(experiment.arms):
            # Lens, at about a quarter of the baseline, would set a rate well below serving load.
            saturations = [r["saturation"]["request_throughput"] for (_, a), r in runs.items() if a != LENS]
            rate = results["fixed_rate"] = fixed_rate(saturations)
        reference = replies[(0, BASELINE)]
        comparisons = {
            key: d1_local.compare_generations(reference, value)
            for key, value in replies.items()
            if key != (0, BASELINE)
        }
        plain = [c for (r, a), c in comparisons.items() if a == BASELINE]
        results["generation"] = {f"r{r}_{a}": c for (r, a), c in comparisons.items()}
        # Caching on can change replies, since it changes the numbers. That's reported, not a stop.
        judged = {k: c for k, c in comparisons.items() if k[1] not in (BASELINE, CACHING_ARM)}
        fails = [f"r{r}_{a}" for (r, a), c in judged.items() if generation_fails(c, plain)]
        if fails:
            results["stopped"] = f"greedy replies changed beyond a fresh baseline's: {', '.join(fails)}"
            break
    results["verdicts"] = verdicts(runs, measurement_timeouts={LENS: lens_timeouts} if lens_timeouts else None)
    results["divergences"] = divergence_table(results.get("generation", {}))
    if experiment.scored_arms:
        results["stock_vs_zero_draft"] = stock_against_zero_draft(results["runs"])
    if any("short_saturation" in run for run in results["runs"]):
        results["short_replies"] = short_reply_overheads(results["runs"])
    return results


def rejudge(results: dict) -> dict:
    """The cost rule's verdicts and the divergence table recomputed from a finished run's
    d1_l4.json, for a run judged under earlier versions of them."""
    runs, timeouts = {}, []
    for run in results["runs"]:
        if run["arm"] == LENS and "timeout" in run:
            timeouts.append({"round": run["round"], **run["timeout"]})
        elif "error" not in run:
            runs[(run["round"], run["arm"])] = {b: run[b] for b in ("saturation", "fixed") if b in run}
    return {
        "verdicts": verdicts(runs, measurement_timeouts={LENS: timeouts} if timeouts else None),
        "divergences": divergence_table(results.get("generation", {})),
    }


def environment_path(run_id: str) -> Path:
    """Where a run's environment record sits on the Volume, relative to its root."""
    return Path("d1-env") / run_id / COLLECT_ENV_FILE


def collect_env_environ(extra: dict | None = None) -> dict:
    """The environment `python -m vllm.collect_env` runs in: the image's uv and this venv, with no uv
    cache, since XDG_CACHE_HOME is on a Volume and uv can't take its lock there (run 20261102T202134Z)."""
    path = f"{IMAGE_UV_DIR}{os.pathsep}{os.environ['PATH']}"
    return {**os.environ, "PATH": path, "VIRTUAL_ENV": sys.prefix, "UV_NO_CACHE": "1", **(extra or {})}


def uv_check(attempts: dict[str, dict]) -> dict:
    """`uv pip list` as collect_env runs it, once per set of extra variables, with what collect_env
    discards: the exit code and the error. A variable set to None is unset."""
    out = {}
    for name, extra in attempts.items():
        env = {k: v for k, v in collect_env_environ(extra).items() if v is not None}
        done = subprocess.run(
            [str(Path(IMAGE_UV_DIR) / "uv"), "pip", "list", "--format=freeze"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            stdin=subprocess.DEVNULL,
        )
        out[name] = {
            "returncode": done.returncode,
            "packages": len(done.stdout.splitlines()),
            "stderr": done.stderr.strip()[:2000],
        }
    return out


def record_environment(out: Path, persist: Path, commit, command: list[str] | None = None,
                       timeout_s: float = COLLECT_ENV_TIMEOUT_S) -> dict:  # fmt: skip
    """`python -m vllm.collect_env` as out/collect_env.txt, also written under persist and committed,
    so a function timeout or crash doesn't lose it. Nothing here stops the run: a failure goes in the
    file and the record it returns. Output goes straight to the file and the process has its own
    group, so a child left holding it open can't hold up the first server. With no pip in the venv it
    falls back to `uv pip list` at 0.30.0 and crashes if that fails (collect_env.py:672), so it gets
    the image's uv and this venv."""
    path = out / COLLECT_ENV_FILE
    started = time.time()
    record: dict = {"returncode": None}
    with path.open("wb") as file:
        try:
            proc = subprocess.Popen(
                command or [sys.executable, "-m", "vllm.collect_env"],
                stdin=subprocess.DEVNULL,
                stdout=file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=collect_env_environ(),
            )
        except OSError as error:
            note = f"collect_env couldn't run: {error}"
        else:
            try:
                record["returncode"] = proc.wait(timeout=timeout_s)
                note = f"collect_env exited {proc.returncode}" if proc.returncode else None
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                note = f"collect_env timed out after {timeout_s:g} s"
        if note:
            file.write(f"\n{note}\n".encode())
            record["note"] = note
    record["seconds"] = round(time.time() - started, 1)
    try:
        persist.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, persist / COLLECT_ENV_FILE)
        commit()
    except Exception as error:  # noqa: BLE001 - a Volume problem must not cost the run
        record["persist_error"] = f"{type(error).__name__}: {error}"
    return record


def recover_environment(read_file, run_id: str, run_dir: Path) -> bool:
    """The committed environment record into run_dir, for a call that returned no files. False if
    there's none to read, which leaves the original failure as the one reported."""
    try:
        data = b"".join(read_file(str(environment_path(run_id))))
    except Exception as error:  # noqa: BLE001 - Modal's not-found errors vary, and the call's failure comes first
        print(f"no environment record on the Volume: {type(error).__name__}: {error}", flush=True)
        return False
    (run_dir / COLLECT_ENV_FILE).write_bytes(data)
    return True


def in_container(cfg: dict, workload: dict, out: Path, experiment: Experiment, timeouts: Timeouts, persist: Path,
                 commit, command: list[str] | None = None) -> dict:  # fmt: skip
    """What one container does: the environment record first, the image's packages, every arm-run,
    then the results and every file to return."""
    out.mkdir(parents=True, exist_ok=True)
    environment = record_environment(out, persist, commit, command)
    (out / "packages.txt").write_text(installed_packages())
    results = run_all(cfg, workload, out, experiment, timeouts)
    commit()  # compile caches, for later runs
    files = {str(p.relative_to(out)): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}
    return {"results": results, "environment": environment, "files": files}


SMOKE = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "tokenizer": "Qwen/Qwen2.5-1.5B-Instruct",
    "hidden_size": 1536,
    "layer": 20,
    "serve": [
        "--revision",
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "--dtype",
        "float16",
        "--gpu-memory-utilization",
        "0.7",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "8",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--seed",
        "0",
    ],
}


def bundled_probe_path() -> str:
    from importlib.resources import files

    return str(files("hulder") / "data" / "probe_demo.npz")


OVERHEAD_15B_CFG = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "hidden_size": 1536,
    "layer": 20,
    "probe_path": bundled_probe_path(),
    # fp16 to match the probe and its reference, captured in fp16 on the 2070.
    "serve": [
        "--revision", d1_local.REVISION, "--dtype", "float16", "--max-model-len", "2048",
        "--enable-chunked-prefill", "--no-enable-prefix-caching", "--seed", "0",
    ],
}  # fmt: skip


def templated_demo_tokens(texts: list[str]) -> list[list[int]]:
    """Each demo prompt's tokens, templated as a score stage's requests are."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(d1_local.MODEL, revision=d1_local.REVISION)
    return [
        tokenizer(
            tokenizer.apply_chat_template(
                d1_local.messages(text), tokenize=False, add_generation_prompt=True, enable_thinking=False
            ),
            add_special_tokens=False,
        )["input_ids"]
        for text in texts
    ]


def demo_score_workload(concurrency: int = 64, tokens_for=templated_demo_tokens) -> dict:
    """The 360 demo prompts as a reply score stage, with the bundled reference's scores to report against."""
    import hulder

    rows = d1_local.prompts()
    ids, texts = [p["id"] for p in rows], [p["text"] for p in rows]
    reference = hulder.Snapshot.load(d1_local.DATA / "demo_fp16.npz")
    if list(reference.ids) != ids:
        raise SystemExit("demo prompts and the bundled reference aren't in the same order")
    probe = hulder.Probe.load(bundled_probe_path())
    return {
        **score_workload(ids, texts, dict(zip(ids, tokens_for(texts), strict=True)), concurrency=concurrency),
        "score_reference": hulder.apply_probe(probe, reference.activations).tolist(),
    }


def smoke_overhead(workload: dict, out: Path, timeouts: Timeouts, lens: bool) -> int:
    """An overhead experiment's whole path on the local GPU, through in_container as a container runs
    it: a calibration round and one more, all 360 demo prompts in its score stages. With lens, the small
    model's four arms, under `uv run --with vllm-lens==1.3.0`. Without, Qwen3-4B's three arms."""
    experiment = (OVERHEAD_15B if lens else OVERHEAD_QWEN3)._replace(rounds=2)
    cfg = {**SMOKE, "probe_path": bundled_probe_path()}
    workload = {**workload, "short_output_len": 8, **demo_score_workload(concurrency=8)}
    returned = in_container(cfg, workload, out, experiment, timeouts, out.parent / f"{out.name}-volume", lambda: None)
    results = returned["results"]
    (out / "d1_l4.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    (out / "manifest.json").write_text(json.dumps({"environment": returned["environment"]}, indent=2) + "\n")
    print(json.dumps({k: results.get(k) for k in ("stopped", "fixed_rate", "verdicts", "short_replies")}, indent=2))
    return 0 if results["stopped"] is None else 1


def smoke_confirmation(workload: dict, out: Path, timeouts: Timeouts) -> int:
    """The confirmation run's whole path on the local GPU, through in_container as a container runs
    it, with all 360 demo prompts in its score stage, for d1_scores_report.py --smoke. Writes what
    that report reads: d1_l4.json, each score.npz, and a manifest with no GPU line."""
    from transformers import AutoTokenizer

    rows = d1_local.prompts()
    ids, texts = [p["id"] for p in rows], [p["text"] for p in rows]
    tokenizer = AutoTokenizer.from_pretrained(SMOKE["model"], revision=d1_local.REVISION)
    templated = [
        tokenizer(
            tokenizer.apply_chat_template(
                d1_local.messages(text), tokenize=False, add_generation_prompt=True, enable_thinking=False
            ),
            add_special_tokens=False,
        )["input_ids"]
        for text in texts
    ]
    workload = {**workload, **score_workload(ids, texts, dict(zip(ids, templated, strict=True)), concurrency=8)}
    returned = in_container(
        SMOKE,
        workload,
        out,
        CONFIRMATION._replace(rounds=2),
        timeouts,
        out.parent / f"{out.name}-volume",
        lambda: None,
    )
    results = returned["results"]
    (out / "d1_l4.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    (out / "manifest.json").write_text(json.dumps({"environment": returned["environment"]}, indent=2) + "\n")
    keys = ("stopped", "verdicts", "stock_vs_zero_draft", "divergences")
    print(json.dumps({k: results.get(k) for k in keys}, indent=2))
    return 0 if results["stopped"] is None else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--smoke", action="store_true", help="the whole path on the local GPU, tiny workload")
    parser.add_argument(
        "--zero-draft", action="store_true", help="smoke the zero-draft experiment: a calibration round and one more"
    )
    parser.add_argument(
        "--confirmation", action="store_true", help="smoke the confirmation run: a calibration round and one more"
    )
    parser.add_argument(
        "--overhead", action="store_true", help="smoke Qwen3-4B's overhead arms: a calibration round and one more"
    )
    parser.add_argument(
        "--overhead-lens", action="store_true", help="smoke the small model's four overhead arms, with vllm-lens"
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--rejudge", type=Path, help="print the verdicts and divergences recomputed from this d1_l4.json"
    )
    args = parser.parse_args(argv)
    if args.rejudge:
        print(json.dumps(rejudge(json.loads(args.rejudge.read_text())), indent=2))
        return 0
    if args.out is None:
        parser.error("--out is required")
    if not args.smoke:
        raise SystemExit("the L4 run goes through d1_modal.py; here only --smoke runs")
    texts = [p["text"] for p in d1_local.prompts()]
    workload = {
        "gen_prompts": texts[:4],
        "gen_tokens": 16,
        "bench_prompts": texts,
        "saturation_prompts": 16,
        "fixed_seconds": 5,
        "output_len": 16,
    }
    timeouts = Timeouts(900, 900, 900)
    if args.confirmation:
        return smoke_confirmation(workload, args.out, timeouts)
    if args.overhead or args.overhead_lens:
        return smoke_overhead(workload, args.out, timeouts, lens=args.overhead_lens)
    args.out.mkdir(parents=True, exist_ok=True)
    experiment = ZERO_DRAFT._replace(rounds=2) if args.zero_draft else COST._replace(rounds=1)
    results = run_all(SMOKE, workload, args.out, experiment, timeouts=timeouts)
    (args.out / "d1_l4.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(json.dumps({k: results[k] for k in ("stopped", "verdicts", "divergences")}, indent=2))
    return 0 if results["stopped"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
