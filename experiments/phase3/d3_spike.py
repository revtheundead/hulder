"""The D3 spike (docs/design.md §5 D3), on the local GPU: a connector scoring at the end of prefill,
the score in a non-streamed reply and on /metrics, and replies without a score counted.

Qwen2.5-1.5B-Instruct in fp16 under `vllm serve`, zero-draft on Model Runner V2 with async
scheduling on, chunked prefill and CUDA graphs on, prefix caching off, the bundled demo probe, and
the 360 demo prompts. Timing on this GPU isn't representative, so nothing here judges it.

The rule, fixed before the first run:
1. Correct. On a server that also keeps the stock file write, each reply's score is within 1e-6 of
   hulder.apply_probe on that request's saved hidden state, on all 360 prompts at concurrency 8 with
   16-token replies, at the default budget and at max_num_batched_tokens=64, which chunks prefill.
   A gap past the tolerance fails. A reply with no score here needs a decision.
2. Replies unchanged. Without the file write, 32 greedy replies of 64 tokens have the same token ids
   as the stock connector on zero-draft, and draft no tokens.
3. Surfaces. Without the file write, at reply lengths of 1, 2, 3, 4, 8 and 16 tokens and concurrency
   1 and 8, /metrics counts a score for every request, a counted reply for every request, and as many
   without a score as came back without one. Its flagged count lies between the replies' and that
   plus the replies without a score, whose late scores could be flagged. No files are written. 16
   streamed replies count a score each on /metrics, and no chunk carries one.

    source scripts/dev-env.sh && uv run python experiments/phase3/d3_spike.py --out .cache/d3-spike

Results go to <out>/d3_spike.json and server logs to <out>/<server>.log.
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from importlib.metadata import version
from pathlib import Path

import d1_local as d1
import numpy as np

TOLERANCE = 1e-6
PROBE = d1.DATA / "probe_demo.npz"
REPLY_TOKENS = (1, 2, 3, 4, 8, 16)
CONCURRENCIES = (1, 8)
N_STREAMED = 16
SETTLE_SECONDS = 30
COUNTERS = {
    "scores": "hulder_probe_score_count",
    "flagged": "hulder_probe_flagged_total",
    "replies_with_score": "hulder_probe_replies_with_score_total",
    "replies_without_score": "hulder_probe_replies_without_score_total",
}


def compare_scores(served: list, saved: np.ndarray) -> dict:
    """Each reply's score against the probe on its saved hidden state."""
    pairs = [(s, float(v)) for s, v in zip(served, saved) if s is not None]
    return {
        "requests": len(served),
        "compared": len(pairs),
        "without_score": len(served) - len(pairs),
        "max_abs_delta": max((abs(s - v) for s, v in pairs), default=None),
    }


def flips(served: list, saved: np.ndarray, threshold: float) -> int:
    """Replies whose decision differs from the saved hidden state's."""
    return sum((s > threshold) != (float(v) > threshold) for s, v in zip(served, saved) if s is not None)


def probe_metrics(before: str, after: str) -> dict:
    from prometheus_client.parser import text_string_to_metric_families

    def read(text: str) -> dict:
        totals = dict.fromkeys(COUNTERS.values(), 0.0)
        for family in text_string_to_metric_families(text):
            for sample in family.samples:
                if sample.name in totals:
                    totals[sample.name] += sample.value
        return totals

    b, a = read(before), read(after)
    return {key: round(a[name] - b[name]) for key, name in COUNTERS.items()}


def correct_verdict(runs: list[dict]) -> str:
    if any(r["max_abs_delta"] is None or r["max_abs_delta"] > TOLERANCE for r in runs if r["compared"]):
        return "fail"
    return "pass" if all(r["without_score"] == 0 and r["compared"] == r["requests"] for r in runs) else "needs decision"


def surface_ok(run: dict) -> bool:
    metrics = run["metrics"]
    return (
        metrics["scores"] == run["requests"]
        and metrics["replies_with_score"] + metrics["replies_without_score"] == run["requests"]
        and metrics["replies_without_score"] == run["without_score"]
        and run["flagged_in_replies"] <= metrics["flagged"] <= run["flagged_in_replies"] + run["without_score"]
        and run["files_written"] == 0
    )


def verdict(correct: list[dict], generation: dict, surfaces: list[dict], streamed: dict) -> dict:
    result = {
        "correct": correct_verdict(correct),
        "replies_unchanged": "pass"
        if generation["identical"] == generation["prompts"] and generation["draft_tokens"] == 0
        else "fail",
        "surfaces": "pass"
        if all(surface_ok(s) for s in surfaces)
        and streamed["metrics"]["scores"] == streamed["requests"]
        and streamed["chunks_with_a_score"] == 0
        else "fail",
    }
    values = set(result.values())
    result["overall"] = "fail" if "fail" in values else "needs decision" if "needs decision" in values else "pass"
    return result


def scoring_flags(storage: Path, save: bool) -> list[str]:
    """Zero-draft's speculative config with the scoring connector in place of the stock one."""
    extra = {"shared_storage_path": str(storage), "probe_path": str(PROBE), "save_hidden_states": save}
    if not save:
        extra["use_synchronization_lock"] = False
    transfer = {
        "kv_connector": "ScoringHiddenStatesConnector",
        "kv_connector_module_path": "scoring_connector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": extra,
    }
    return [*d1.extraction(storage, zero_draft=True)[:2], "--kv-transfer-config", json.dumps(transfer)]


CONNECTOR_ENV = {
    "PYTHONPATH": os.pathsep.join(filter(None, [str(Path(__file__).parent), os.environ.get("PYTHONPATH")]))
}


def settled_metrics(url: str, before: str, expected_scores: int) -> dict:
    """/metrics since before, once every expected score is in or SETTLE_SECONDS have passed. Stats
    reach the API process with engine steps, a little after the last reply."""
    import httpx

    deadline = time.perf_counter() + SETTLE_SECONDS
    while True:
        metrics = probe_metrics(before, httpx.get(f"{url}/metrics").text)
        if time.perf_counter() > deadline or (
            metrics["scores"] >= expected_scores
            and metrics["replies_with_score"] + metrics["replies_without_score"] >= expected_scores
        ):
            return metrics
        time.sleep(0.5)


def correctness_stage(out: Path, name: str, texts: list[str], probe, extra: list[str] = ()) -> dict:
    import hulder

    storage = out / f"{name}_storage"
    with d1.server(name, out, [*scoring_flags(storage, save=True), *extra], env=CONNECTOR_ENV) as url:
        responses = asyncio.run(d1.chat_all(url, texts, 8, max_tokens=16, return_token_ids=True))
    served = [r["kv_transfer_params"]["hulder_score"] for r in responses]
    acts, lengths = d1.connector_acts(responses)
    shutil.rmtree(storage, ignore_errors=True)
    saved = hulder.apply_probe(probe, acts)
    return {
        **compare_scores(served, saved),
        "prompts_over_64_tokens": sum(n > 64 for n in lengths),
        "flips_at_threshold": flips(served, saved, probe.metadata["threshold"]),
    }


def stream_chunks(url: str, text: str) -> dict:
    import httpx

    body = {"model": d1.MODEL, "temperature": 0, "max_tokens": 16, "stream": True, "messages": d1.messages(text)}
    chunks = with_score = 0
    with httpx.stream("POST", f"{url}/v1/chat/completions", json=body, timeout=600) as response:
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks += 1
                with_score += "hulder_score" in line
    return {"chunks": chunks, "with_score": with_score}


def monitor_stage(out: Path, texts: list[str], threshold: float) -> tuple[list[dict], dict, list[dict], dict]:
    import httpx

    storage = out / "monitor_storage"
    storage.mkdir(exist_ok=True)
    surfaces = []
    with d1.server("monitor", out, scoring_flags(storage, save=False), env=CONNECTOR_ENV) as url:
        before = httpx.get(f"{url}/metrics").text
        generate = asyncio.run(
            d1.chat_all(url, texts[: d1.N_GENERATE], 1, max_tokens=d1.GENERATE_TOKENS, return_token_ids=True)
        )
        decode = d1.step_tokens(before, httpx.get(f"{url}/metrics").text)
        settled_metrics(url, before, d1.N_GENERATE)

        for concurrency in CONCURRENCIES:
            for tokens in REPLY_TOKENS:
                before = httpx.get(f"{url}/metrics").text
                responses = asyncio.run(d1.chat_all(url, texts, concurrency, max_tokens=tokens))
                metrics = settled_metrics(url, before, len(texts))
                scores = [r["kv_transfer_params"]["hulder_score"] for r in responses]
                present = [s for s in scores if s is not None]
                surfaces.append(
                    {
                        "concurrency": concurrency,
                        "reply_tokens": tokens,
                        "requests": len(scores),
                        "without_score": len(scores) - len(present),
                        "flagged_in_replies": sum(s > threshold for s in present),
                        "files_written": len(list(storage.iterdir())),
                        "metrics": metrics,
                    }
                )
                print(surfaces[-1], flush=True)

        before = httpx.get(f"{url}/metrics").text
        streams = [stream_chunks(url, text) for text in texts[:N_STREAMED]]
        streamed = {
            "requests": N_STREAMED,
            "chunks_with_a_score": sum(s["with_score"] for s in streams),
            "metrics": settled_metrics(url, before, N_STREAMED),
        }
    shutil.rmtree(storage, ignore_errors=True)
    return generate, decode, surfaces, streamed


def run(out: Path) -> dict:
    import hulder

    probe = hulder.Probe.load(PROBE)
    texts = [r["text"] for r in d1.prompts()]
    correct = [
        {"budget": "default", **correctness_stage(out, "saved", texts, probe)},
        {
            "budget": 64,
            **correctness_stage(out, "saved_chunked", texts, probe, ["--max-num-batched-tokens", "64"]),
        },
    ]
    generate, decode, surfaces, streamed = monitor_stage(out, texts, float(probe.metadata["threshold"]))

    storage = out / "stock_storage"
    with d1.server("stock_zero_draft", out, d1.extraction(storage, zero_draft=True)) as url:
        stock = asyncio.run(
            d1.chat_all(url, texts[: d1.N_GENERATE], 1, max_tokens=d1.GENERATE_TOKENS, return_token_ids=True)
        )
    d1.connector_acts(stock)  # deletes the files
    shutil.rmtree(storage, ignore_errors=True)
    identical = sum(a["choices"][0]["token_ids"] == b["choices"][0]["token_ids"] for a, b in zip(generate, stock))
    generation = {"prompts": d1.N_GENERATE, "identical": identical, "draft_tokens": decode["draft_tokens"]}
    return {
        "correct": correct,
        "generation": generation,
        "surfaces": surfaces,
        "streamed": streamed,
        "verdict": verdict(correct, generation, surfaces, streamed),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=d1.ROOT / ".cache" / "d3-spike")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{shutil.disk_usage(args.out).free / 1e9:.0f} GB free in {args.out}", flush=True)
    results = run(args.out)
    results["versions"] = {"vllm": version("vllm")}
    (args.out / "d3_spike.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
