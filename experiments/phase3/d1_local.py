"""D1 local checks (docs/design.md §5 D1), on the local GPU before any remote run.

Qwen2.5-1.5B-Instruct in fp16 under `vllm serve`, with chunked prefill and CUDA graphs on and
prefix caching off, against the bundled demo snapshots and probe (layer 20, position -1):

1. skew: the 360 demo prompts through B (the hidden-state connector), scored against the
   bundled offline reference, with the bundled rerun as the noise floor
2. generate: greedy chat on 32 prompts, B against a server without extraction, and which
   streamed chunk carries B's kv_transfer_params
3. chunked: check 1 with max_num_batched_tokens=64, so most prompts prefill over several steps
4. decode: one 64-token greedy request on each server, tokens per engine step from /metrics
5. lens: vllm-lens in eager mode at layers 19 and 20, and with VLLM_LENS_DISABLE=1 against the
   plain server's greedy outputs from check 2

--repeats runs check 2's generation on two fresh plain servers and two fresh B servers, token by
token with logprobs, so plain against B can be set against plain against plain.
--no-write smoke-tests the L4 run's B no-op arm (no_write_connector.py).
--zero-draft runs B with and without num_speculative_tokens_per_batch_size=[[1, 1, 0]], which
should stop the per-step draft on Model Runner V2, and compares their drafts and saved hidden states.

Checks 1 to 4 run in the project environment, check 5 with vllm-lens layered on:

    source scripts/dev-env.sh && uv run python experiments/phase3/d1_local.py --out .cache/d1-local
    uv run --with vllm-lens==1.3.0 python experiments/phase3/d1_local.py --out .cache/d1-local --lens

Each server is a fresh `vllm serve` process. Results go to <out>/d1_local.json and server logs
to <out>/<server>.log. Hidden-state files are deleted as they're read.
"""

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "examples" / "live_demo" / "demo_prompts.jsonl"
DATA = ROOT / "src" / "hulder" / "data"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SYSTEM_PROMPT = "You are a helpful assistant."
LAYER = 20  # hulder's layer id: the residual stream after 20 decoder layers
LENS_LAYERS = (19, 20)  # vllm-lens's indexing against hulder's is undocumented, so both
VLLM_LENS = "vllm-lens==1.3.0"
PORT = 8765
N_GENERATE = 32
GENERATE_TOKENS = 64
DECODE_PROMPT = "Write a short story about a lighthouse keeper."
# The bundled demo snapshots' engine settings (examples/live_demo), except chunked prefill.
SERVE = [
    "--revision", REVISION,
    "--dtype", "float16",
    "--gpu-memory-utilization", "0.7",
    "--max-model-len", "1024",
    "--max-num-seqs", "8",
    "--enable-chunked-prefill",
    "--no-enable-prefix-caching",
    "--seed", "0",
]  # fmt: skip
COUNTERS = {
    "steps": "vllm:iteration_tokens_total_count",
    "tokens": "vllm:iteration_tokens_total_sum",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}


def shift(reference: np.ndarray, candidate: np.ndarray, threshold: float) -> dict:
    """Decision flips (flagged strictly above the threshold) and |Δscore| between two runs."""
    delta = np.abs(np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    return {
        "flips": int(np.sum((reference > threshold) != (candidate > threshold))),
        "mean_abs_delta_score": float(delta.mean()),
        "max_abs_delta_score": float(delta.max()),
    }


def last_position(hidden: np.ndarray, token_ids, prompt_token_ids) -> np.ndarray:
    """The last prompt token's row of a connector file's hidden_states [tokens, layers, d]."""
    if list(token_ids) != list(prompt_token_ids):
        raise ValueError("the file's token ids aren't the prompt's")
    return hidden[-1, 0]


def step_tokens(before: str, after: str) -> dict:
    """Engine steps, tokens scheduled and spec-decode tokens between two /metrics scrapes. Imports
    prometheus_client, from the vllm extra, only here, so the base install can import this module."""
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


def prompts() -> list[dict]:
    return [json.loads(line) for line in PROMPTS.read_text().splitlines()]


def messages(text: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]


def extraction(storage: Path, zero_draft: bool = False) -> list[str]:
    speculative = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [LAYER]}},
        # Every batch size from 1 up drafts 0 tokens: (1, 1, 0) is carried forward to max_num_seqs.
        **({"num_speculative_tokens_per_batch_size": [[1, 1, 0]]} if zero_draft else {}),
    }
    transfer = {
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"shared_storage_path": str(storage)},
    }
    return ["--speculative-config", json.dumps(speculative), "--kv-transfer-config", json.dumps(transfer)]


@contextlib.contextmanager
def server(name: str, out: Path, extra: list[str] = (), env: dict | None = None):
    """A fresh `vllm serve`, yielding its base URL once /health answers, then stopped."""
    import httpx

    log = (out / f"{name}.log").open("w")
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL, *SERVE, *extra, "--port", str(PORT)]
    log.write(" ".join(command) + "\n")
    log.flush()
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})})
    url = f"http://127.0.0.1:{PORT}"
    started = time.perf_counter()
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"{name} exited with {proc.returncode}; see {log.name}")
            with contextlib.suppress(httpx.HTTPError):
                if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                    break
            if time.perf_counter() - started > 900:
                raise RuntimeError(f"{name} didn't start in 900 s; see {log.name}")
            time.sleep(2)
        print(f"{name}: up in {time.perf_counter() - started:.0f} s", flush=True)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()


async def chat_all(url: str, texts: list[str], concurrency: int, **params) -> list[dict]:
    """Chat completions for texts, at most concurrency in flight, in prompt order."""
    import httpx

    gate = asyncio.Semaphore(concurrency)
    body = {"model": MODEL, "temperature": 0, "seed": 0, **params}
    async with httpx.AsyncClient(base_url=url, timeout=600) as client:

        async def one(text: str) -> dict:
            async with gate:
                response = await client.post("/v1/chat/completions", json={**body, "messages": messages(text)})
                response.raise_for_status()
                return response.json()

        return await asyncio.gather(*(one(t) for t in texts))


def connector_acts(responses: list[dict]) -> tuple[np.ndarray, list[int]]:
    """Each response's layer-20 last-token activation from its connector file, and the prompt
    lengths. Deletes the files. Responses need return_token_ids."""
    from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
        cleanup_hidden_states,
        load_hidden_states,
    )

    acts, lengths = [], []
    for response in responses:
        path = response["kv_transfer_params"]["hidden_states_path"]
        data = load_hidden_states(path)
        hidden = data["hidden_states"].float().numpy()
        acts.append(last_position(hidden, data["token_ids"].tolist(), response["prompt_token_ids"]))
        lengths.append(int(data["token_ids"].shape[0]))
        cleanup_hidden_states(path)
    return np.stack(acts).astype(np.float16), lengths


def against_reference(acts: np.ndarray, probe, reference, rerun) -> dict:
    """hulder check of a server's activations against the bundled reference and rerun."""
    import hulder

    meta = probe.metadata
    report = hulder.check(
        reference.activations,
        acts,
        reference.labels,
        meta["threshold"],
        meta["target_fpr"],
        noise_floor=rerun.activations,
        probe=probe,
    )
    scores = hulder.apply_probe(probe, acts)
    return {
        "verdict": report.verdict.name,
        "decision_flips": report.decision_flips,
        "mean_abs_shift": report.mean_abs_shift,
        "noise_floor_decision_flips": report.noise_floor_decision_flips,
        "noise_floor_mean_abs_shift": report.noise_floor_mean_abs_shift,
        "shift_exceeds_noise_floor": report.shift_exceeds_noise_floor,
        "max_abs_delta_score": shift(hulder.apply_probe(probe, reference.activations), scores, meta["threshold"])[
            "max_abs_delta_score"
        ],
    }


def decode_metrics(url: str) -> dict:
    import httpx

    before = httpx.get(f"{url}/metrics").text
    asyncio.run(
        chat_all(url, [DECODE_PROMPT], 1, max_tokens=GENERATE_TOKENS, min_tokens=GENERATE_TOKENS, ignore_eos=True)
    )
    return step_tokens(before, httpx.get(f"{url}/metrics").text)


def streamed_transfer_chunk(url: str, text: str) -> dict:
    """Which chunk of a streamed reply, if any, carries kv_transfer_params."""
    import httpx

    body = {"model": MODEL, "temperature": 0, "max_tokens": 16, "stream": True, "messages": messages(text)}
    chunks, carrying = 0, []
    with httpx.stream("POST", f"{url}/v1/chat/completions", json=body, timeout=600) as response:
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: ") :])
            if chunk.get("kv_transfer_params"):
                carrying.append(chunks)
            chunks += 1
    return {"chunks": chunks, "chunks_with_kv_transfer_params": carrying}


def texts_match(a: list[dict], b: list[dict]) -> dict:
    left = [r["choices"][0]["message"]["content"] for r in a]
    right = [r["choices"][0]["message"]["content"] for r in b]
    differ = [i for i, (x, y) in enumerate(zip(left, right, strict=True)) if x != y]
    return {"identical": len(left) - len(differ), "of": len(left), "differing_prompts": differ}


def compare_generations(a: list[dict], b: list[dict]) -> dict:
    """Token by token, where each prompt's two greedy replies first differ, with each run's
    logprob margin between its top two candidates there (None past a reply's end). Replies need
    logprobs with top_logprobs >= 2."""

    def steps(response: dict) -> list[dict]:
        return response["choices"][0]["logprobs"]["content"]

    def margin(step: dict | None) -> float | None:
        top = step["top_logprobs"] if step else []
        return top[0]["logprob"] - top[1]["logprob"] if len(top) >= 2 else None

    diverged = []
    for prompt, (x, y) in enumerate(zip(a, b, strict=True)):
        left, right = steps(x), steps(y)
        k = next((i for i, (u, v) in enumerate(zip(left, right)) if u["token"] != v["token"]), None)
        if k is None and len(left) == len(right):
            continue
        k = min(len(left), len(right)) if k is None else k
        pair = [left[k] if k < len(left) else None, right[k] if k < len(right) else None]
        diverged.append(
            {
                "prompt": prompt,
                "token": k,
                "tokens": [step["token"] if step else None for step in pair],
                "margins": [margin(step) for step in pair],
            }
        )
    return {"identical": len(a) - len(diverged), "of": len(a), "diverged": diverged}


def run_main(out: Path) -> dict:
    import hulder

    probe = hulder.Probe.load(DATA / "probe_demo.npz")
    reference = hulder.Snapshot.load(DATA / "demo_fp16.npz")
    rerun = hulder.Snapshot.load(DATA / "demo_fp16_run1.npz")
    rows = prompts()
    if [r["id"] for r in rows] != list(reference.ids):
        raise SystemExit("demo prompts and the bundled reference aren't in the same order")
    texts = [r["text"] for r in rows]
    gen = {"max_tokens": GENERATE_TOKENS}
    results: dict = {}

    with server("plain", out) as url:
        plain_generate = asyncio.run(chat_all(url, texts[:N_GENERATE], 1, **gen))
        results["decode_plain"] = decode_metrics(url)
    (out / "plain_generate.json").write_text(json.dumps(plain_generate))

    storage = out / "hidden_states"
    with server("connector", out, extraction(storage)) as url:
        responses = asyncio.run(chat_all(url, texts, 8, max_tokens=1, return_token_ids=True))
        acts, lengths = connector_acts(responses)
        results["skew"] = against_reference(acts, probe, reference, rerun)
        results["prompt_tokens"] = {"min": min(lengths), "median": int(np.median(lengths)), "max": max(lengths)}
        connector_generate = asyncio.run(chat_all(url, texts[:N_GENERATE], 1, **gen))
        for response in connector_generate:
            cleanup(response)
        results["generate"] = texts_match(plain_generate, connector_generate)
        results["stream"] = streamed_transfer_chunk(url, texts[0])
        results["decode_connector"] = decode_metrics(url)
    np.save(out / "connector_acts.npy", acts)

    with server("connector_chunked64", out, [*extraction(storage), "--max-num-batched-tokens", "64"]) as url:
        responses = asyncio.run(chat_all(url, texts, 8, max_tokens=1, return_token_ids=True))
        chunked, _ = connector_acts(responses)
        results["chunked"] = against_reference(chunked, probe, reference, rerun)
        results["chunked"]["prompts_over_64_tokens"] = sum(n > 64 for n in lengths)
        results["chunked_vs_unchunked"] = shift(
            hulder.apply_probe(probe, acts), hulder.apply_probe(probe, chunked), probe.metadata["threshold"]
        )
    shutil.rmtree(storage, ignore_errors=True)
    return results


def run_repeats(out: Path) -> dict:
    """Greedy generation from two fresh plain servers and two fresh B servers, compared pairwise,
    so a difference between plain and B can be set against each one's own run-to-run variation."""
    texts = [r["text"] for r in prompts()][:N_GENERATE]
    params = {"max_tokens": GENERATE_TOKENS, "logprobs": True, "top_logprobs": 2}
    runs = {}
    storage = out / "hidden_states"
    for name, extra in (("plain", []), ("connector", extraction(storage))):
        for i in (1, 2):
            with server(f"repeat_{name}_{i}", out, extra) as url:
                runs[f"{name}_{i}"] = asyncio.run(chat_all(url, texts, 1, **params))
            if extra:
                for response in runs[f"{name}_{i}"]:
                    cleanup(response)
    shutil.rmtree(storage, ignore_errors=True)
    (out / "repeat_generate.json").write_text(json.dumps(runs))
    pairs = [
        ("plain_1", "plain_2"),
        ("connector_1", "connector_2"),
        ("plain_1", "connector_1"),
        ("plain_2", "connector_2"),
    ]
    return {"repeats": {f"{a} vs {b}": compare_generations(runs[a], runs[b]) for a, b in pairs}}


def run_no_write(out: Path) -> dict:
    """The L4 run's B no-op arm (no_write_connector.py) on the local GPU: it serves, returns no
    kv_transfer_params, writes nothing and still drafts one token per decode step."""
    import httpx

    storage = out / "no_write_storage"
    storage.mkdir(exist_ok=True)
    transfer = {
        "kv_connector": "NoWriteHiddenStatesConnector",
        "kv_connector_module_path": "no_write_connector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"shared_storage_path": str(storage), "use_synchronization_lock": False},
    }
    extra = [*extraction(storage)[:2], "--kv-transfer-config", json.dumps(transfer)]
    env = {"PYTHONPATH": os.pathsep.join(filter(None, [str(Path(__file__).parent), os.environ.get("PYTHONPATH")]))}
    with server("no_write", out, extra, env=env) as url:
        before = httpx.get(f"{url}/metrics").text
        [response] = asyncio.run(chat_all(url, [DECODE_PROMPT], 1, max_tokens=16, min_tokens=16, ignore_eos=True))
        tokens = step_tokens(before, httpx.get(f"{url}/metrics").text)
        # Several more, concurrently: a connector that mishandles finished requests fails on the second.
        more = asyncio.run(chat_all(url, [r["text"] for r in prompts()[:16]], 8, max_tokens=16))
    written = sorted(p.name for p in storage.iterdir())
    shutil.rmtree(storage, ignore_errors=True)
    return {
        "no_write": {
            "kv_transfer_params": response.get("kv_transfer_params"),
            "files_written": written,
            "decode": tokens,
            "more_requests_served": len(more),
        }
    }


def saved_hidden_states(responses: list[dict]) -> list[dict]:
    """Each response's connector file as tensors on the CPU. Deletes the files."""
    from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
        cleanup_hidden_states,
        load_hidden_states,
    )

    saved = []
    for response in responses:
        path = response["kv_transfer_params"]["hidden_states_path"]
        data = load_hidden_states(path)
        saved.append({key: data[key].cpu() for key in ("hidden_states", "token_ids")})
        cleanup_hidden_states(path)
    return saved


def compare_saved(a: list[dict], b: list[dict]) -> dict:
    """How many prompts' saved token ids and hidden states are bit-identical between two servers,
    and the largest absolute difference among the rest."""
    import torch

    same_ids = sum(torch.equal(x["token_ids"], y["token_ids"]) for x, y in zip(a, b, strict=True))
    same_states = [torch.equal(x["hidden_states"], y["hidden_states"]) for x, y in zip(a, b, strict=True)]
    differ = [
        (x["hidden_states"].float() - y["hidden_states"].float()).abs().max().item()
        for x, y, same in zip(a, b, same_states, strict=True)
        if not same and x["hidden_states"].shape == y["hidden_states"].shape
    ]
    return {
        "of": len(a),
        "token_ids_identical": same_ids,
        "hidden_states_identical": sum(same_states),
        "shape": list(a[0]["hidden_states"].shape) if a else None,
        "max_abs_difference": max(differ, default=0.0),
        "last_position_max_abs_difference": max(
            (x["hidden_states"][-1].float() - y["hidden_states"][-1].float()).abs().max().item()
            for x, y in zip(a, b, strict=True)
        ),
    }


def run_zero_draft(out: Path) -> dict:
    """B as on the L4 against B with a zero-token draft schedule. On each server: the 32 generation
    prompts one at a time with 16 tokens (prefill alone in its step, so batch composition can't
    move the saved states), then the same prompts at concurrency 8, with the draft counters over
    each phase from /metrics. A second plain B server is the control for what concurrency alone
    changes between two fresh servers."""
    import httpx

    texts = [r["text"] for r in prompts()][:N_GENERATE]
    params = {"max_tokens": 16, "min_tokens": 16, "ignore_eos": True, "return_token_ids": True}
    storage = out / "hidden_states"
    saved, results = {}, {}
    names = ("b", "b_repeat", "b_zero_draft")
    for name, zero in zip(names, (False, False, True), strict=True):
        with server(f"zero_draft_{name}", out, extraction(storage, zero_draft=zero)) as url:
            arm = {}
            for phase, concurrency in (("sequential", 1), ("concurrent", 8)):
                before = httpx.get(f"{url}/metrics").text
                responses = asyncio.run(chat_all(url, texts, concurrency, **params))
                arm[phase] = step_tokens(before, httpx.get(f"{url}/metrics").text)
                arm[phase]["texts"] = [r["choices"][0]["message"]["content"] for r in responses]
                saved[(name, phase)] = saved_hidden_states(responses)
            results[name] = arm
    shutil.rmtree(storage, ignore_errors=True)
    replies = {(n, phase): results[n][phase].pop("texts") for n in names for phase in ("sequential", "concurrent")}
    comparisons = {}
    for other in ("b_repeat", "b_zero_draft"):
        comparisons[f"b_vs_{other}"] = {
            phase: {
                **compare_saved(saved[("b", phase)], saved[(other, phase)]),
                "replies_identical": sum(
                    x == y for x, y in zip(replies[("b", phase)], replies[(other, phase)], strict=True)
                ),
            }
            for phase in ("sequential", "concurrent")
        }
    return {"zero_draft": {"counters": results, **comparisons}}


def cleanup(response: dict) -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
        cleanup_hidden_states,
        load_hidden_states,
    )

    path = response["kv_transfer_params"]["hidden_states_path"]
    load_hidden_states(path)  # waits for the write, so the file is there to delete
    cleanup_hidden_states(path)


def run_lens(out: Path) -> dict:
    from transformers import AutoTokenizer
    from vllm_lens.client import VLLMLensClient

    import hulder

    probe = hulder.Probe.load(DATA / "probe_demo.npz")
    reference = hulder.Snapshot.load(DATA / "demo_fp16.npz")
    rerun = hulder.Snapshot.load(DATA / "demo_fp16_run1.npz")
    texts = [r["text"] for r in prompts()]
    plain_generate = json.loads((out / "plain_generate.json").read_text())
    results: dict = {}

    with server("lens_disabled", out, env={"VLLM_LENS_DISABLE": "1"}) as url:
        disabled = asyncio.run(chat_all(url, texts[:N_GENERATE], 1, max_tokens=GENERATE_TOKENS))
        results["lens_disabled_generate"] = texts_match(plain_generate, disabled)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    with server("lens", out) as url:
        client = VLLMLensClient(url)
        acts = {layer: [] for layer in LENS_LAYERS}
        for text in texts:
            prompt = tokenizer.apply_chat_template(messages(text), tokenize=False, add_generation_prompt=True)
            output = client.generate(prompt, max_tokens=1, temperature=0, capture_layers=list(LENS_LAYERS))
            stream = np.asarray(output.activations["residual_stream"], dtype=np.float32)  # (layers, positions, d)
            for i, layer in enumerate(LENS_LAYERS):
                acts[layer].append(stream[i, -1])
        results["lens"] = {
            str(layer): against_reference(np.stack(a).astype(np.float16), probe, reference, rerun)
            for layer, a in acts.items()
        }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / ".cache" / "d1-local")
    parser.add_argument("--lens", action="store_true", help="run check 5, under uv run --with " + VLLM_LENS)
    parser.add_argument("--repeats", action="store_true", help="run plain and B twice each and compare generations")
    parser.add_argument("--no-write", action="store_true", help="smoke-test the L4 run's B no-op connector")
    parser.add_argument("--zero-draft", action="store_true", help="B with and without a zero-token draft schedule")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{shutil.disk_usage(args.out).free / 1e9:.0f} GB free in {args.out}", flush=True)
    path = args.out / "d1_local.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    run = (
        run_lens
        if args.lens
        else run_repeats
        if args.repeats
        else run_no_write
        if args.no_write
        else run_zero_draft
        if args.zero_draft
        else run_main
    )
    results.update(run(args.out))
    results["versions"] = {
        "vllm": version("vllm"),
        **({"vllm-lens": version("vllm-lens")} if args.lens else {}),
    }
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
