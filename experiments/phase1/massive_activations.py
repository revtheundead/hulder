"""Locate the |h| ~ 6600 outlier seen in fp16_sanity.json: which tokens, which dimensions.

Tests it against the massive-activations pattern of Sun et al. (2024, arXiv 2402.17762):
a few fixed dimensions, huge values at the first token (and some delimiters), appearing
in early layers and roughly constant through the middle layers.

Run from the repo root:
    source scripts/dev-env.sh && PYTHONPATH=experiments/phase1 uv run python experiments/phase1/massive_activations.py

Writes docs/results/phase1/massive_activations.json.
"""

import json
import tempfile
from pathlib import Path

import torch
import vllm
from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector as hsc

from fp16_sanity import MODEL, NUM_LAYERS, PROMPTS, REVISION

LAYER_IDS = list(range(NUM_LAYERS + 1))
TOP_K = 4
OUT = Path("docs/results/phase1/massive_activations.json")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        llm = LLM(
            model=MODEL,
            revision=REVISION,
            dtype="float16",
            gpu_memory_utilization=0.7,
            max_model_len=1024,
            max_num_batched_tokens=1024,
            max_num_seqs=8,
            seed=0,
            enable_chunked_prefill=False,
            enable_prefix_caching=False,
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": LAYER_IDS}},
            },
            kv_transfer_config=KVTransferConfig(
                kv_connector="ExampleHiddenStatesConnector",
                kv_role="kv_producer",
                kv_connector_extra_config={"shared_storage_path": tmp},
            ),
        )
        # Raw prompts as well as chat-templated ones, so position 0 is not always <|im_start|>.
        chat = [[{"role": "user", "content": p}] for p in PROMPTS]
        outs = llm.chat(chat, SamplingParams(max_tokens=1), use_tqdm=False)
        outs += llm.generate(PROMPTS, SamplingParams(max_tokens=1), use_tqdm=False)
        tok = llm.get_tokenizer()
        hs = [hsc.load_hidden_states(o.kv_transfer_params["hidden_states_path"]) for o in outs]

    prompts = []
    for i, obj in enumerate(hs):
        h = obj["hidden_states"].float().abs()  # [T, layers, d]
        ids = obj["token_ids"].tolist()
        per_layer = []
        for li in range(h.shape[1]):
            flat = h[:, li, :].flatten()
            vals, idx = flat.topk(TOP_K)
            per_layer.append({
                "layer": LAYER_IDS[li],
                "median_abs": round(h[:, li, :].median().item(), 4),
                "top": [
                    {"pos": int(j // h.shape[2]), "token": tok.convert_ids_to_tokens(ids[int(j // h.shape[2])]),
                     "dim": int(j % h.shape[2]), "abs": round(v.item(), 1)}
                    for v, j in zip(vals, idx)
                ],
            })
        prompts.append({"templated": i < len(PROMPTS), "num_tokens": len(ids), "layers": per_layer})

    result = {
        "model": MODEL,
        "revision": REVISION,
        "dtype": "float16",
        "prompts": prompts,
        "manifest": {
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "command": "PYTHONPATH=experiments/phase1 uv run python experiments/phase1/massive_activations.py",
        },
    }
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print("saved", OUT)


if __name__ == "__main__":
    main()
