# Extraction prototype (Phase 1, step 4)

Evidence for the offline extraction decision. vLLM 0.30.0, Qwen2.5-1.5B-Instruct @ `989aa798`, fp16, RTX 2070, Triton attention, piecewise CUDA graphs, chunked prefill off. Layers 8, 16, 24. Six chat prompts sharing a 61-token system prompt.

Reproduce: the four commands in the docstring of `experiments/phase1/extract_scenario.py`.

## Answers

| Question | Answer | Evidence |
|---|---|---|
| Works at 0.30.0 for an arbitrary layer? | Yes. Layer id k is the residual stream after k decoder layers (0 = embeddings, 28 = final, pre-norm) | `fp16_sanity.json` extracts all 29 ids; `vllm/model_executor/models/qwen2.py:423-433`, `interfaces.py:1608-1610` |
| Still writes to disk? | Yes, one safetensors file per request (`hidden_states [tokens, layers, d]`, `token_ids`), written by a thread pool | `example_hidden_states_connector.py:347-410` |
| Which positions? | Every prompt token. End of user turn (`<|im_end|>`) is index −5 on Qwen's template, the last templated token is −1 | `compare_scenarios.py` prints the template tail |
| Deterministic across reruns? | Bit-identical, in the same engine and in a fresh one, when batch composition is fixed | table below |
| Does prefix caching change what's computed? | Yes, see below | table below |

## Differences in hidden states (max \|Δh\| over the 3 layers)

| Comparison | Last token | End of user turn | Prompts differing |
|---|---|---|---|
| Rerun, same engine, caching off | 0 | 0 | 0/6 |
| Fresh engine, caching off | 0 | 0 | 0/6 |
| Batched vs one prompt per call, caching off | 0.0625 | 0.031 | 5/6 |
| Caching on vs off | 0.0625 | 0.0625 | 6/6 |

Relative L2 at the last token for batched vs single is about 1.2e-3. Last-token cosine between caching on and off is ≥ 0.999999. Whether either moves a probe score past the noise floor is unknown until a probe exists.

## Prefix caching, in detail

On a cache hit, every full 16-token block is served from the stored hidden-state cache, bit-identical to whatever request first computed it. Only the trailing partial block is recomputed. In this run, positions 0 to 63 were reused and 64 onward recomputed. The recomputed tokens, including the last token, differ from a cold prefill because attention reads the prefix from the KV cache. So with caching on, a request's hidden states depend on the request history.

Unexplained: with caching on, the first pass already differs from caching off at position 0, before any hit could occur. Not investigated.

## Implications (proposed, for the offline extraction decision)

- Offline extraction for the gate runs with prefix caching **off** and a **fixed batch composition** (fixed prompt order and batch size). That makes snapshots bit-reproducible.
- Batch composition is a measured source of variation, so the noise floor should vary it on purpose rather than hold it fixed by accident.
- Prefix caching on/off is a serving config in its own right, which matches README §6 D2.
- Chunked prefill must be off (connector docs). Production servers usually run with it on, so the gate's extraction settings are not the production settings. That gap is D1's problem (Phase 3), but the README should say so.
