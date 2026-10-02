# The |h| ≈ 6600 outlier is a massive activation

Follow-up to `fp16_sanity.json`, where layers 3 to 26 all peaked near |h| = 6600. Qwen2.5-1.5B-Instruct @ `989aa798`, fp16, vLLM 0.30.0, RTX 2070. Ten prompts: the five sanity prompts both chat-templated and raw, so position 0 is sometimes `<|im_start|>` and sometimes an ordinary word.

Reproduce: `source scripts/dev-env.sh && PYTHONPATH=experiments/phase1 uv run python experiments/phase1/massive_activations.py`. The output is `massive_activations.json`, which holds the top-4 |h| per layer per prompt.

## What it is

- **Token.** Always position 0, whatever the token (`<|im_start|>`, `What`, `Ignore`, ...). Across layers 3 to 26, no other position reaches any layer's top 4.
- **Dimensions.** Always the same four, 408 > 520 > 940 > 609 in that order, for every prompt.
- **Layers.** It appears at layer 2 (about 2900 to 3600), is roughly constant from layer 3 to 26 (6300 to 7470), and collapses at layer 27 (about 190 to 220).
- **Size.** The top value is 3,000 to 18,500 times the layer's median |h|. The maximum over all prompts is 7468, about 8.8× below the fp16 max. That's a little less headroom than the 9.9× reported in `fp16_sanity.json`, because raw prompts run larger than templated ones.

This matches Sun et al. (2024, arXiv 2402.17762, "Massive Activations in Large Language Models"). Their §2 says massive activations "are consistently present in very few fixed dimensions" and "emerge in the initial layers and start to diminish in the last few layers". Qwen2.5-1.5B falls in their "starting token only" category. One caveat: the paper analyses no Qwen model, so this is our observation for this family, not a result we can cite. The values are also not fully input-independent. The paper says "largely constant", and here they vary by about 10% with the first token.

## Why it matters here

- **Probe input.** The probe reads position −1 or −5, never position 0, so the outlier is not in the probe's input. Any statistic pooled over all positions would be dominated by it, though (for example, an MMD diagnostic over every token).
- **Precision.** At |h| ≈ 7000, one fp16 ULP is 4. That's why the prototype's all-position max |Δh| came out at exactly 4 or 1. Report position-0 differences in ULPs, not as raw magnitudes.
- **fp16 overflow.** About 9× headroom here. A larger model in the same family could have less, which is worth rechecking on the Phase 4 model.
