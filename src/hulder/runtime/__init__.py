"""In-server probe scoring for `vllm serve`, the hulder[vllm] extra (docs/design.md §5 D1, D3).
Experimental: these names may change in any 0.x release, and it runs only on the pinned vLLM.

A KV connector, hulder.runtime.connector, scores a linear probe on each request's last prompt
token at the end of prefill. The score goes into a non-streamed reply's kv_transfer_params as
hulder_score and onto /metrics. It's monitor-only: nothing waits for the score. serve_args gives
the arguments that load it. Importing this package, and serve_args itself, need no vLLM or torch.
"""

import json
from pathlib import Path

from hulder.gate.probe import Probe

__all__ = ["serve_args"]

# The connector reads the hidden state of each request's last prompt token, nothing else.
SCORED_POSITION = -1

# One drafted token per step is the extraction method's minimum, and the per-batch-size schedule
# then takes it to 0 at every batch size from 1 up. Model Runner V2 with async scheduling only.
ZERO_DRAFT = [[1, 1, 0]]


def serve_args(probe_path: str | Path) -> list[str]:
    """`--speculative-config` and `--kv-transfer-config` for `vllm serve`, scoring the probe at
    probe_path. Pass them after the model, with async scheduling left on (the default):

        print(shlex.join(["vllm", "serve", MODEL, *serve_args("probe.npz")]))

    The layer comes from the probe's metadata, and the probe must score position -1 and record a
    threshold. Raises ValueError otherwise.
    """
    path = Path(probe_path).resolve()
    metadata = Probe.load(path).metadata
    for key in ("layer", "position", "threshold"):
        if metadata.get(key) is None:
            raise ValueError(f"{path}: the probe records no {key}, which the runtime needs")
    if metadata["position"] != SCORED_POSITION:
        raise ValueError(
            f"{path}: the probe reads position {metadata['position']}, and the runtime scores only "
            f"the last prompt token, position {SCORED_POSITION}"
        )
    speculative = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "num_speculative_tokens_per_batch_size": ZERO_DRAFT,
        "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [int(metadata["layer"])]}},
    }
    transfer = {
        "kv_connector": "ScoringHiddenStatesConnector",
        "kv_connector_module_path": "hulder.runtime.connector",
        "kv_role": "kv_producer",
        # Nothing is written to disk, so no lock files either.
        "kv_connector_extra_config": {"probe_path": str(path), "use_synchronization_lock": False},
    }
    return ["--speculative-config", json.dumps(speculative), "--kv-transfer-config", json.dumps(transfer)]
