"""Activation capture from vLLM, the hulder[vllm] extra.

Offline extraction through vLLM's hidden-state connector, prefix caching off, chunked
prefill off, fixed prompt order and batch size. Never imported by the base package.
Any script calling `capture` needs an `if __name__ == "__main__":` guard (spawn on WSL).
"""

import os
import tempfile
from dataclasses import dataclass

import numpy as np
import torch
from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector


@dataclass(frozen=True)
class CaptureConfig:
    """Everything that fixes the captured values. Recorded verbatim in the run manifest."""

    model: str
    revision: str
    layers: tuple[int, ...]  # id k = residual stream after k decoder layers, 0 = embeddings
    positions: tuple[int, ...]  # negative indices into the templated prompt
    system_prompt: str
    quantization: str | None = None
    dtype: str = "float16"
    batch_size: int = 8
    # Fits Qwen2.5-1.5B on an 8 GB RTX 2070 with extraction on (measured 2026-10-02).
    gpu_memory_utilization: float = 0.7
    max_model_len: int = 1024
    max_num_batched_tokens: int = 1024
    seed: int = 0
    # Passed to the chat template when set. Pinned for hybrid-thinking models such as Qwen3,
    # whose templated prompt, and so its last token, depends on it. None leaves the template's default.
    enable_thinking: bool | None = None
    # The offline extraction decision: required off for reproducible snapshots. Fields so the manifest records them.
    enable_prefix_caching: bool = False
    enable_chunked_prefill: bool = False

    def __post_init__(self) -> None:
        if self.enable_prefix_caching or self.enable_chunked_prefill:
            raise ValueError("the offline extraction decision: capture needs prefix caching and chunked prefill off")


@dataclass(frozen=True)
class Capture:
    activations: np.ndarray  # (n, layers, positions, d), float16, prompt order preserved
    position_tokens: list[list[str]]  # token string at each requested position, per prompt
    # torch.cuda.get_device_name() of the current device, the first visible one. Under tensor
    # parallelism the other GPUs aren't recorded, a known limit. Another GPU is another serving
    # config.
    gpu: str
    enable_thinking: bool | None  # as passed to the chat template, None for its default


def capture(prompts: list[str], cfg: CaptureConfig) -> Capture:
    convs = [[{"role": "system", "content": cfg.system_prompt}, {"role": "user", "content": p}] for p in prompts]
    params = SamplingParams(max_tokens=1, temperature=0)
    template = {} if cfg.enable_thinking is None else {"chat_template_kwargs": {"enable_thinking": cfg.enable_thinking}}
    acts, tokens = [], []
    with tempfile.TemporaryDirectory() as tmp:
        llm = LLM(
            model=cfg.model,
            revision=cfg.revision,
            quantization=cfg.quantization,
            dtype=cfg.dtype,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_model_len=cfg.max_model_len,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            max_num_seqs=cfg.batch_size,
            seed=cfg.seed,
            enable_chunked_prefill=cfg.enable_chunked_prefill,
            enable_prefix_caching=cfg.enable_prefix_caching,
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": list(cfg.layers)}},
            },
            kv_transfer_config=KVTransferConfig(
                kv_connector="ExampleHiddenStatesConnector",
                kv_role="kv_producer",
                kv_connector_extra_config={"shared_storage_path": tmp},
            ),
        )
        tok = llm.get_tokenizer()
        pos = list(cfg.positions)
        for start in range(0, len(convs), cfg.batch_size):
            for out in llm.chat(convs[start : start + cfg.batch_size], params, use_tqdm=False, **template):
                path = out.kv_transfer_params["hidden_states_path"]
                obj = example_hidden_states_connector.load_hidden_states(path)
                assert obj["token_ids"].tolist() == list(out.prompt_token_ids)
                # hidden_states is [tokens, layers, d]; keep only the requested positions.
                acts.append(obj["hidden_states"][pos].transpose(0, 1).to("cpu").numpy().astype(np.float16))
                tokens.append(tok.convert_ids_to_tokens(obj["token_ids"][pos].tolist()))
                os.remove(path)
            if start // cfg.batch_size % 100 == 0:
                print(f"captured {min(start + cfg.batch_size, len(convs))}/{len(convs)}", flush=True)
    return Capture(
        activations=np.stack(acts),
        position_tokens=tokens,
        gpu=torch.cuda.get_device_name(),
        enable_thinking=cfg.enable_thinking,
    )
