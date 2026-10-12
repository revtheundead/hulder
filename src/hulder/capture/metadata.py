"""Snapshot metadata from a capture. Imports no vLLM or torch, so it runs anywhere."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hulder.capture.hidden_states import Capture, CaptureConfig


def snapshot_metadata(
    cfg: "CaptureConfig",
    captured: "Capture",
    base_model: str,
    vllm_version: str,
    layer: int | None = None,
    position: int | None = None,
) -> dict:
    """The metadata a Snapshot of one layer and position of a capture records: the
    serving config, the layer and position, the GPU, and enable_thinking when it was pinned.
    Only keys comparable across runs. layer and position default to the config's only ones."""
    layer = _one(cfg.layers, layer, "layer")
    position = _one(cfg.positions, position, "position")
    metadata = {
        "base_model": base_model,
        "model": cfg.model,
        "revision": cfg.revision,
        "quantization": cfg.quantization,
        "dtype": cfg.dtype,
        "batch_size": cfg.batch_size,
        "vllm_version": vllm_version,
        "system_prompt": cfg.system_prompt,
        "layer": layer,
        "position": position,
        "gpu": captured.gpu,
    }
    if captured.enable_thinking is not None:
        metadata["enable_thinking"] = captured.enable_thinking
    return metadata


def _one(captured: tuple[int, ...], chosen: int | None, name: str) -> int:
    if chosen is None:
        if len(captured) != 1:
            raise ValueError(f"the capture has {name}s {captured}; name the snapshot's {name}")
        return captured[0]
    if chosen not in captured:
        raise ValueError(f"{name} {chosen} wasn't captured, only {captured}")
    return chosen
