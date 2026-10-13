"""Captured hidden states, stored as float16. Numpy only, apart from as_float32's tensor method call,
so the cast and its range check are tested on the base install.

bf16 has 8 significand bits and float16 11, so every bf16 value in float16's normal range
(6.1e-5 to 65504 in magnitude) converts exactly. Smaller values round to float16's subnormals or
to zero, which is negligible next to a hidden state's typical magnitude. Larger ones would become
inf, so they raise instead, naming where they were.
"""

from collections.abc import Sequence

import numpy as np

FLOAT16_MAX = float(np.finfo(np.float16).max)  # 65504


def as_float32(tensor) -> np.ndarray:
    """A torch tensor of any float dtype as a float32 numpy array. Converted on its device first,
    since numpy has no bfloat16. bf16 and float16 both convert to float32 exactly."""
    return tensor.float().cpu().numpy()


def to_float16(
    hidden: np.ndarray, layers: Sequence[int], positions: Sequence[int], first_prompt: int = 0
) -> np.ndarray:
    """hidden, shaped (prompts, layers, positions, d), as float16. Raises ValueError if the engine
    returned a NaN or infinity, and OverflowError if a value is beyond float16's range, each naming
    the first such prompt (counted from first_prompt), layer and position."""
    hidden = np.asarray(hidden, dtype=np.float32)
    bad = ~np.isfinite(hidden)
    if bad.any():
        raise ValueError(
            f"the engine returned non-finite hidden states, first at {_where(bad, layers, positions, first_prompt)}"
        )
    with np.errstate(over="ignore"):
        converted = hidden.astype(np.float16)
    over = np.isinf(converted)
    if over.any():
        peak = float(np.abs(hidden).max())
        raise OverflowError(
            f"a hidden state of magnitude {peak:.6g} is beyond float16's range (|h| <= 65504), first at "
            f"{_where(over, layers, positions, first_prompt)}. Snapshots store float16, "
            "so this layer and position can't be captured"
        )
    return converted


def _where(mask: np.ndarray, layers: Sequence[int], positions: Sequence[int], first_prompt: int) -> str:
    prompt, layer, position, _ = (int(i) for i in np.argwhere(mask)[0])
    return f"prompt {first_prompt + prompt}, layer {layers[layer]}, position {positions[position]}"
