"""Captured hidden states are stored as float16 (docs/design.md §5). A bf16 engine's values convert
exactly within float16's normal range, and a value beyond float16's range fails loudly, naming
where it was. Numpy only, so CI runs it; the torch step is tested where torch is installed."""

import numpy as np
import pytest

from hulder.capture.precision import FLOAT16_MAX, to_float16

LAYERS, POSITIONS = (18, 27), (-5, -1)


def bf16_values(n: int, seed: int = 0) -> np.ndarray:
    """Random float32 values that bf16 can represent exactly: the low 16 bits of the float32 are zero."""
    bits = np.random.default_rng(seed).integers(0, 2**32, size=n, dtype=np.uint64).astype(np.uint32)
    values = (bits & np.uint32(0xFFFF0000)).view(np.float32)
    return values[np.isfinite(values)]


def test_bf16_values_in_float16s_normal_range_convert_exactly():
    values = bf16_values(200_000)
    normal = values[(np.abs(values) >= np.finfo(np.float16).tiny) & (np.abs(values) <= FLOAT16_MAX)]
    assert len(normal) > 1_000
    hidden = normal[: len(normal) // 4 * 4].reshape(-1, 2, 2, 1).repeat(3, axis=-1)  # (prompts, layers, positions, d)
    converted = to_float16(hidden, LAYERS, POSITIONS)
    assert converted.dtype == np.float16
    assert np.array_equal(converted.astype(np.float32), hidden)


def test_values_below_the_normal_range_round_or_underflow_but_stay_finite():
    tiny = np.array([1e-6, 1e-8, -1e-9], np.float32).reshape(1, 1, 1, 3)
    converted = to_float16(tiny, LAYERS[:1], POSITIONS[:1])
    assert np.all(np.isfinite(converted))
    assert converted[0, 0, 0, 1] == 0 and converted[0, 0, 0, 2] == 0


@pytest.mark.parametrize("value", [65_536.0, -1e6, 3e38])
def test_a_value_beyond_float16s_range_raises_naming_its_prompt_layer_and_position(value):
    hidden = np.ones((3, 2, 2, 4), np.float32)
    hidden[1, 1, 0, 2] = value
    with pytest.raises(OverflowError) as raised:
        to_float16(hidden, LAYERS, POSITIONS)
    message = str(raised.value)
    assert "prompt 1" in message and "layer 27" in message and "position -5" in message
    assert "65504" in message


def test_the_largest_float16_value_is_kept():
    hidden = np.full((1, 1, 1, 2), FLOAT16_MAX, np.float32)
    assert np.all(to_float16(hidden, LAYERS[:1], POSITIONS[:1]) == FLOAT16_MAX)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_a_non_finite_value_from_the_engine_raises_as_such(value):
    hidden = np.zeros((1, 2, 2, 4), np.float32)
    hidden[0, 0, 1, 3] = value
    with pytest.raises(ValueError, match="non-finite") as raised:
        to_float16(hidden, LAYERS, POSITIONS)
    assert "layer 18" in str(raised.value) and "position -1" in str(raised.value)


def test_bf16_tensors_reach_numpy_as_exact_float32():
    torch = pytest.importorskip("torch")
    from hulder.capture.precision import as_float32

    tensor = torch.tensor([1.5, -3.140625, 65280.0, 1e-3], dtype=torch.bfloat16)
    array = as_float32(tensor)
    assert array.dtype == np.float32
    assert np.array_equal(array, tensor.to(torch.float64).numpy().astype(np.float32))


def test_the_named_prompt_counts_from_first_prompt():
    hidden = np.ones((1, 2, 2, 4), np.float32)
    hidden[0, 0, 0, 0] = 1e6
    with pytest.raises(OverflowError, match="prompt 40, layer 18, position -5"):
        to_float16(hidden, LAYERS, POSITIONS, first_prompt=40)
