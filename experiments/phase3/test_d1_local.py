"""The D1 local checks' pure parts: the shift between two runs' scores, the last prompt
position, and the per-step token counts from two /metrics scrapes. CPU only."""

import numpy as np
import pytest

import d1_local as d


def test_shift_counts_flips_strictly_above_the_threshold_and_mean_and_max_abs_delta():
    reference = np.array([0.1, 0.5, 0.39, 0.9])
    candidate = np.array([0.1, 0.3, 0.41, 0.9])
    assert d.shift(reference, candidate, threshold=0.4) == {
        "flips": 2,
        "mean_abs_delta_score": pytest.approx(0.055),
        "max_abs_delta_score": pytest.approx(0.2),
    }


def test_last_position_takes_the_final_prompt_token_and_checks_the_tokens():
    hidden = np.arange(3 * 1 * 2).reshape(3, 1, 2)  # [tokens, layers, d]
    assert d.last_position(hidden, token_ids=[5, 6, 7], prompt_token_ids=[5, 6, 7]).tolist() == [4, 5]
    with pytest.raises(ValueError, match="token ids"):
        d.last_position(hidden, token_ids=[5, 6, 7], prompt_token_ids=[5, 6, 8])


METRICS = """\
# TYPE vllm:iteration_tokens_total histogram
vllm:iteration_tokens_total_sum{{engine="0",model_name="m"}} {tokens}
vllm:iteration_tokens_total_count{{engine="0",model_name="m"}} {steps}
# TYPE vllm:spec_decode_num_draft_tokens counter
vllm:spec_decode_num_draft_tokens_total{{engine="0",model_name="m"}} {drafts}
# TYPE vllm:spec_decode_num_accepted_tokens counter
vllm:spec_decode_num_accepted_tokens_total{{engine="0",model_name="m"}} {accepted}
"""


def test_step_tokens_is_the_difference_between_two_scrapes():
    before = METRICS.format(tokens=100, steps=10, drafts=0, accepted=0)
    after = METRICS.format(tokens=100 + 30 + 2 * 63, steps=10 + 64, drafts=63, accepted=0)
    assert d.step_tokens(before, after) == {
        "steps": 64,
        "tokens": 156,
        "draft_tokens": 63,
        "accepted_tokens": 0,
    }


def test_step_tokens_reads_missing_spec_decode_counters_as_zero():
    plain = "\n".join(METRICS.splitlines()[:3]) + "\n"
    assert d.step_tokens(plain.format(tokens=0, steps=0), plain.format(tokens=93, steps=64)) == {
        "steps": 64,
        "tokens": 93,
        "draft_tokens": 0,
        "accepted_tokens": 0,
    }
