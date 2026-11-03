"""The scoring connector's GPU arithmetic and its stats, on CPU. The rest of it is vLLM's
interface, exercised by d3_spike.py on a real server. Needs the vllm extra."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

import scoring_connector as c

import hulder


def test_the_score_of_a_worked_example():
    # z = ((3 - 1) / 2, (0 - 0) / 1) = (1, 0), so the logit is 2 * 1 + 5 * 0 - 2 = 0 and the score 0.5.
    weights = torch.tensor([2.0, 5.0])
    mean, std = torch.tensor([1.0, 0.0]), torch.tensor([2.0, 1.0])
    score = c.probe_score(torch.tensor([[3.0, 0.0]], dtype=torch.float16), mean, std, weights, -2.0)
    assert score.tolist() == [0.5]


def test_the_scores_match_hulders_apply_probe_on_the_bundled_probe_and_activations():
    from importlib.resources import files

    data = files("hulder") / "data"
    probe = hulder.Probe.load(data / "probe_demo.npz")
    acts = hulder.Snapshot.load(data / "demo_fp16.npz").activations[:64]
    scores = c.probe_score(
        torch.from_numpy(acts),
        torch.from_numpy(probe.mean.astype(np.float32)),
        torch.from_numpy(probe.std.astype(np.float32)),
        torch.from_numpy(probe.weights.astype(np.float32)),
        float(probe.bias),
    )
    assert scores.numpy() == pytest.approx(hulder.apply_probe(probe, acts), abs=1e-6)


def test_worker_meta_from_several_workers_merges():
    merged = c.ScoresWorkerMeta({"a": 0.1}).aggregate(c.ScoresWorkerMeta({"b": 0.2}))
    assert merged.scores == {"a": 0.1, "b": 0.2}


def test_stats_aggregate_over_an_interval_and_reduce_for_the_log():
    a = c.ScoreStats({"scores": [0.25], "flagged": 0, "replies_with_score": 1, "replies_without_score": 0})
    b = c.ScoreStats({"scores": [0.75], "flagged": 1, "replies_with_score": 0, "replies_without_score": 2})
    both = a.aggregate(b)
    assert both.reduce() == {
        "scores": 2,
        "mean_score": 0.5,
        "flagged": 1,
        "replies_with_score": 1,
        "replies_without_score": 2,
    }
    assert not both.is_empty()
    assert c.ScoreStats({"scores": [], "flagged": 0, "replies_with_score": 0, "replies_without_score": 0}).is_empty()
