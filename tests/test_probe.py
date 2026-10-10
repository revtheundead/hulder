import numpy as np
import pytest

import hulder


def tiny_probe() -> hulder.Probe:
    # Standardise with mean [1, 1] and std [2, 0.5], then logit = 1*z0 - 2*z1 + 0.5.
    return hulder.Probe(
        weights=np.array([1.0, -2.0]),
        bias=0.5,
        mean=np.array([1.0, 1.0]),
        std=np.array([2.0, 0.5]),
    )


def test_apply_probe_matches_worked_example():
    # [3, 1.5] standardises to [1, 1], logit -0.5, sigmoid(-0.5) = 0.37754066879814546.
    # [1, 1] standardises to [0, 0], logit 0.5, sigmoid(0.5) = 0.6224593312018546.
    acts = np.array([[3.0, 1.5], [1.0, 1.0]], dtype=np.float32)
    scores = hulder.apply_probe(tiny_probe(), acts)
    np.testing.assert_allclose(scores, [0.37754066879814546, 0.6224593312018546], rtol=1e-6)


def test_apply_probe_on_fp16_activations_agrees_with_fp32():
    # Real snapshots are fp16 with d=1536 and values up to ~1e2. Accumulating the dot
    # product in fp16 would err by ~1e-2 in the logit; computing in float32 should not.
    rng = np.random.default_rng(0)
    d = 1536
    probe = hulder.Probe(
        weights=rng.normal(0, 0.05, d),
        bias=-0.3,
        mean=rng.normal(0, 1, d),
        std=rng.uniform(0.5, 3, d),
    )
    acts16 = (rng.normal(0, 4, (64, d))).astype(np.float16)
    np.testing.assert_allclose(
        hulder.apply_probe(probe, acts16),
        hulder.apply_probe(probe, acts16.astype(np.float32)),
        atol=1e-6,
    )


def test_apply_probe_rejects_activations_of_the_wrong_width():
    # A layer or model mix-up shows up as a width mismatch; it must not broadcast silently.
    with pytest.raises(ValueError, match="width 3.*probe expects 2"):
        hulder.apply_probe(tiny_probe(), np.zeros((4, 3)))


def test_probe_file_round_trip_keeps_scores_standardisation_and_metadata(tmp_path):
    rng = np.random.default_rng(1)
    d = 16
    probe = hulder.Probe(
        weights=rng.normal(size=d),
        bias=0.25,
        mean=rng.normal(size=d),
        std=rng.uniform(0.5, 2, size=d),
        metadata={
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "layer": 16,
            "position": -1,
            "system_prompt": "You are a helpful assistant.",
        },
    )
    path = tmp_path / "probe.npz"
    probe.save(path)
    loaded = hulder.Probe.load(path)

    acts = rng.normal(size=(8, d)).astype(np.float16)
    np.testing.assert_array_equal(hulder.apply_probe(loaded, acts), hulder.apply_probe(probe, acts))
    np.testing.assert_array_equal(loaded.mean, probe.mean)
    np.testing.assert_array_equal(loaded.std, probe.std)
    assert loaded.bias == probe.bias
    assert loaded.metadata == probe.metadata
    # Shipped probes are downloaded by others, so the file must load without pickle.
    with np.load(path, allow_pickle=False) as f:
        assert set(f.files) >= {"weights", "bias", "mean", "std"}


def test_a_probe_file_without_a_format_version_raises_a_value_error(tmp_path):
    path = tmp_path / "probe.npz"
    one = np.array([1.0])
    np.savez(path, weights=one, bias=np.float64(0), mean=one, std=one, metadata=np.array("{}"))
    with pytest.raises(ValueError, match="no format_version"):
        hulder.Probe.load(path)
