import json

import numpy as np
import pytest

import hulder


def _snapshot(**kwargs):
    rng = np.random.default_rng(0)
    defaults = {
        "ids": np.array(["slabs/3", "slabs/1", "bpairs/apibp_0001_a"]),
        "labels": np.array([0, 1, 0]),
        "activations": rng.normal(size=(3, 4)).astype(np.float16),
        "metadata": {"base_model": "Qwen/Qwen2.5-1.5B-Instruct", "layer": 20, "position": -1, "quantization": None},
    }
    return hulder.Snapshot(**{**defaults, **kwargs})


def test_snapshot_file_round_trip_keeps_ids_labels_activations_and_metadata(tmp_path):
    snapshot = _snapshot()
    snapshot.save(tmp_path / "s.npz")
    loaded = hulder.Snapshot.load(tmp_path / "s.npz")
    assert loaded.ids.tolist() == ["slabs/3", "slabs/1", "bpairs/apibp_0001_a"]
    np.testing.assert_array_equal(loaded.labels, snapshot.labels)
    assert loaded.activations.dtype == np.float16
    np.testing.assert_array_equal(loaded.activations, snapshot.activations)
    assert loaded.scores is None
    assert loaded.metadata == {**snapshot.metadata, "prompt_ids_sha256": hulder.prompt_ids_sha256(snapshot.ids)}


def test_a_snapshot_of_scores_round_trips_too(tmp_path):
    _snapshot(activations=None, scores=np.array([0.1, 0.9, 0.2])).save(tmp_path / "s.npz")
    loaded = hulder.Snapshot.load(tmp_path / "s.npz")
    assert loaded.activations is None
    np.testing.assert_array_equal(loaded.scores, [0.1, 0.9, 0.2])


def test_a_snapshot_holds_scores_or_activations_not_both_or_neither():
    with pytest.raises(ValueError, match="exactly one of scores or activations"):
        _snapshot(scores=np.zeros(3))
    with pytest.raises(ValueError, match="exactly one of scores or activations"):
        _snapshot(activations=None)


def test_a_snapshot_whose_ids_no_longer_match_their_stored_hash_fails_to_load(tmp_path):
    _snapshot().save(tmp_path / "s.npz")
    with np.load(tmp_path / "s.npz", allow_pickle=False) as f:
        arrays = dict(f)
    arrays["ids"] = arrays["ids"][::-1]
    np.savez(tmp_path / "s.npz", **arrays)
    with pytest.raises(ValueError, match="prompt_ids_sha256"):
        hulder.Snapshot.load(tmp_path / "s.npz")


def test_snapshot_metadata_is_plain_json_in_the_file(tmp_path):
    _snapshot().save(tmp_path / "s.npz")
    with np.load(tmp_path / "s.npz", allow_pickle=False) as f:
        assert json.loads(str(f["metadata"]))["format_version"] == hulder.Snapshot.FORMAT_VERSION
