import json

import numpy as np
import pytest

import hulder
from hulder.gate.snapshot import _content_sha256


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
    # Consistent arrays, so only the ids' hash in the metadata disagrees.
    arrays["ids"] = arrays["ids"][::-1]
    content = {k: arrays[k] for k in ("ids", "labels", "activations")}
    arrays["content_sha256"] = np.array(_content_sha256(content))
    np.savez(tmp_path / "s.npz", **arrays)
    with pytest.raises(ValueError, match="prompt_ids_sha256"):
        hulder.Snapshot.load(tmp_path / "s.npz")


def test_snapshot_metadata_is_plain_json_in_the_file(tmp_path):
    _snapshot().save(tmp_path / "s.npz")
    with np.load(tmp_path / "s.npz", allow_pickle=False) as f:
        assert json.loads(str(f["metadata"]))["format_version"] == hulder.Snapshot.FORMAT_VERSION


@pytest.mark.parametrize("entry", ["activations", "labels", "ids"])
def test_a_snapshot_whose_arrays_were_altered_fails_to_load(tmp_path, entry):
    # One flipped value, as a bad disk block would leave it.
    _snapshot().save(tmp_path / "s.npz")
    with np.load(tmp_path / "s.npz", allow_pickle=False) as f:
        arrays = dict(f)
    altered = arrays[entry].copy()
    altered[0] = altered[1]
    arrays[entry] = altered
    np.savez(tmp_path / "s.npz", **arrays)
    with pytest.raises(ValueError, match="content_sha256"):
        hulder.Snapshot.load(tmp_path / "s.npz")


def test_the_content_hash_is_not_part_of_the_metadata_check_compares(tmp_path):
    _snapshot().save(tmp_path / "s.npz")
    assert "content_sha256" not in hulder.Snapshot.load(tmp_path / "s.npz").metadata
