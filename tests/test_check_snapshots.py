import numpy as np
import pytest
from synthetic import FRESH_ENGINE_NOISE, FRESH_ENGINE_P_ZERO, unchanged_scorer

import hulder

_META = {"base_model": "Qwen/Qwen2.5-1.5B-Instruct", "quantization": None, "layer": 20}


def _snapshots(**rerun_meta):
    data = unchanged_scorer(seed=0, noise=FRESH_ENGINE_NOISE, p_zero=FRESH_ENGINE_P_ZERO)
    rerun = unchanged_scorer(seed=0, noise=FRESH_ENGINE_NOISE, p_zero=FRESH_ENGINE_P_ZERO, noise_seed=1).candidate
    ids = np.array([f"slabs/{i}" for i in range(len(data.labels))])

    def snapshot(scores, metadata):
        return hulder.Snapshot(ids=ids, labels=data.labels, scores=scores, metadata=metadata)

    return (
        data,
        snapshot(data.reference, _META),
        snapshot(data.candidate, {**_META, "quantization": "awq"}),
        snapshot(rerun, {**_META, **rerun_meta}),
    )


def test_checking_snapshots_gives_the_same_report_as_checking_their_arrays():
    data, reference, candidate, rerun = _snapshots()
    report = hulder.check_snapshots(reference, candidate, rerun, threshold=data.threshold, target_fpr=0.01)
    expected = hulder.check(
        reference.scores,
        candidate.scores,
        reference.labels,
        threshold=data.threshold,
        target_fpr=0.01,
        noise_floor=rerun.scores,
        reference_meta=reference.metadata,
        candidate_meta=candidate.metadata,
        reference_ids=reference.ids,
        candidate_ids=candidate.ids,
        noise_floor_ids=rerun.ids,
    )
    assert report == expected
    assert report.warnings == ()
    assert report.changes_under_test == (("quantization", None, "awq"),)


@pytest.mark.parametrize("role", ["candidate", "rerun"])
def test_labels_that_differ_from_the_references_raise_a_pairing_error(role):
    # Same ids, so prompt_ids_sha256 agrees, but one label flipped: the snapshots disagree
    # about what the examples are (review S5).
    data, reference, candidate, rerun = _snapshots()
    runs = {"candidate": candidate, "rerun": rerun}
    labels = runs[role].labels.copy()
    labels[0] = 1 - labels[0]
    runs[role] = hulder.Snapshot(ids=runs[role].ids, labels=labels, scores=runs[role].scores, metadata=_META)
    with pytest.raises(hulder.PairingError, match=f"{role}'s labels differ from the reference's at 1 of 2101 examples"):
        hulder.check_snapshots(reference, **runs, threshold=data.threshold, target_fpr=0.01)


def test_a_rerun_whose_metadata_differs_from_the_reference_warns():
    # The rerun is the reference's serving config run again, so every key should match.
    data, reference, candidate, rerun = _snapshots(quantization="awq")
    report = hulder.check_snapshots(reference, candidate, rerun, threshold=data.threshold, target_fpr=0.01)
    [warning] = report.warnings
    assert "rerun's metadata 'quantization' differs" in warning.lower()


def test_snapshots_of_activations_need_the_probe_and_snapshots_of_scores_refuse_it():
    data, reference, candidate, _ = _snapshots()
    probe = hulder.Probe(weights=np.array([1.0]), bias=0.0, mean=np.array([0.0]), std=np.array([1.0]))
    with pytest.raises(ValueError, match="hold scores, so pass no probe"):
        hulder.check_snapshots(reference, candidate, threshold=data.threshold, target_fpr=0.01, probe=probe)
    as_activations = hulder.Snapshot(
        ids=candidate.ids, labels=candidate.labels, activations=candidate.scores[:, None], metadata=_META
    )
    with pytest.raises(ValueError, match="candidate holds activations"):
        hulder.check_snapshots(reference, as_activations, threshold=data.threshold, target_fpr=0.01)


def test_snapshots_of_activations_are_scored_by_the_probe():
    data, reference, candidate, _ = _snapshots()
    probe = hulder.Probe(weights=np.array([1.0]), bias=0.0, mean=np.array([0.0]), std=np.array([1.0]))

    def activations(snapshot):
        logits = np.log(snapshot.scores / (1 - snapshot.scores)).astype(np.float32)[:, None]
        return hulder.Snapshot(ids=snapshot.ids, labels=snapshot.labels, activations=logits, metadata=snapshot.metadata)

    report = hulder.check_snapshots(
        activations(reference), activations(candidate), threshold=data.threshold, target_fpr=0.01, probe=probe
    )
    assert report.mean_abs_logit_shift is not None
    assert report.verdict is hulder.Verdict.PASS


_PROBE_META = {"model": "Qwen/Qwen2.5-1.5B-Instruct", "layer": 20, "position": -1, "system_prompt": "s"}


def _activation_snapshots(probe_meta, **snapshot_meta):
    """Reference and candidate as one-feature activations, scored by an identity probe."""
    data, reference, candidate, _ = _snapshots()
    probe = hulder.Probe(
        weights=np.array([1.0]), bias=0.0, mean=np.array([0.0]), std=np.array([1.0]), metadata=probe_meta
    )

    def activations(snapshot):
        logits = np.log(snapshot.scores / (1 - snapshot.scores)).astype(np.float32)[:, None]
        metadata = {**snapshot.metadata, "position": -1, "system_prompt": "s", **snapshot_meta}
        return hulder.Snapshot(ids=snapshot.ids, labels=snapshot.labels, activations=logits, metadata=metadata)

    return data, probe, activations(reference), activations(candidate)


def test_a_probe_whose_metadata_matches_the_snapshots_gives_no_warning():
    data, probe, reference, candidate = _activation_snapshots(_PROBE_META)
    report = hulder.check_snapshots(reference, candidate, threshold=data.threshold, target_fpr=0.01, probe=probe)
    assert report.warnings == ()


@pytest.mark.parametrize(("key", "value"), [("layer", 10), ("position", 0), ("base_model", "Qwen/Qwen2.5-7B-Instruct")])
def test_a_probe_for_another_layer_position_or_model_raises_a_pairing_error(key, value):
    # A probe of the same width applied to the wrong layer scores nonsense without failing.
    data, probe, reference, candidate = _activation_snapshots({**_PROBE_META, key: value})
    with pytest.raises(hulder.PairingError, match=f"probe's {key}"):
        hulder.check_snapshots(reference, candidate, threshold=data.threshold, target_fpr=0.01, probe=probe)


def test_a_probe_that_records_no_layer_warns_that_it_was_not_confirmed():
    data, probe, reference, candidate = _activation_snapshots({k: v for k, v in _PROBE_META.items() if k != "layer"})
    report = hulder.check_snapshots(reference, candidate, threshold=data.threshold, target_fpr=0.01, probe=probe)
    [warning] = report.warnings
    assert "couldn't confirm the probe's layer" in warning


def test_a_probe_whose_other_metadata_differs_from_the_reference_warns():
    data, probe, reference, candidate = _activation_snapshots({**_PROBE_META, "system_prompt": "other"})
    report = hulder.check_snapshots(reference, candidate, threshold=data.threshold, target_fpr=0.01, probe=probe)
    [warning] = report.warnings
    assert "probe's metadata 'system_prompt' differs" in warning.lower()


@pytest.mark.parametrize("role", ["candidate", "rerun"])
def test_a_snapshot_whose_recorded_ids_hash_contradicts_its_ids_raises_a_pairing_error(role):
    # Snapshot.load catches this for files; a Snapshot built in memory must be caught too,
    # for the rerun as for the candidate.
    data, reference, candidate, rerun = _snapshots()
    runs = {"candidate": candidate, "rerun": rerun}
    other_order = hulder.prompt_ids_sha256(runs[role].ids[::-1])
    with pytest.raises(hulder.PairingError, match="ids don't match the prompt_ids_sha256"):
        runs[role] = hulder.Snapshot(
            ids=runs[role].ids,
            labels=runs[role].labels,
            scores=runs[role].scores,
            metadata={**runs[role].metadata, "prompt_ids_sha256": other_order},
        )
        hulder.check_snapshots(reference, **runs, threshold=data.threshold, target_fpr=0.01)


def test_a_candidate_captured_on_another_gpu_warns():
    # Another GPU is another serving config, so it mixes into the change under test (metadata rule 2).
    data, reference, candidate, _ = _snapshots()
    on_4070 = hulder.Snapshot(
        ids=reference.ids,
        labels=reference.labels,
        scores=reference.scores,
        metadata={**_META, "gpu": "NVIDIA GeForce RTX 4070"},
    )
    on_5070 = hulder.Snapshot(
        ids=candidate.ids,
        labels=candidate.labels,
        scores=candidate.scores,
        metadata={**_META, "quantization": "awq", "gpu": "NVIDIA GeForce RTX 5070"},
    )
    report = hulder.check_snapshots(on_4070, on_5070, threshold=data.threshold, target_fpr=0.01)
    [warning] = report.warnings
    assert "'gpu'" in warning and "RTX 5070" in warning
