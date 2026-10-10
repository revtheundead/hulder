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
