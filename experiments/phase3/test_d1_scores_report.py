"""The confirmation run's score report on synthetic stages: the outcome rule fixed before the run, the
pairing with the reference, the real hulder check and what stops the report. CPU only."""

import json

import numpy as np
import pytest

import hulder

pytest.importorskip("sklearn")  # exp2_report's train_qwen3
import d1_scores_report as r

FLOOR = {"flips": 4, "mean_abs_dscore": 0.00164}


def stage(flips, mean):
    return {"flips": flips, "mean_abs_dscore": mean}


STOCK_STAGES = [stage(5, 0.002), stage(7, 0.003), stage(6, 0.0025)]  # highest mean 0.003, most flips 7


def test_a_median_and_flips_at_b_stocks_highest_pass():
    zero = [stage(7, 0.001), stage(7, 0.003), stage(0, 0.009)]  # median 0.003, one stage far above
    assert r.score_outcome(zero, STOCK_STAGES, FLOOR)["outcome"] == "pass"


def test_a_median_above_b_stocks_highest_with_a_stage_below_it_needs_a_decision():
    zero = [stage(0, 0.002), stage(0, 0.0031), stage(0, 0.004)]
    assert r.score_outcome(zero, STOCK_STAGES, FLOOR)["outcome"] == "needs decision"


def test_one_stage_with_more_flips_than_b_stocks_maximum_needs_a_decision_however_low_its_mean():
    zero = [stage(0, 0.001), stage(8, 0.001), stage(0, 0.001)]
    assert r.score_outcome(zero, STOCK_STAGES, FLOOR)["outcome"] == "needs decision"


def test_a_lowest_stage_above_b_stocks_highest_fails():
    zero = [stage(0, 0.0031), stage(0, 0.004), stage(0, 0.005)]
    assert r.score_outcome(zero, STOCK_STAGES, FLOOR)["outcome"] == "fail"


def test_a_lowest_stage_exactly_at_b_stocks_highest_doesnt_fail():
    zero = [stage(0, 0.003), stage(0, 0.004), stage(0, 0.005)]
    assert r.score_outcome(zero, STOCK_STAGES, FLOOR)["outcome"] == "needs decision"


def test_the_rule_is_applied_with_each_arm_in_its_own_place():
    low, high = [stage(0, 0.001)] * 3, [stage(0, 0.01)] * 3
    assert r.score_outcome(low, high, FLOOR)["outcome"] == "pass"
    assert r.score_outcome(high, low, FLOOR)["outcome"] == "fail"


def test_a_pass_still_shows_each_stage_against_the_floor_since_the_floor_no_longer_decides():
    # Both arms above the floor can pass, so the report must not read as matching the reference.
    zero = [stage(5, 0.002), stage(0, 0.001), stage(6, 0.0025)]
    out = r.score_outcome(zero, STOCK_STAGES, FLOOR)
    assert out["outcome"] == "pass"
    assert out["zero_draft_stages_above_floor"] == 2
    assert out["b_stock_stages_above_floor"] == 3
    assert out["floor"] == FLOOR


def test_the_outcome_reports_what_it_judged():
    out = r.score_outcome([stage(1, 0.001), stage(2, 0.002), stage(3, 0.004)], STOCK_STAGES, FLOOR)
    assert out["zero_draft_median_mean_abs_dscore"] == 0.002
    assert out["zero_draft_lowest_mean_abs_dscore"] == 0.001
    assert out["zero_draft_most_flips"] == 3
    assert out["b_stock_highest_mean_abs_dscore"] == 0.003
    assert out["b_stock_most_flips"] == 7


def test_the_fixed_floor_is_experiment_2s_shuffled_rerun():
    exp2 = json.loads(r.EXP2.read_text())
    floor = r.floor_from(exp2)
    assert floor["flips"] == 4
    assert floor["mean_abs_dscore"] == pytest.approx(0.00164, abs=5e-6)


D, N = 4, 40
IDS = [f"p{i}" for i in range(N)]
TOKENS = {i: [k, 7] for k, i in enumerate(IDS)}


def snapshot(acts, order=IDS, **meta):
    labels = np.array([int(i[1:]) % 2 for i in order])
    return hulder.Snapshot(
        ids=np.array(order),
        labels=labels,
        activations=np.asarray(acts, dtype=np.float16),
        metadata={
            "layer": 18,
            "position": -1,
            "base_model": "m",
            "prompt_ids_sha256": hulder.prompt_ids_sha256(list(order)),
            **meta,
        },
    )


def activations(seed=0):
    return np.random.default_rng(seed).normal(size=(N, D)).astype(np.float16)


@pytest.fixture
def probe_path(tmp_path):
    path = tmp_path / "probe.npz"
    hulder.Probe(
        weights=np.ones(D),
        bias=0.0,
        mean=np.zeros(D),
        std=np.ones(D),
        metadata={"layer": 18, "position": -1, "base_model": "m", "threshold": 0.5, "target_fpr": 0.05},
    ).save(path)
    return path


def write_stage(run_dir, rnd, arm, acts, ids=IDS):
    folder = run_dir / f"r{rnd}_{arm}"
    folder.mkdir(parents=True)
    np.savez(folder / "score.npz", ids=np.array(ids), activations=np.asarray(acts, dtype=np.float16))


def finished_run(tmp_path, rounds=(1,), stopped=None):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runs = []
    for rnd in rounds:
        for arm in ("b_stock", "b_zero_draft"):
            write_stage(run_dir, rnd, arm, activations() + 0.01)
            runs.append({"round": rnd, "arm": arm, "score": {"rows": N, "concurrency": 64}})
    (run_dir / "d1_l4.json").write_text(json.dumps({"stopped": stopped, "runs": runs}))
    (run_dir / "manifest.json").write_text(
        json.dumps({"measure": {"nvidia_smi": "NVIDIA L4, GPU-abc, 610.57.04, 23034 MiB"}})
    )
    return run_dir


def test_candidate_rows_are_paired_with_the_reference_by_prompt_id():
    # score.npz is in the workload's order, the reference in kept/'s.
    reference = snapshot(activations(), order=IDS[::-1])
    rows = activations(1)
    candidate = r.candidate_snapshot(reference, IDS, rows, {"arm": "b_zero_draft"})
    assert candidate.ids.tolist() == IDS[::-1]
    assert candidate.labels.tolist() == reference.labels.tolist()
    assert candidate.activations[0].tolist() == rows[N - 1].tolist()
    assert candidate.metadata["arm"] == "b_zero_draft"
    assert candidate.metadata["prompt_ids_sha256"] == reference.metadata["prompt_ids_sha256"]


def test_a_stage_missing_a_reference_prompt_is_an_error_not_a_shorter_comparison():
    with pytest.raises(ValueError, match="1 of the reference's prompts"):
        r.candidate_snapshot(snapshot(activations()), IDS[:-1], activations(1)[:-1], {})


def test_the_report_scores_every_stage_with_the_real_check_and_applies_the_rule(tmp_path, probe_path):
    run_dir = finished_run(tmp_path, rounds=(1, 2))
    reference, rerun = snapshot(activations()), snapshot(activations() + 0.001)
    out = r.report(run_dir, reference, rerun, probe_path, FLOOR)
    assert out["stopped_at"] is None
    assert sorted(out["stages"]) == ["r1_b_stock", "r1_b_zero_draft", "r2_b_stock", "r2_b_zero_draft"]
    stage_report = out["stages"]["r2_b_zero_draft"]
    assert stage_report["shift"]["n"] == N
    assert stage_report["shift"]["mean_abs_dscore"] > 0
    assert set(stage_report["shift"]["flips_by_class"]) == {"attack", "benign"}
    assert stage_report["check"]["exit_code"] in (0, 10, 11, 12)
    assert stage_report["check"]["json"]["verdict"] in ("PASS", "THRESHOLD STALE", "SCORER DEGRADED", "INCONCLUSIVE")
    assert out["outcome"]["outcome"] in ("pass", "needs decision", "fail")


def test_the_candidate_records_the_container_gpu_and_the_served_arm(tmp_path, probe_path):
    run_dir = finished_run(tmp_path)
    captured = {}

    def fake_check(probe, reference, rerun, candidate):
        captured["metadata"] = hulder.Snapshot.load(candidate).metadata
        return {"exit_code": 0}

    out = r.report(
        run_dir, snapshot(activations()), snapshot(activations() + 0.001), probe_path, FLOOR, run_check=fake_check
    )
    assert out["stopped_at"] is None
    assert captured["metadata"]["gpu_uuid"] == "GPU-abc"
    assert captured["metadata"]["arm"] == "b_zero_draft"
    assert captured["metadata"]["batch_size"] == 64


def test_a_stopped_run_stops_the_report(tmp_path, probe_path):
    run_dir = finished_run(tmp_path, stopped="round 2 b_stock: 3 failed requests")
    out = r.report(run_dir, snapshot(activations()), snapshot(activations() + 0.001), probe_path, FLOOR)
    assert out["stopped_at"] == "run"
    assert "3 failed requests" in out["stops"][0]


def test_a_missing_stage_stops_the_report(tmp_path, probe_path):
    run_dir = finished_run(tmp_path)
    (run_dir / "r1_b_stock" / "score.npz").unlink()
    out = r.report(run_dir, snapshot(activations()), snapshot(activations() + 0.001), probe_path, FLOOR)
    assert out["stopped_at"] == "run"
    assert "r1_b_stock" in out["stops"][0]


def test_the_report_refuses_a_second_run_on_the_same_capture(tmp_path):
    (tmp_path / r.REPORT_FILE).write_text("{}")
    with pytest.raises(SystemExit, match="runs once"):
        r.refuse_to_rerun(tmp_path)


def test_a_rerun_that_doesnt_reproduce_the_recorded_floor_stops_the_report(tmp_path, probe_path):
    run_dir = finished_run(tmp_path)
    reference, rerun = snapshot(activations()), snapshot(activations() + 0.001)
    out = r.report(run_dir, reference, rerun, probe_path, FLOOR, recorded_floor={"flips": 99, "mean_abs_dscore": 9.0})
    assert out["stopped_at"] == "floor"
    assert "doesn't reproduce" in out["stops"][0]


def test_the_report_judges_zero_draft_against_b_stock_and_not_the_other_way_round(tmp_path, probe_path):
    # Zero-draft moves scores far more than b_stock does, so this fails whichever floor applies.
    run_dir = finished_run(tmp_path, rounds=(1, 2))
    for rnd in (1, 2):
        np.savez(
            run_dir / f"r{rnd}_b_zero_draft" / "score.npz",
            ids=np.array(IDS),
            activations=(activations() + 1.0).astype(np.float16),
        )
    floor = {"flips": 0, "mean_abs_dscore": 1e-6}
    out = r.report(
        run_dir,
        snapshot(activations()),
        snapshot(activations()),
        probe_path,
        floor,
        run_check=lambda *paths: {"exit_code": 0},
    )
    assert out["outcome"]["outcome"] == "fail"
    assert out["outcome"]["b_stock_highest_mean_abs_dscore"] < 0.05
