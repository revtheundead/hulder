"""The sweep's configs (docs/results/phase4/sweep.md) and the runner keeping each config as it
finishes. CPU only, no Modal."""

import hashlib
import json

import pytest
import smoke_lib as lib

AWQ_REVISION = "74d4bd2bd4bff9cafc9345221320bffb08b406a3"  # docs/design.md §7


def test_the_awq_smoke_is_the_full_captures_config_on_the_official_awq_checkpoint_in_bf16():
    full, _ = lib.CONFIGS["qwen3_4b_bf16_full"]
    awq, source = lib.CONFIGS["qwen3_4b_awq_smoke"]
    assert awq == {**full, "model": "Qwen/Qwen3-4B-AWQ", "revision": AWQ_REVISION, "quantization": "awq"}
    assert awq["dtype"] == "bfloat16"
    assert source == ("smoke", 64)


def test_the_fp8_kv_smoke_is_the_full_bf16_config_on_one_prompt_with_fp8_kv_passed_to_the_engine():
    full, _ = lib.CONFIGS["qwen3_4b_bf16_full"]
    fields, _ = lib.CONFIGS["qwen3_4b_bf16_fp8kv_smoke"]
    assert fields == full
    assert lib.ENGINE_KWARGS == {"qwen3_4b_bf16_fp8kv_smoke": {"kv_cache_dtype": "fp8"}}
    spec = lib.worker_spec(("bf16_fp8kv", "qwen3_4b_bf16_fp8kv_smoke"), 480)
    assert spec["engine_kwargs"] == {"kv_cache_dtype": "fp8"}
    # One prompt behind the canary and no snapshot, since CaptureConfig can't record the KV dtype.
    assert len(spec["prompts"]) == 1 and spec["save"] is False


def test_every_other_config_passes_nothing_extra_to_the_engine():
    for name in lib.CONFIGS:
        if name != "qwen3_4b_bf16_fp8kv_smoke":
            assert lib.worker_spec(("x", name), 480)["engine_kwargs"] == {}


def test_step_a_is_one_container_awq_then_fp8_kv_within_its_declared_worst_case():
    containers, single_use, timeout_s = lib.EXPERIMENTS["sweep_a"]
    assert containers == [[("awq_bf16", "qwen3_4b_awq_smoke"), ("bf16_fp8kv", "qwen3_4b_bf16_fp8kv_smoke")]]
    assert not single_use and timeout_s == 8 * 60
    expected = json.loads((lib.ROOT / "docs/results/phase4/sweep_expected.json").read_text())["steps"]["A"]
    assert lib.run_worst_case(containers, timeout_s) == pytest.approx(expected["cost_usd"]["worst"], abs=1e-4)


def test_step_a_runs_its_second_arm_after_the_first_fails_and_other_experiments_stop():
    specs = lib.experiment_specs("sweep_a")
    assert [label for label, _ in specs[0]] == ["awq_bf16", "bf16_fp8kv"]
    assert all(spec["continue_on_failure"] for group in specs for _, spec in group)
    assert not any(spec["continue_on_failure"] for group in lib.experiment_specs("exp2") for _, spec in group)


def test_experiment_specs_are_the_worker_specs_of_each_container_in_order():
    containers, _, timeout_s = lib.EXPERIMENTS["smoke"]
    specs = lib.experiment_specs("smoke")
    assert [[label for label, _ in group] for group in specs] == [[e[0] for e in group] for group in containers]
    for group, spec_group in zip(containers, specs, strict=True):
        for entry, (_, spec) in zip(group, spec_group, strict=True):
            assert {k: v for k, v in spec.items() if k != "continue_on_failure"} == lib.worker_spec(entry, timeout_s)


FILES = {"18_-1.npz": b"layer 18 rows", "27_-1.npz": b"layer 27 rows", "token_ids.npz": b"ids"}


def test_a_configs_snapshot_hashes_are_written_beside_them_on_the_volume(tmp_path):
    for name, data in FILES.items():
        (tmp_path / name).write_bytes(data)
    (tmp_path / "capture.log").write_text("not a snapshot")
    hashes = lib.record_volume_files(tmp_path)
    assert hashes == {name: hashlib.sha256(data).hexdigest() for name, data in FILES.items()}
    assert json.loads((tmp_path / lib.VOLUME_FILES).read_text()) == hashes
    assert not list(tmp_path.glob("*.part"))


class VolumeDir:
    """One config's folder on the Volume, read by name as capture.py's recover reads it."""

    def __init__(self, files: dict[str, bytes]):
        self.files = files

    def read(self, name: str) -> bytes:
        if name not in self.files:
            raise FileNotFoundError(name)
        return self.files[name]


def finished(tmp_path, returncode=0):
    """A config folder as run_group leaves it on the Volume once its engine has finished."""
    tmp_path.mkdir(parents=True)
    for name, data in FILES.items():
        (tmp_path / name).write_bytes(data)
    hashes = lib.record_volume_files(tmp_path)
    manifest = {"label": "r1", "returncode": returncode, "seconds": 12.0}
    files = {name: (tmp_path / name).read_bytes() for name in [*FILES, lib.VOLUME_FILES]}
    return VolumeDir({**files, "manifest.json": json.dumps(manifest).encode(), "capture.log": b"log"}), hashes


def test_recover_rebuilds_a_finished_config_and_backs_it_up_checked_against_the_containers_hashes(tmp_path):
    volume, hashes = finished(tmp_path / "vol")
    entry = lib.recover_config(volume.read, tmp_path / "local", tmp_path / "backup")
    assert entry["returncode"] == 0 and entry["volume"]["problem"] is None
    assert entry["volume"]["fetched"] and entry["volume"]["backed_up"]
    assert entry["volume"]["files"] == hashes
    assert (tmp_path / "local" / "capture.log").read_bytes() == b"log"
    assert json.loads((tmp_path / "local" / "manifest.json").read_text())["seconds"] == 12.0
    for name, data in FILES.items():
        assert (tmp_path / "local" / name).read_bytes() == data == (tmp_path / "backup" / name).read_bytes()


def test_recover_skips_a_config_whose_manifest_was_never_written(tmp_path):
    volume, _ = finished(tmp_path / "vol")
    del volume.files["manifest.json"]
    assert lib.recover_config(volume.read, tmp_path / "local", None) is None
    assert not (tmp_path / "local").exists()


def test_recover_flags_a_snapshot_that_doesnt_match_its_recorded_hash(tmp_path):
    volume, _ = finished(tmp_path / "vol")
    volume.files["18_-1.npz"] = b"damaged"
    entry = lib.recover_config(volume.read, tmp_path / "local", None)
    assert entry["volume"]["problem"] == "18_-1.npz as fetched doesn't match the container's sha256"


def test_recover_keeps_a_failed_configs_log_and_manifest_with_no_snapshots(tmp_path):
    volume = VolumeDir({"manifest.json": json.dumps({"returncode": 1}).encode(), "capture.log": b"AssertionError"})
    entry = lib.recover_config(volume.read, tmp_path / "local", None)
    assert entry["returncode"] == 1 and entry["volume"]["files"] == {}
    assert (tmp_path / "local" / "capture.log").read_bytes() == b"AssertionError"


def test_containers_are_collected_in_the_order_they_finish_not_the_order_they_started():
    done_after = {0: 3, 1: 1, 2: 2}  # polls before each call has a result
    polls = dict.fromkeys(done_after, 0)
    waits = []

    def poll(i):
        polls[i] += 1
        return (True, f"result {i}") if polls[i] > done_after[i] else (False, None)

    order = list(lib.in_finish_order(3, poll, lambda: waits.append(1)))
    assert order == [(1, "result 1"), (2, "result 2"), (0, "result 0")]
    assert all(polls[i] == done_after[i] + 1 for i in done_after)  # nothing polled after it finished
    assert len(waits) == 1  # only the first round finds nothing; each later round finds a result
