"""The Qwen3 probe's select stage on synthetic data: the C grid's tie and edge rules, the primary
cell, the threshold, the frozen selection, and probe metadata that check_snapshots accepts. CPU only."""

import json

import numpy as np
import pytest

import hulder

# The train extra; CI's base install skips this module.
pytest.importorskip("sklearn")
import train_qwen3 as t


def peaked_at(best: int, height: float = 0.99, slope: float = 0.01):
    """A validation AUROC over C that peaks at C = 10**best and falls by slope per decade."""
    return lambda c: height - slope * abs(np.log10(c) - best)


def test_a_tie_within_1e_4_of_the_maximum_goes_to_the_smaller_c():
    aurocs = {-3: 0.99000, -2: 0.99009, -1: 0.98990}
    assert t.choose_c(aurocs) == -3
    assert t.choose_c({-3: 0.98980, -2: 0.99000}) == -2  # 2e-4 below: no tie


def test_an_interior_maximum_extends_nothing():
    cell = t.select_c(peaked_at(-3))
    assert cell["C"] == 1e-3 and cell["position_in_grid"] == "interior"
    assert cell["extensions"] == [] and cell["edge_unresolved"] is False
    assert [g["C"] for g in cell["grid"]] == [10.0**e for e in range(-5, 1)]


def test_a_maximum_on_the_lower_edge_extends_the_grid_down_until_it_is_interior():
    cell = t.select_c(peaked_at(-7))
    assert cell["extensions"] == [1e-6, 1e-7, 1e-8]
    assert cell["C"] == 1e-7 and cell["position_in_grid"] == "interior" and cell["edge_unresolved"] is False


def test_a_maximum_on_the_upper_edge_extends_the_grid_up():
    cell = t.select_c(peaked_at(1))
    assert cell["extensions"] == [10.0, 100.0]
    assert cell["C"] == 10.0 and cell["position_in_grid"] == "interior"


def test_after_3_extensions_a_cell_still_on_an_edge_is_frozen_and_flagged():
    cell = t.select_c(lambda c: 0.9 - 0.001 * np.log10(c))  # always better at smaller C
    assert cell["extensions"] == [1e-6, 1e-7, 1e-8]
    assert cell["C"] == 1e-8 and cell["position_in_grid"] == "lower edge" and cell["edge_unresolved"] is True


def test_an_exact_tie_at_the_edge_extends_nothing_so_a_flat_grid_stops():
    cell = t.select_c(lambda c: 1.0)
    assert cell["extensions"] == [] and cell["edge_unresolved"] is False
    # The tie rule still takes the smallest C, and the report says where it sits.
    assert cell["C"] == 1e-5 and cell["position_in_grid"] == "lower edge"


def test_selection_is_the_same_whatever_order_the_cells_are_visited_in():
    cells = {
        (18, -1): {"C": 1e-3, "val_auroc": 0.9950},
        (27, -1): {"C": 1e-2, "val_auroc": 0.9960},
        (18, -9): {"C": 1e-3, "val_auroc": 0.9990},  # best overall, but not at position -1
    }
    forward = t.choose_primary(cells)
    backward = t.choose_primary(dict(reversed(list(cells.items()))))
    assert forward == backward == (27, -1)


def test_a_primary_tie_within_1e_4_goes_to_layer_18():
    cells = {(27, -1): {"C": 1e-2, "val_auroc": 0.99605}, (18, -1): {"C": 1e-3, "val_auroc": 0.99600}}
    assert t.choose_primary(cells) == (18, -1)


def test_the_threshold_is_the_benign_99th_percentile_taken_higher_and_flags_at_most_the_target():
    benign = np.random.default_rng(0).normal(size=1191)
    threshold = t.threshold_for_fpr(benign, 0.01)
    assert threshold == np.quantile(benign, 0.99, method="higher")
    assert (benign > threshold).sum() <= 0.01 * len(benign)


KEPT_META = {
    "base_model": "Qwen/Qwen3-4B",
    "model": "Qwen/Qwen3-4B",
    "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    "quantization": None,
    "dtype": "bfloat16",
    "batch_size": 8,
    "vllm_version": "0.30.0",
    "system_prompt": "You are a helpful assistant.",
    "layer": 27,
    "position": -10,
    "gpu": "NVIDIA L4",
    "enable_thinking": False,
}


def test_the_probe_metadata_passes_check_snapshots_against_a_kept_shaped_snapshot():
    rng = np.random.default_rng(0)
    y = np.arange(400) % 2
    x = (rng.normal(size=(400, 16)) + y[:, None]).astype(np.float16)
    meta = t.probe_metadata(KEPT_META, layer=27, position=-10, c=1e-3)
    probe = t.fit(x, y, 1e-3, meta)
    snapshot = hulder.Snapshot(
        ids=np.array([f"p/{i}" for i in range(400)]), labels=y, activations=x, metadata=KEPT_META
    )
    scores = hulder.apply_probe(probe, x)
    threshold = t.threshold_for_fpr(scores[y == 0], 0.05)
    report = hulder.check_snapshots(snapshot, snapshot, threshold=threshold, target_fpr=0.05, probe=probe)
    assert not [w for w in report.warnings if "probe" in w.lower()]
    assert meta["base_model"] == "Qwen/Qwen3-4B" and (meta["layer"], meta["position"]) == (27, -10)


class FakeData:
    """Two separable features per cell, and a record of which splits were read."""

    def __init__(self):
        self.read = []

    def rows(self, split, layer, position):
        # The same rows on every read of a split and cell, as kept/ gives.
        self.read.append(split)
        n = {"train": 300, "validation": 200, "calibration": 400}[split]
        y = np.arange(n) % 2
        rng = np.random.default_rng([n, layer, -position])
        x = rng.normal(size=(n, 8)) + (0.5 + layer / 100 - position / 50) * y[:, None]
        return x.astype(np.float16), y


def test_the_select_stage_reads_no_test_or_boundary_pairs_rows_and_freezes_its_selection(tmp_path):
    data = FakeData()
    out = tmp_path / "selection.json"
    selection = t.select_stage(data.rows, out, probe_dir=tmp_path / "probes", source={"kept": "x"})
    assert set(data.read) == {"train", "validation", "calibration"}
    saved = json.loads(out.read_text())
    assert saved == selection
    assert set(saved["cells"]) == {f"{layer}_{position}" for layer in (18, 27) for position in (-1, -9, -10)}
    assert saved["primary"] in {"18_-1", "27_-1"}
    for name, cell in saved["cells"].items():
        assert set(cell) >= {"C", "grid", "position_in_grid", "extensions", "edge_unresolved", "val_auroc", "threshold"}
        assert (tmp_path / "probes" / f"{name}.npz").exists()
    assert saved["refit_identical"] is True


def test_the_select_stage_refuses_to_overwrite_a_frozen_selection(tmp_path):
    out = tmp_path / "selection.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="frozen"):
        t.select_stage(FakeData().rows, out, probe_dir=tmp_path / "probes", source={})
    assert out.read_text() == "{}"
