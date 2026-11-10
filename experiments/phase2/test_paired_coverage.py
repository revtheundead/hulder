"""The coverage run's bookkeeping, on seeds outside its committed range so no result is seen early."""

import paired_coverage as pc
import pytest


@pytest.fixture
def other_seeds(monkeypatch):
    monkeypatch.setattr(pc, "SEEDS", range(30))


def test_grid_covers_every_size_and_rate():
    cells = pc.cells()
    assert len(cells) == len(pc.SIZES) * len(pc.BOTH) * len(pc.ONE_WAY) ** 2
    assert len({c["key"] for c in cells}) == len(cells)


def test_the_seed_range_is_fresh():
    assert pc.SEEDS == range(7000, 8000)
    used = [range(1000), range(1000, 2000), range(2000, 3000)]
    assert all(set(pc.SEEDS).isdisjoint(r) for r in used)


def test_tables_are_deterministic_and_sum_to_n(other_seeds):
    cell = {"n": 180, "a": 0.05, "b": 0.01, "c": 0.0}
    first, second = pc.tables(cell), pc.tables(cell)
    assert (first == second).all()
    assert first.shape == (30 * pc.DRAWS_PER_SEED, 4)
    assert (first.sum(axis=1) == 180).all()
    assert (first[:, 2] == 0).all()


def test_no_change_and_no_discordance_is_never_missed(other_seeds):
    row = pc.run_cell({"key": "k", "n": 180, "a": 0.05, "b": 0.0, "c": 0.0})
    assert row["no_discordant"] == 1.0
    for name in pc.INTERVALS:
        assert row[name]["low_miss"] == row[name]["high_miss"] == 0.0


def test_exact_conditional_misses_high_whenever_a_real_change_shows_no_discordance(other_seeds):
    row = pc.run_cell({"key": "k", "n": 180, "a": 0.05, "b": 0.0075, "c": 0.0})
    assert row["exact_conditional"]["high_miss"] >= row["no_discordant"] > 0


def test_summary_picks_the_narrowest_interval_meeting_both_bars():
    def cell(key, misses):
        return {name: {"low_miss": m, "high_miss": m, "mean_width": w} for name, (m, w) in misses.items()} | {
            "key": key,
            "n": 180,
            "b": 0.005,
            "c": 0.0,
            "no_discordant": 0.4,
        }

    done = {
        "x": cell("x", {"exact_conditional": (0.20, 0.001), "tango": (0.02, 0.03), "newcombe": (0.02, 0.02)}),
        "y": cell("y", {"exact_conditional": (0.01, 0.001), "tango": (0.02, 0.03), "newcombe": (0.03, 0.02)}),
    }
    summary = pc.summarise([{"key": "x"}, {"key": "y"}], done)
    assert not summary["per_interval"]["exact_conditional"]["meets_bar"]
    # Both score intervals meet the bars, and newcombe is the narrower.
    assert summary["chosen"] == "newcombe"
    assert [r["key"] for r in summary["low_discordance_n180"]] == ["x", "y"]


def test_expected_document_states_the_bars():
    doc = pc.expected_document()
    assert doc["seeds"] == [7000, 8000]
    assert doc["cell_max"] == 0.035 and doc["mean_max"] == 0.0275
    assert doc["intervals"] == ["exact_conditional", "tango", "newcombe"]
