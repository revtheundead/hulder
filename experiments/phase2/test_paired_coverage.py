"""The coverage run's bookkeeping, on seeds outside its committed range so no result is seen early."""

import json

import paired_coverage as pc
import pytest

OTHER_SEEDS = range(30)


@pytest.fixture
def other_seeds(monkeypatch):
    for run in pc.RUNS:
        monkeypatch.setitem(pc.RUNS[run], "seeds", OTHER_SEEDS)


def test_grid_covers_every_size_and_rate():
    cells = pc.cells("first")
    assert len(cells) == len(pc.SIZES) * len(pc.BOTH) * len(pc.ONE_WAY) ** 2
    assert len({c["key"] for c in cells}) == len(cells)


def test_each_runs_seed_range_is_fresh():
    assert pc.RUNS["first"]["seeds"] == range(7000, 8000)
    assert pc.RUNS["continuity_corrected"]["seeds"] == range(9000, 10000)
    # Settle, the flip check and planted faults, the shift note, and the paired rule's other runs.
    used = [range(3000), range(3000, 4000), range(5000, 7000), range(8000, 9000)]
    for run in pc.RUNS.values():
        assert all(set(run["seeds"]).isdisjoint(r) for r in used)
    assert set(pc.RUNS["first"]["seeds"]).isdisjoint(pc.RUNS["continuity_corrected"]["seeds"])


def test_tables_are_deterministic_and_sum_to_n(other_seeds):
    cell = {"n": 180, "a": 0.05, "b": 0.01, "c": 0.0}
    first, second = pc.tables(cell, OTHER_SEEDS), pc.tables(cell, OTHER_SEEDS)
    assert (first == second).all()
    assert first.shape == (30 * pc.DRAWS_PER_SEED, 4)
    assert (first.sum(axis=1) == 180).all()
    assert (first[:, 2] == 0).all()


def test_no_change_and_no_discordance_is_never_missed(other_seeds):
    row = pc.run_cell({"key": "k", "run": "first", "n": 180, "a": 0.05, "b": 0.0, "c": 0.0})
    assert row["no_discordant"] == 1.0
    for name in pc.RUNS["first"]["intervals"]:
        assert row[name]["low_miss"] == row[name]["high_miss"] == 0.0


def test_exact_conditional_misses_high_whenever_a_real_change_shows_no_discordance(other_seeds):
    row = pc.run_cell({"key": "k", "run": "first", "n": 180, "a": 0.05, "b": 0.0075, "c": 0.0})
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
    summary = pc.summarise([{"key": "x"}, {"key": "y"}], done, pc.RUNS["first"]["intervals"])
    assert not summary["per_interval"]["exact_conditional"]["meets_bar"]
    # Both score intervals meet the bars, and newcombe is the narrower.
    assert summary["chosen"] == "newcombe"
    assert [r["key"] for r in summary["low_discordance_n180"]] == ["x", "y"]


def test_the_first_runs_committed_bar_is_unchanged():
    committed = json.loads(pc.expected_path("first").read_text())
    assert committed == json.loads(json.dumps(pc.expected_document("first"), ensure_ascii=False))


def test_the_second_run_keeps_the_bar_and_says_when_it_was_chosen():
    first, second = pc.expected_document("first"), pc.expected_document("continuity_corrected")
    for key in ("bars", "cell_max", "mean_max", "grid", "low_discordance", "alpha_two_sided", "miss", "truth"):
        assert second[key] == first[key]
    assert second["intervals"] == ["newcombe_cc"]
    assert second["seeds"] == [9000, 10000]
    assert "chosen after seeing the first run's cells" in second["chosen_after_the_first_run"]


def test_the_second_run_scores_only_its_candidate(other_seeds):
    row = pc.run_cell({"key": "k", "run": "continuity_corrected", "n": 180, "a": 0.05, "b": 0.005, "c": 0.0})
    assert set(row) == {"key", "tables", "no_discordant", "newcombe_cc"}
