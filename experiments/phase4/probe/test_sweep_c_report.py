"""Sweep step C's report: its stop-rule helpers. CPU only."""

import pytest

pytest.importorskip("sklearn")
import sweep_c_report as r


def test_majority_stops_become_the_plans_c1_and_c2_and_a_canary_without_majority_is_a_note():
    stops, notes = r.relabel(
        [
            "R1: the replicas' prompt_token_ids_sha256 differ",
            "R2: 9 of 676 batches have no majority, over the 1% limit",
            "R3: the replicas' canaries have no majority",
        ]
    )
    assert stops == [
        "C1: the replicas' prompt_token_ids_sha256 differ",
        "C2: 9 of 676 batches have no majority, over the 1% limit",
    ]
    assert notes == ["the replicas' canaries have no majority"]


def test_c1_names_rows_fed_other_tokens_than_kept_or_missing_from_it():
    kept = {"a": [1, 2], "b": [3]}
    assert r.tokens_differ({"a": [1, 2], "b": [4], "c": [5]}, kept) == ["b", "c"]
    assert r.tokens_differ(kept, kept) == []
