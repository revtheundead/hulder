"""v0.1's public API is hulder.__all__ plus the CLI commands. Changing it means amending the written decision."""

import hulder

PUBLIC = [
    "Estimate",
    "PairingError",
    "Probe",
    "Report",
    "Snapshot",
    "Verdict",
    "apply_probe",
    "check",
    "check_snapshots",
    "prompt_ids_sha256",
]


def test_hulder_all_is_the_public_api_recorded_in_adr_0005():
    assert sorted(hulder.__all__) == PUBLIC
    assert all(hasattr(hulder, name) for name in PUBLIC)
