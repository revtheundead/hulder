"""Pairing: whether two runs scored the same prompts in the same order. Used by check and Snapshot."""

import hashlib
import json
from collections.abc import Iterable

# Hashes the prompt ids in order. If they differ, the two runs aren't paired.
PROMPT_IDS_KEY = "prompt_ids_sha256"


class PairingError(ValueError):
    """Reference and candidate were not run on the same prompts in the same order."""


def prompt_ids_sha256(ids: Iterable) -> str:
    """Hash of example ids in order, each as a string. JSON keeps id boundaries unambiguous."""
    return hashlib.sha256(json.dumps([str(i) for i in ids]).encode()).hexdigest()


def check_pairing(
    reference_meta: dict | None,
    candidate_meta: dict | None,
    reference_ids: Iterable | None,
    candidate_ids: Iterable | None,
    noise_floor_ids: Iterable | None,
    has_noise_floor: bool,
) -> list[str]:
    """Each run's ids hash comes from its metadata, its ids, or both, which must then agree.
    Known hashes that differ from the reference's raise PairingError; an unknown one warns."""
    reference = _ids_hash(reference_meta, reference_ids, "reference")
    warnings = _pair(reference, _ids_hash(candidate_meta, candidate_ids, "candidate"), "candidate")
    if has_noise_floor:
        warnings += _pair(reference, _ids_hash(None, noise_floor_ids, "rerun"), "rerun")
    return warnings


def _ids_hash(meta: dict | None, ids: Iterable | None, role: str) -> str | None:
    recorded = (meta or {}).get(PROMPT_IDS_KEY)
    if ids is None:
        return recorded
    computed = prompt_ids_sha256(ids)
    if recorded is not None and recorded != computed:
        raise PairingError(
            f"The {role}'s ids don't match the {PROMPT_IDS_KEY} in its metadata "
            f"({computed!r} vs {recorded!r}): the ids or the metadata belong to another run."
        )
    return computed


def _pair(reference: str | None, other: str | None, role: str) -> list[str]:
    if reference is None or other is None:
        return [
            (
                f"{role.capitalize()} pairing not verified; its scores are assumed to be in the reference's order. "
                f"Pass example ids for the reference and the {role}."
            )
        ]
    if reference != other:
        raise PairingError(
            f"{PROMPT_IDS_KEY} differs ({reference!r} vs {other!r}): "
            f"reference and {role} were not run on the same prompts in the same order."
        )
    return []
