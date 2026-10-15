"""The base_model a capture records, which check requires to match across runs.

Kept apart from hidden_states.py so it imports without vLLM.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path

# What reading from the Hugging Face Hub raises, all handled as "couldn't read": every hub error,
# network failure and offline miss is an OSError (huggingface_hub 1.33.0), a local path as the
# repo id a ValueError, and the hub client missing an ImportError.
UNREADABLE = (OSError, ValueError, ImportError)


def card_base_model(model: str, revision: str) -> object:
    """The base_model field of a model card at a pinned revision, as base_model_in_card reads it."""
    from huggingface_hub import hf_hub_download

    return base_model_in_card(Path(hf_hub_download(model, "README.md", revision=revision)).read_text())


def base_model_in_card(text: str) -> object:
    """The base_model field of a model card's YAML front matter, as written: a string, a list, or
    None when absent. base_model_for interprets it. Damaged front matter is a ValueError."""
    # Imported here so base_model_for, and its tests, run on the base install.
    import yaml

    front = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if front is None:
        return None
    try:
        fields = yaml.safe_load(front.group(1))
    except yaml.YAMLError as error:
        raise ValueError(f"the model card's front matter isn't valid YAML: {error}") from error
    if not isinstance(fields, dict):
        raise ValueError(f"the model card's front matter is a {type(fields).__name__}, not a mapping")  # noqa: TRY004, the file is at fault, not the caller
    return fields.get("base_model")


def checkpoint_quantized(model: str, revision: str) -> bool:
    """Whether the checkpoint was quantized before upload: its config.json has a quantization_config.
    Without one, a quantized capture is vLLM quantizing the weights at load."""
    from huggingface_hub import hf_hub_download

    return "quantization_config" in json.loads(
        Path(hf_hub_download(model, "config.json", revision=revision)).read_text()
    )


def base_model_for(
    model: str,
    revision: str,
    quantization: str | None,
    override: str | None = None,
    card: Callable[[str, str], object] = card_base_model,
    checkpoint_quantized: Callable[[str, str], bool] = checkpoint_quantized,
) -> tuple[str | None, str | None]:
    """(base_model, warning). An override wins. A checkpoint whose config.json has no
    quantization_config is its own base, unquantized or quantized by vLLM at load: its card's
    base_model names the parent it was fine-tuned from, not what it re-encodes. One quantized
    before upload takes its card's base_model, a string or a one-element list. Every fallback
    returns a warning saying what was assumed, and none raises."""
    if override is not None:
        return override, None
    try:
        quantized_before_upload = checkpoint_quantized(model, revision)
    except UNREADABLE as error:
        if quantization is not None:
            return None, _recorded_none(
                f"{model}@{revision}'s config.json couldn't be read ({error}), so hulder can't tell "
                f"whether {quantization!r} was applied before upload or at load"
            )
        return model, (
            f"{model}@{revision}'s config.json couldn't be read ({error}), so hulder assumed the "
            "checkpoint is not quantized and recorded it as its own base_model. If it is a quantized "
            "checkpoint, pass its base model as override= to base_model_for."
        )
    if not quantized_before_upload:
        return model, None
    try:
        base_model = card(model, revision)
    except UNREADABLE as error:
        return None, _recorded_none(f"{model}@{revision} is quantized but its model card couldn't be read ({error})")
    if isinstance(base_model, list) and len(base_model) == 1:
        base_model = base_model[0]
    if isinstance(base_model, list) and len(base_model) > 1:
        return None, _recorded_none(f"{model}@{revision}'s model card names several base models ({base_model!r})")
    if not isinstance(base_model, str):
        return None, _recorded_none(f"{model}@{revision} is quantized but its model card names no base_model")
    return base_model, None


def _recorded_none(reason: str) -> str:
    return (
        f"{reason}, so hulder recorded no base_model, and check will warn that it can't confirm both "
        "runs serve the same model. Pass the base model as override= to base_model_for."
    )
