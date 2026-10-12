"""The base_model a capture records, which check requires to match across runs.

Kept apart from hidden_states.py so it imports without vLLM.
"""

import re
from collections.abc import Callable
from pathlib import Path


def card_base_model(model: str, revision: str) -> str | None:
    """The base_model field of a model card's YAML front matter at a pinned revision."""
    # Imported here so base_model_for, and its tests, run on the base install.
    import yaml
    from huggingface_hub import hf_hub_download

    text = Path(hf_hub_download(model, "README.md", revision=revision)).read_text()
    front = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    value = yaml.safe_load(front.group(1)).get("base_model") if front else None
    return value if isinstance(value, str) else None


def base_model_for(
    model: str,
    revision: str,
    quantization: str | None,
    override: str | None = None,
    card: Callable[[str, str], str | None] = card_base_model,
) -> tuple[str | None, str | None]:
    """(base_model, warning). Unquantized, the model is its own base: a card's base_model
    names the parent it was fine-tuned from, not what it re-encodes. Quantized, the override
    or else the card's field; with neither, None and a warning rather than a guess."""
    if override is not None:
        return override, None
    if quantization is None:
        return model, None
    base_model = card(model, revision)
    if base_model is None:
        return None, (
            f"{model}@{revision} is quantized but its model card names no single base_model; "
            "pass the base model explicitly so check can confirm both runs serve the same model."
        )
    return base_model, None
