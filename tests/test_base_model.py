import pytest

from hulder.capture.base_model import base_model_for, base_model_in_card


def _card(base_model):
    def read(model, revision):
        return base_model

    return read


def _raises(error):
    def read(model, revision):
        raise error

    return read


# What huggingface_hub 1.33.0 raises, all OSErrors: RemoteEntryNotFoundError for a repo without the
# file, LocalEntryNotFoundError (a FileNotFoundError) offline or unreachable without a cached copy.
UNREADABLE = [
    FileNotFoundError("404: README.md not found"),
    ConnectionError("Hugging Face unreachable"),
    FileNotFoundError("HF_HUB_OFFLINE=1 and no cached copy"),
]
UNREADABLE_IDS = ["no card", "unreachable", "offline"]


def _quantized(answer):
    def read(model, revision):
        return answer

    return read


def test_an_unquantized_capture_is_its_own_base_model():
    # The Instruct card names its pretrained parent, Qwen/Qwen2.5-1.5B; that is not the
    # model the AWQ checkpoint re-encodes, so the card must not be read here.
    assert base_model_for(
        "Qwen/Qwen2.5-1.5B-Instruct",
        "989aa79",
        None,
        card=_card("Qwen/Qwen2.5-1.5B"),
        checkpoint_quantized=_quantized(False),
    ) == (
        "Qwen/Qwen2.5-1.5B-Instruct",
        None,
    )


def test_a_quantized_checkpoint_takes_its_cards_base_model():
    assert base_model_for(
        "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        "3ecffa0",
        "awq",
        card=_card("Qwen/Qwen2.5-1.5B-Instruct"),
        checkpoint_quantized=_quantized(True),
    ) == ("Qwen/Qwen2.5-1.5B-Instruct", None)


def test_an_override_wins_over_the_card():
    assert base_model_for("org/m-AWQ", "r", "awq", override="org/m", card=_card("org/other")) == ("org/m", None)


@pytest.mark.parametrize("field", [None, []], ids=["missing", "empty list"])
def test_a_card_without_a_base_model_warns_rather_than_guesses(field):
    base_model, warning = base_model_for(
        "org/m-AWQ", "r", "awq", card=_card(field), checkpoint_quantized=_quantized(True)
    )
    assert base_model is None
    assert "org/m-AWQ" in warning
    assert "names no base_model" in warning
    assert "recorded no base_model" in warning
    assert "override=" in warning


def test_online_quantization_records_the_model_itself_without_reading_its_card():
    # vLLM's quantization="fp8" on Qwen/Qwen3-4B quantizes the bf16 weights at load. Its card
    # names the pretrained parent, which would make the gate report a model swap.
    assert base_model_for(
        "Qwen/Qwen3-4B", "1cfa9a7", "fp8", card=_card(["Qwen/Qwen3-4B-Base"]), checkpoint_quantized=_quantized(False)
    ) == ("Qwen/Qwen3-4B", None)


def test_a_quantized_checkpoint_left_to_vllm_to_detect_still_takes_its_cards_base_model():
    # With quantization=None, vLLM 0.30.0 reads the method from config.json's quantization_config.
    assert base_model_for(
        "Qwen/Qwen3-4B-AWQ", "74d4bd2", None, card=_card("Qwen/Qwen3-4B"), checkpoint_quantized=_quantized(True)
    ) == ("Qwen/Qwen3-4B", None)


def test_a_one_element_base_model_list_is_read_as_its_entry():
    # Qwen/Qwen3-4B-FP8's card at 96b30dc1 gives base_model as a one-element YAML list.
    assert base_model_for(
        "Qwen/Qwen3-4B-FP8", "96b30dc", "fp8", card=_card(["Qwen/Qwen3-4B"]), checkpoint_quantized=_quantized(True)
    ) == ("Qwen/Qwen3-4B", None)


def test_several_base_models_warn_naming_them_and_record_none():
    base_model, warning = base_model_for(
        "org/merge-AWQ", "r", "awq", card=_card(["org/a", "org/b"]), checkpoint_quantized=_quantized(True)
    )
    assert base_model is None
    assert "several base models" in warning
    assert "'org/a'" in warning and "'org/b'" in warning
    assert "recorded no base_model" in warning
    assert "override=" in warning


@pytest.mark.parametrize("error", UNREADABLE, ids=UNREADABLE_IDS)
def test_an_unreadable_card_warns_with_the_reason_instead_of_raising(error):
    base_model, warning = base_model_for(
        "org/m-AWQ", "r", "awq", card=_raises(error), checkpoint_quantized=_quantized(True)
    )
    assert base_model is None
    assert str(error) in warning
    assert "recorded no base_model" in warning
    assert "override=" in warning


@pytest.mark.parametrize("error", UNREADABLE, ids=UNREADABLE_IDS)
def test_unquantized_with_an_unreadable_config_assumes_the_model_is_its_own_base_and_says_so(error):
    base_model, warning = base_model_for(
        "org/m", "r", None, card=_card("org/parent"), checkpoint_quantized=_raises(error)
    )
    assert base_model == "org/m"
    assert str(error) in warning
    assert "assumed" in warning and "not quantized" in warning
    assert "override=" in warning


@pytest.mark.parametrize("error", UNREADABLE, ids=UNREADABLE_IDS)
def test_quantized_with_an_unreadable_config_records_none_rather_than_guess_online_or_not(error):
    # Online quantization would make the model its own base, a quantized checkpoint its card's
    # base_model; config.json is what tells them apart.
    base_model, warning = base_model_for(
        "org/m", "r", "fp8", card=_card("org/parent"), checkpoint_quantized=_raises(error)
    )
    assert base_model is None
    assert str(error) in warning
    assert "recorded no base_model" in warning
    assert "override=" in warning


@pytest.mark.parametrize("method", ["awq", "gptq", "fp8", None])
def test_every_checkpoint_quantized_before_upload_takes_its_cards_base_model(method):
    assert base_model_for(
        "org/m-quantized", "r", method, card=_card(["org/m"]), checkpoint_quantized=_quantized(True)
    ) == ("org/m", None)


# The front matter of Qwen/Qwen3-4B-FP8's README.md at 96b30dc13593a244a5e59e84687309f53c375cfa, verbatim.
QWEN3_4B_FP8_CARD = """---
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3-4B-FP8/blob/main/LICENSE
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3-4B
---

# Qwen3-4B-FP8
"""


def test_a_cards_base_model_is_read_as_written():
    assert base_model_in_card(QWEN3_4B_FP8_CARD) == ["Qwen/Qwen3-4B"]


@pytest.mark.parametrize(
    "text", ["---\nbase_model: [unclosed\n---\n", "---\njust a string\n---\n"], ids=["malformed", "not a mapping"]
)
def test_a_damaged_front_matter_is_a_value_error_which_base_model_for_reports_as_unreadable(text):
    with pytest.raises(ValueError):
        base_model_in_card(text)
