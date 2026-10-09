from hulder.capture.base_model import base_model_for


def _card(base_model):
    def read(model, revision):
        return base_model

    return read


def test_an_unquantized_capture_is_its_own_base_model():
    # The Instruct card names its pretrained parent, Qwen/Qwen2.5-1.5B; that is not the
    # model the AWQ checkpoint re-encodes, so the card must not be read here.
    assert base_model_for("Qwen/Qwen2.5-1.5B-Instruct", "989aa79", None, card=_card("Qwen/Qwen2.5-1.5B")) == (
        "Qwen/Qwen2.5-1.5B-Instruct",
        None,
    )


def test_a_quantized_checkpoint_takes_its_cards_base_model():
    assert base_model_for(
        "Qwen/Qwen2.5-1.5B-Instruct-AWQ", "3ecffa0", "awq", card=_card("Qwen/Qwen2.5-1.5B-Instruct")
    ) == ("Qwen/Qwen2.5-1.5B-Instruct", None)


def test_an_override_wins_over_the_card():
    assert base_model_for("org/m-AWQ", "r", "awq", override="org/m", card=_card("org/other")) == ("org/m", None)


def test_a_quantized_checkpoint_without_card_or_override_warns_rather_than_guesses():
    base_model, warning = base_model_for("org/m-AWQ", "r", "awq", card=_card(None))
    assert base_model is None
    assert "org/m-AWQ" in warning
    assert "base_model" in warning
