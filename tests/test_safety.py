import pytest

from app.gemma_extractor import (
    ExtractionResult,
    _apply_safety_rules,
    _apply_transaction_type_rules,
    _check_amount_against_transcript,
)


def make(
    amount=340,
    amount_conf=1.0,
    ttype="credit",
    needs=False,
):
    return ExtractionResult(
        customer={
            "value": "Ramesh",
            "confidence": 1.0,
        },
        items=[],
        amount_inr={
            "value": amount,
            "confidence": amount_conf,
        },
        due_date_text={
            "value": None,
            "confidence": 0.0,
        },
        transaction_type=ttype,
        overall_confidence=1.0,
        needs_confirmation=needs,
        ambiguities=[],
    )


def test_clear_credit_passes_without_confirmation():
    result = _apply_safety_rules(make())

    assert result.amount_inr.value == 340
    assert result.needs_confirmation is False


def test_low_amount_confidence_forces_confirmation():
    result = _apply_safety_rules(
        make(amount_conf=0.50)
    )

    assert result.needs_confirmation is True


def test_negative_amount_is_rejected():
    with pytest.raises(ValueError):
        _apply_safety_rules(
            make(amount=-340)
        )


def test_two_spoken_monetary_amounts_clear_the_amount():
    result = _check_amount_against_transcript(
        make(),
        "Rahul-er 340 na 3400 baaki",
    )

    assert result.amount_inr.value is None
    assert result.amount_inr.confidence == 0.0
    assert result.needs_confirmation is True


def test_quantity_numbers_are_not_treated_as_money():
    result = _check_amount_against_transcript(
        make(),
        "Gopal 5 kg chal 340 baaki",
    )

    assert result.amount_inr.value == 340
    assert result.needs_confirmation is False


def test_future_promise_is_not_a_payment():
    result = _apply_transaction_type_rules(
        make(
            amount=500,
            ttype="payment",
        ),
        "Mita 500 taka debe",
    )

    assert result.transaction_type == "unknown"
    assert result.needs_confirmation is True


def test_actual_payment_remains_payment():
    result = _apply_transaction_type_rules(
        make(
            amount=500,
            ttype="payment",
        ),
        "Mita 500 taka diyeche",
    )

    assert result.transaction_type == "payment"
    assert result.needs_confirmation is False
