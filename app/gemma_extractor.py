import json
import re
from typing import Literal

import ollama
from pydantic import BaseModel, Field, ValidationError


MODEL = "gemma3:4b"


# -----------------------------
# Pydantic schemas
# -----------------------------

class CustomerField(BaseModel):
    value: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class Item(BaseModel):
    name: str
    quantity: float | None = None
    unit: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class AmountField(BaseModel):
    value: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class DueDateField(BaseModel):
    value: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    customer: CustomerField
    items: list[Item] = Field(default_factory=list)

    amount_inr: AmountField

    due_date_text: DueDateField

    transaction_type: Literal["credit", "payment", "unknown"]

    overall_confidence: float = Field(ge=0.0, le=1.0)

    needs_confirmation: bool

    ambiguities: list[str] = Field(default_factory=list)


# -----------------------------
# Prompt
# -----------------------------

SYSTEM_PROMPT = """
You are OpenKhata's financial transaction extraction engine.

OpenKhata is a LOCAL credit ledger for a small Indian kirana shop.

The shopkeeper speaks naturally in Bengali, Hindi, English, or
Bengali/Hindi/English code-mixed speech.

Your job is ONLY to extract information explicitly present in the
transcript.

DO NOT explain the transaction.
DO NOT summarize the transaction.
DO NOT invent missing information.
DO NOT reinterpret the transaction.
DO NOT write conversational text.

Return ONLY valid JSON.

IMPORTANT KIRANA VOCABULARY:

"baaki", "udhar", "baki", "bakaya", "due", "owes"
usually indicate that MONEY IS OWED TO THE SHOP.

For example:

"Ramesh-er 340 baaki"

means:

Ramesh owes the shop ₹340.

It does NOT mean that Ramesh gave ₹340 change.

"Friday-te dibe", "Friday dega", "Friday ko dega"
means the customer is expected to PAY/GIVE THE MONEY on Friday.

It does NOT mean the transaction happened on Friday.

TRANSACTION TYPES:

"credit":
The customer owes the shop money.

"payment":
The customer is paying an existing debt.

"unknown":
The transcript does not clearly establish either.

FINANCIAL SAFETY:

Never silently change an amount.

Examples:

340 -> 340
3,400 -> 3400
₹340 -> 340
₹3,400 -> 3400

If the transcript genuinely contains two possible amounts,
do NOT choose one.

Use:

"value": null

and explain the ambiguity.

Never invent an amount.

CUSTOMER SAFETY:

Extract the customer's spoken name as closely as possible.

Do not replace a nickname with a guessed legal/full name.

DATE SAFETY:

Preserve the spoken date expression.

Examples:

"Friday-te" -> "Friday-te"
"kal" -> "kal"
"next Monday" -> "next Monday"

Do not calculate calendar dates here.

ITEM SAFETY:

Extract only items explicitly mentioned.

Do not invent quantities.

CONFIDENCE:

0.95-1.00:
Explicit and very clear.

0.85-0.94:
Clear but minor linguistic uncertainty.

0.70-0.84:
Some uncertainty.

Below 0.70:
Ambiguous or potentially incorrect.

MANDATORY CONFIRMATION:

Set needs_confirmation=true when:

- amount confidence < 0.95
- customer confidence < 0.90
- an important field is ambiguous
- multiple monetary amounts are possible
- the transcript contains an unclear number
- information had to be inferred
- the extraction is otherwise unsafe to save automatically

For a completely clear amount, customer and transaction type,
needs_confirmation may be false.

OUTPUT SCHEMA:

{
  "customer": {
    "value": "string or null",
    "confidence": 0.0
  },
  "items": [
    {
      "name": "string",
      "quantity": 0,
      "unit": "string or null",
      "confidence": 0.0
    }
  ],
  "amount_inr": {
    "value": 0,
    "confidence": 0.0
  },
  "due_date_text": {
    "value": "string or null",
    "confidence": 0.0
  },
  "transaction_type": "credit | payment | unknown",
  "overall_confidence": 0.0,
  "needs_confirmation": true,
  "ambiguities": []
}

Return JSON only.
"""


RETRY_PROMPT = """
Your previous response was invalid for OpenKhata.

Return ONLY one valid JSON object matching the required schema.

Do not explain anything.
Do not use Markdown.
Do not add ```json fences.

Remember:
- baaki/udhar normally means money owed to the shop
- Friday-te dibe means payment is expected on Friday
- never silently change a monetary amount
- ambiguous amounts must use null and needs_confirmation=true
"""


# -----------------------------
# Helpers
# -----------------------------

def _extract_json(text: str) -> dict:
    """
    Parse JSON even if the model accidentally surrounds it
    with whitespace or a JSON code fence.
    """

    text = text.strip()

    # Remove accidental Markdown fences.
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return json.loads(text)


def _apply_safety_rules(result: ExtractionResult) -> ExtractionResult:
    """
    Application-level safety rules.

    These rules are trusted more than the LLM's own
    needs_confirmation field.
    """

    reasons = list(result.ambiguities)
    needs_confirmation = result.needs_confirmation

    # Amount safety.
    if result.amount_inr.value is not None:
        if result.amount_inr.value < 0:
            raise ValueError("Amount cannot be negative.")

    if result.amount_inr.confidence < 0.95:
        needs_confirmation = True

        if "Amount confidence is below the safe threshold." not in reasons:
            reasons.append(
                "Amount confidence is below the safe threshold."
            )

    # Customer safety.
    if result.customer.confidence < 0.90:
        needs_confirmation = True

        if "Customer identity needs confirmation." not in reasons:
            reasons.append(
                "Customer identity needs confirmation."
            )

    # Item safety.
    for item in result.items:
        if item.quantity is not None and item.quantity < 0:
            raise ValueError("Item quantity cannot be negative.")

    # Overall confidence.
    if result.overall_confidence < 0.90:
        needs_confirmation = True

    result.needs_confirmation = needs_confirmation
    result.ambiguities = reasons

    return result


def _validate_result(data: dict) -> ExtractionResult:
    """
    Validate model output using Pydantic, then apply
    deterministic financial safety rules.
    """

    result = ExtractionResult.model_validate(data)

    return _apply_safety_rules(result)


# -----------------------------
# Main extraction function
# -----------------------------


def _check_amount_against_transcript(
    result: ExtractionResult,
    transcript: str,
) -> ExtractionResult:
    """
    Deterministic monetary-amount safety check.

    The LLM must not silently choose between multiple spoken
    monetary amounts.
    """

    spoken = set(_spoken_amounts(transcript))

    # Remove integer item quantities such as "2 kg".
    spoken -= {
        int(item.quantity)
        for item in result.items
        if item.quantity is not None and float(item.quantity).is_integer()
    }

    if len(spoken) > 1:
        result.amount_inr.value = None
        result.amount_inr.confidence = 0.0
        result.needs_confirmation = True

        options = " / ".join(
            f"₹{n:,}" for n in sorted(spoken)
        )

        message = (
            f"Several numbers were spoken: {options}. "
            "Pick the correct amount."
        )

        if message not in result.ambiguities:
            result.ambiguities.append(message)

    elif len(spoken) == 1:
        spoken_amount = next(iter(spoken))

        if result.amount_inr.value != spoken_amount:
            result.amount_inr.value = None
            result.amount_inr.confidence = 0.0
            result.needs_confirmation = True

            message = "Extracted amount does not match the spoken number."

            if message not in result.ambiguities:
                result.ambiguities.append(message)

    return result


def _apply_transaction_type_rules(
    result: ExtractionResult,
    transcript: str,
) -> ExtractionResult:
    """
    Deterministic safety rules for future-payment language.

    A future promise must never be treated as a payment already received.
    A clear credit remains a credit.
    """

    text = transcript.lower()

    future_words = [
        "debe",
        "dibe",
        "dega",
        "dengi",
    ]

    paid_words = [
        "diyeche",
        "diye gelo",
        "diyechhe",
        "paid",
        "de diya",
        "payment koreche",
    ]

    is_future = any(word in text for word in future_words)
    is_already_paid = any(word in text for word in paid_words)

    if is_future and not is_already_paid:
        if result.transaction_type == "payment":
            result.transaction_type = "unknown"
            result.needs_confirmation = True

            message = (
                "This sounds like a future promise to pay, "
                "not a payment already received."
            )

            if message not in result.ambiguities:
                result.ambiguities.append(message)

    return result


def _spoken_amounts(transcript: str) -> list[int]:
    """Numbers in the transcript, ignoring quantities like '2 kg'."""
    cleaned = re.sub(
        r"\d[\d,]*\s*(?:kgs?|gms?|grams?|g|litres?|liters?|ltr|l|ml|pcs|pieces?|packets?|dozen|seer)\b",
        " ",
        transcript,
        flags=re.IGNORECASE,
    )
    return sorted({
        int(n.replace(",", ""))
        for n in re.findall(r"\d[\d,]*", cleaned)
    })


def extract_transaction(
    transcript: str,
    model: str = MODEL,
) -> ExtractionResult:
    """
    Extract a structured transaction from a shopkeeper transcript.

    The model gets at most one retry if its first response is
    invalid JSON or fails schema validation.

    Nothing is written to the ledger here.
    """

    if not transcript.strip():
        raise ValueError("Transcript cannot be empty.")

    user_prompt = f"""
Extract the transaction from this shopkeeper transcript.

TRANSCRIPT:
{transcript}

Remember:
Extract what was said.
Do not reinterpret it.
Return JSON only.
"""

    # -------------------------
    # Attempt 1
    # -------------------------

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        options={
            "temperature": 0,
        },
    )

    raw_response = response["message"]["content"]

    try:
        data = _extract_json(raw_response)

        result = _validate_result(data)
        result = _check_amount_against_transcript(result, transcript)
        result = _apply_transaction_type_rules(result, transcript)

        return result

    except (json.JSONDecodeError, ValidationError, ValueError):
        pass

    # -------------------------
    # Attempt 2 — retry
    # -------------------------

    retry_response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
            {
                "role": "assistant",
                "content": raw_response,
            },
            {
                "role": "user",
                "content": RETRY_PROMPT,
            },
        ],
        options={
            "temperature": 0,
        },
    )

    retry_raw = retry_response["message"]["content"]

    try:
        data = _extract_json(retry_raw)

        result = _validate_result(data)
        result = _check_amount_against_transcript(result, transcript)
        result = _apply_transaction_type_rules(result, transcript)

        return result

    except (json.JSONDecodeError, ValidationError, ValueError):
        return ExtractionResult(
            customer=CustomerField(
                value=None,
                confidence=0.0,
            ),
            amount_inr=AmountField(
                value=None,
                confidence=0.0,
            ),
            due_date_text=DueDateField(
                value=None,
                confidence=0.0,
            ),
            transaction_type="unknown",
            overall_confidence=0.0,
            needs_confirmation=True,
            ambiguities=[
                "Could not read this automatically. "
                "Please enter the details manually."
            ],
        )


if __name__ == "__main__":
    example = "Ramesh-da 2 kg rice, 340 baaki, Friday-te dibe"

    result = extract_transaction(example)

    print(result.model_dump_json(indent=2))
