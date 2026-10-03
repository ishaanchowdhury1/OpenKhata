from urllib.parse import quote


def build_reminder_message(
    name: str,
    amount: int,
) -> str:
    """
    Create a polite WhatsApp reminder using the customer's exact name.
    """

    return (
        f"Namaste {name}, your pending khata balance is "
        f"₹{amount:,}. Please pay when convenient. Thank you."
    )


def build_reminder_link(
    name: str,
    amount: int,
    phone: str | None = None,
) -> str:
    message = build_reminder_message(
        name=name,
        amount=amount,
    )

    encoded_message = quote(message)

    if phone:
        clean_phone = "".join(
            character
            for character in phone
            if character.isdigit()
        )

        if clean_phone:
            return (
                f"https://wa.me/{clean_phone}"
                f"?text={encoded_message}"
            )

    return (
        f"https://wa.me/?text={encoded_message}"
    )
