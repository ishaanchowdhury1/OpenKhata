import uuid
from pathlib import Path

import streamlit as st

from app.ledger import (
    init_db,
    get_all_customers,
    add_transaction,
    get_or_create_customer,
    update_customer_phone,
)
from app.whisper_transcriber import transcribe_audio
from app.gemma_extractor import extract_transaction
from app.whatsapp import build_reminder_link


# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------

init_db()

AUDIO_DIR = Path("data/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="OpenKhata",
    page_icon="📒",
    layout="wide",
)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "entry_version" not in st.session_state:
    st.session_state.entry_version = 0

if "success_message" not in st.session_state:
    st.session_state.success_message = None

if "voice_version" not in st.session_state:
    st.session_state.voice_version = 0

if "voice_result" not in st.session_state:
    st.session_state.voice_result = None

if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""

if "voice_extraction_id" not in st.session_state:
    st.session_state.voice_extraction_id = None


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("📒 OpenKhata")
st.caption("Local-first khata ledger")

if st.session_state.success_message:
    st.success(st.session_state.success_message)
    st.session_state.success_message = None


tab1, tab2, tab3 = st.tabs([
    "📝 Quick Entry",
    "🎙️ Voice Entry",
    "📒 Ledger",
])


# =========================================================
# QUICK ENTRY
# =========================================================

with tab1:

    st.subheader("Add transaction")

    version = st.session_state.entry_version

    customers = get_all_customers()

    customer_options = ["➕ New customer"] + [
        customer["name"]
        for customer in customers
    ]

    selected_customer = st.selectbox(
        "Customer",
        customer_options,
        key=f"customer_select_{version}",
    )

    if selected_customer == "➕ New customer":

        customer_name = st.text_input(
            "Customer name",
            key=f"new_customer_name_{version}",
        )

        phone = st.text_input(
            "WhatsApp number (optional)",
            key=f"new_customer_phone_{version}",
            placeholder="919876543210",
        )

    else:

        customer_name = selected_customer

        customer = next(
            c for c in customers
            if c["name"] == selected_customer
        )

        phone = st.text_input(
            "WhatsApp number (optional)",
            value=customer["phone"] or "",
            key=f"phone_{customer['id']}_{version}",
            placeholder="919876543210",
        )

    transaction_type = st.selectbox(
        "Transaction type",
        [
            "Select transaction type",
            "Credit — customer owes shop",
            "Payment — customer paid shop",
        ],
        key=f"transaction_type_{version}",
    )

    amount = st.number_input(
        "Amount (₹)",
        min_value=0,
        value=0,
        step=10,
        key=f"amount_{version}",
    )

    description = st.text_input(
        "Items / note",
        placeholder="e.g. 2 kg rice",
        key=f"description_{version}",
    )

    due_date = st.text_input(
        "Due date / payment promise (optional)",
        placeholder="e.g. Friday",
        key=f"due_date_{version}",
    )

    if st.button(
        "Save transaction",
        type="primary",
        use_container_width=True,
        key=f"save_transaction_{version}",
    ):

        if not customer_name.strip():

            st.error("Enter a customer name.")

        elif transaction_type == "Select transaction type":

            st.error("Select Credit or Payment.")

        elif amount <= 0:

            st.error("Enter an amount greater than ₹0.")

        else:

            actual_type = (
                "credit"
                if transaction_type.startswith("Credit")
                else "payment"
            )

            customer_id = get_or_create_customer(
                customer_name.strip()
            )

            if phone.strip():
                update_customer_phone(
                    customer_id,
                    phone.strip(),
                )

            add_transaction(
                customer_name=customer_name.strip(),
                transaction_type=actual_type,
                amount_inr=int(amount),
                description=description.strip() or None,
                due_date_text=due_date.strip() or None,
            )

            st.session_state.success_message = (
                f"Saved ₹{int(amount):,} for "
                f"{customer_name.strip()}."
            )

            st.session_state.entry_version += 1
            st.rerun()


# =========================================================
# VOICE ENTRY
# =========================================================

with tab2:

    st.subheader("🎙️ Voice entry")

    st.info(
        "Record a short shop-style sentence. "
        "Whisper transcribes it locally, then Gemma extracts "
        "the transaction. Nothing is saved until you confirm it."
    )

    voice_version = st.session_state.voice_version

    audio = st.audio_input(
        "Record a transaction",
        key=f"voice_audio_{voice_version}",
    )

    if audio is not None:

        process_key = f"process_voice_{voice_version}"

        if st.button(
            "🔎 Transcribe and extract",
            type="primary",
            use_container_width=True,
            key=process_key,
        ):

            audio_id = uuid.uuid4().hex[:12]

            suffix = Path(audio.name).suffix or ".wav"
            audio_path = AUDIO_DIR / f"voice_{audio_id}{suffix}"

            audio_path.write_bytes(audio.getvalue())

            try:

                with st.spinner(
                    "Whisper is transcribing locally..."
                ):
                    transcript = transcribe_audio(audio_path)

                if not transcript.strip():
                    st.error(
                        "No speech was detected. "
                        "Please record again, preferably for 5–10 seconds."
                    )

                else:

                    st.session_state.voice_transcript = transcript

                    with st.spinner(
                        "Gemma is extracting the transaction locally..."
                    ):
                        result = extract_transaction(transcript)

                    st.session_state.voice_result = result
                    st.session_state.voice_extraction_id = (
                        uuid.uuid4().hex[:10]
                    )

                    st.success("Extraction complete.")

                    st.rerun()

            except Exception as exc:

                st.error(
                    f"Voice processing failed: {exc}"
                )


    # -----------------------------------------------------
    # TRANSCRIPT
    # -----------------------------------------------------

    transcript = st.session_state.voice_transcript

    if transcript:

        st.markdown("### 🗣️ Transcript")

        st.code(
            transcript,
            language=None,
        )


    # -----------------------------------------------------
    # EXTRACTION REVIEW
    # -----------------------------------------------------

    result = st.session_state.voice_result

    if result is not None:

        extraction_id = (
            st.session_state.voice_extraction_id
            or "current"
        )

        st.markdown("### 🤖 Gemma extraction")

        if result.needs_confirmation:

            st.warning(
                "⚠️ This entry needs your confirmation before it "
                "can be saved."
            )

        else:

            st.success(
                "Gemma extracted this entry. "
                "Please check it before saving."
            )

        if result.ambiguities:

            st.markdown("**Things to check:**")

            for ambiguity in result.ambiguities:

                st.write(f"• {ambiguity}")


        # -------------------------------------------------
        # CUSTOMER
        # -------------------------------------------------

        customer_value = (
            result.customer.value
            if result.customer.value
            else ""
        )

        existing_customers = get_all_customers()
        existing_names = [c["name"] for c in existing_customers]

        exact = next(
            (
                name
                for name in existing_names
                if name.strip().lower()
                == customer_value.strip().lower()
            ),
            None,
        )

        pick_options = [
            "➕ New customer (type name below)"
        ] + existing_names

        pick_default = (
            pick_options.index(exact)
            if exact
            else 0
        )

        if customer_value and not exact:
            st.warning(
                f"Heard the name “{customer_value}”, "
                "which is not in the ledger. "
                "If this is an existing customer, choose them below. "
                "Otherwise a new customer will be created."
            )

        pick = st.selectbox(
            "Customer",
            pick_options,
            index=pick_default,
            key=f"voice_customer_pick_{extraction_id}",
        )

        is_new_customer = pick.startswith("➕")

        if is_new_customer:
            customer_name = st.text_input(
                "New customer name",
                value=customer_value,
                key=f"voice_customer_new_{extraction_id}",
            )

            confirm_new_customer = st.checkbox(
                "I confirm this is a NEW customer.",
                value=False,
                key=f"voice_new_customer_confirm_{extraction_id}",
            )
        else:
            customer_name = pick
            confirm_new_customer = True

        st.caption(
            f"Customer confidence: "
            f"{result.customer.confidence:.2f}"
        )


        # -------------------------------------------------
        # AMOUNT
        # -------------------------------------------------

        amount_value = (
            result.amount_inr.value
            if result.amount_inr.value is not None
            else 0
        )

        amount = st.number_input(
            "Amount (₹)",
            min_value=0,
            value=int(amount_value),
            step=10,
            key=f"voice_amount_{extraction_id}",
        )

        st.caption(
            f"Amount confidence: "
            f"{result.amount_inr.confidence:.2f}"
        )


        # -------------------------------------------------
        # TRANSACTION TYPE
        # -------------------------------------------------

        type_options = [
            "Select transaction type",
            "Credit — customer owes shop",
            "Payment — customer paid shop",
        ]

        if result.transaction_type == "credit":
            default_type_index = 1

        elif result.transaction_type == "payment":
            default_type_index = 2

        else:
            default_type_index = 0

        transaction_type = st.selectbox(
            "Transaction type",
            type_options,
            index=default_type_index,
            key=f"voice_type_{extraction_id}",
        )


        # -------------------------------------------------
        # DUE DATE
        # -------------------------------------------------

        due_date_value = (
            result.due_date_text.value
            if result.due_date_text.value
            else ""
        )

        due_date = st.text_input(
            "Due date / payment promise",
            value=due_date_value,
            placeholder="e.g. Friday",
            key=f"voice_due_date_{extraction_id}",
        )


        # -------------------------------------------------
        # DESCRIPTION
        # -------------------------------------------------

        item_descriptions = []

        for item in result.items:

            parts = []

            if item.quantity is not None:
                parts.append(str(item.quantity))

            if item.unit:
                parts.append(item.unit)

            parts.append(item.name)

            item_descriptions.append(" ".join(parts))

        default_description = ", ".join(item_descriptions)

        description = st.text_input(
            "Items / note",
            value=default_description,
            key=f"voice_description_{extraction_id}",
        )


        # -------------------------------------------------
        # OPTIONAL PHONE
        # -------------------------------------------------

        customers = get_all_customers()

        matching_customer = next(
            (
                customer
                for customer in customers
                if customer["name"].strip().lower()
                == customer_name.strip().lower()
            ),
            None,
        )

        existing_phone = (
            matching_customer["phone"]
            if matching_customer
            else ""
        )

        phone = st.text_input(
            "WhatsApp number (optional)",
            value=existing_phone or "",
            placeholder="919876543210",
            key=f"voice_phone_{extraction_id}",
        )


        # -------------------------------------------------
        # EXPLICIT CONFIRMATION
        # -------------------------------------------------

        needs_explicit_check = bool(result.needs_confirmation)

        checked = st.checkbox(
            "I checked the customer name and amount.",
            value=False,
            key=f"voice_checked_{extraction_id}",
        )


        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------

        if st.button(
            "✅ Confirm and save",
            type="primary",
            use_container_width=True,
            key=f"voice_save_{extraction_id}",
        ):

            if not customer_name.strip():

                st.error("Enter a customer name.")

            elif transaction_type == "Select transaction type":

                st.error(
                    "Select Credit or Payment. "
                    "Voice extraction cannot assume the transaction type."
                )

            elif amount <= 0:

                st.error("Enter an amount greater than ₹0.")

            elif needs_explicit_check and not checked:

                st.error(
                    "Please check the customer name and amount "
                    "before saving this flagged entry."
                )

            elif is_new_customer and not confirm_new_customer:

                st.error(
                    "Please confirm that this is a NEW customer, "
                    "or select an existing customer from the ledger."
                )

            else:

                actual_type = (
                    "credit"
                    if transaction_type.startswith("Credit")
                    else "payment"
                )

                customer_id = get_or_create_customer(
                    customer_name.strip()
                )

                if phone.strip():

                    update_customer_phone(
                        customer_id,
                        phone.strip(),
                    )

                add_transaction(
                    customer_name=customer_name.strip(),
                    transaction_type=actual_type,
                    amount_inr=int(amount),
                    description=description.strip() or None,
                    due_date_text=due_date.strip() or None,
                )

                st.session_state.success_message = (
                    f"Saved ₹{int(amount):,} for "
                    f"{customer_name.strip()}."
                )

                # Clear the voice extraction completely.
                st.session_state.voice_result = None
                st.session_state.voice_transcript = ""
                st.session_state.voice_extraction_id = None

                # New audio widget key.
                st.session_state.voice_version += 1

                st.rerun()


# =========================================================
# LEDGER
# =========================================================

with tab3:

    st.subheader("Customer ledger")

    customers = get_all_customers()

    if not customers:

        st.info("No customers yet.")

    else:

        total_outstanding = sum(
            max(customer["balance"], 0)
            for customer in customers
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Customers",
                len(customers),
            )

        with col2:

            st.metric(
                "Outstanding",
                f"₹{total_outstanding:,}",
            )

        st.divider()

        for customer in customers:

            with st.expander(
                f"{customer['name']} — "
                f"₹{customer['balance']:,}"
            ):

                st.write(
                    f"**WhatsApp:** "
                    f"{customer['phone'] or 'Not added'}"
                )

                if customer["balance"] > 0:

                    reminder_link = build_reminder_link(
                        name=customer["name"],
                        amount=int(customer["balance"]),
                        phone=customer["phone"],
                    )

                    st.link_button(
                        "💬 WhatsApp reminder",
                        reminder_link,
                        use_container_width=True,
                    )

                    st.write(
                        f"**Outstanding:** "
                        f"₹{customer['balance']:,}"
                    )

                elif customer["balance"] == 0:

                    st.write(
                        "**Balance:** Settled"
                    )

                else:

                    st.write(
                        f"**Customer credit:** "
                        f"₹{abs(customer['balance']):,}"
                    )
