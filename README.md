# 📒 OpenKhata

A local-first digital khata for a small kirana shop.

OpenKhata lets a shopkeeper record customer credit using typed entry or voice. AI assists with transcription and extraction, but the shopkeeper confirms the transaction before anything is written to the ledger.

## Core flow

Voice → Local Whisper → Transcript → Local Gemma 3 4B → Structured transaction → Human confirmation → SQLite

## Features

- Quick typed credit/payment entry
- Browser voice recording
- Local Whisper transcription
- Local Gemma 3 4B extraction through Ollama
- Human confirmation before saving voice transactions
- Ambiguous amounts are not silently selected
- Existing-customer selection for unrecognised spoken names
- Explicit confirmation before creating a new customer
- SQLite customer ledger
- Optional WhatsApp reminder with a pre-filled message
- No automatic WhatsApp sending

## Known limitations

Whisper medium can mishear Bengali/Hindi/English code-mixed speech and can produce names in different scripts. Voice is therefore an assisted workflow rather than an automatic write path.

## Requirements

- Python 3.11+
- FFmpeg
- Ollama
- Gemma 3 4B

## Setup

    git clone https://github.com/ishaanchowdhury1/OpenKhata.git
    cd OpenKhata
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ollama pull gemma3:4b
    python -m streamlit run app/ui.py

## Data

Customer data is stored locally in data/openkhata.db. The data directory is ignored by Git.

## Tests

    python -m pytest tests -q
