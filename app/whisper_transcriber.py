from pathlib import Path

import torch
import whisper


MODEL_NAME = "medium"

_model = None


def get_device() -> str:
    """Use Apple's MPS GPU when available, otherwise CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model():
    """Load Whisper once and reuse it."""
    global _model

    if _model is None:
        device = get_device()
        print(f"Loading Whisper '{MODEL_NAME}' on {device}...")
        _model = whisper.load_model(MODEL_NAME, device=device)

    return _model


def transcribe_audio(audio_path: str | Path) -> str:
    """
    Transcribe an audio file locally using Whisper.

    Returns only the cleaned transcript.
    """
    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    model = get_model()

    result = model.transcribe(
        str(path),
        task="transcribe",
        language=None,
        temperature=0,
        condition_on_previous_text=False,
        fp16=False,
    )

    return result["text"].strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m app.whisper_transcriber <audio_file>")
        raise SystemExit(1)

    audio_file = sys.argv[1]

    try:
        transcript = transcribe_audio(audio_file)
        print("\nTranscript:")
        print(transcript)
    except Exception as exc:
        print(f"\nTranscription failed: {exc}")
        raise SystemExit(1)
