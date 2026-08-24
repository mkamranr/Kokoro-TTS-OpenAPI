"""Request validation that raises ApiError, keeping routes free of error shaping."""
import numpy as np

from app.audio import encode
from app.errors import ApiError
from app.voices import DEFAULT_VOICE_ID, canonical_spec, lang_for, parse_voice_spec


def clean_text(text: str, max_chars: int) -> str:
    stripped = (text or "").strip()
    if not stripped:
        raise ApiError(400, "Text is empty")
    if len(stripped) > max_chars:
        raise ApiError(
            400,
            f"Text is {len(stripped)} characters; the limit is {max_chars}",
        )
    return stripped


def resolve_voice(spec: str | None, default: str) -> tuple[str, str]:
    """Normalize a voice spec and derive its language code.

    Returns (engine_spec, lang). A single voice keeps its plain id so Kokoro's
    own voice cache is used; blends become the canonical weighted form, which
    parse_voice_spec can read back.
    """
    raw = (spec or default or DEFAULT_VOICE_ID).strip()
    try:
        components = parse_voice_spec(raw)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from None
    engine_spec = (
        components[0].voice_id if len(components) == 1 else canonical_spec(components)
    )
    return engine_spec, lang_for(components)


def encode_or_400(
    audio: np.ndarray, fmt: str, sample_rate: int
) -> tuple[bytes, str]:
    try:
        return encode(audio, fmt, sample_rate)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from None
