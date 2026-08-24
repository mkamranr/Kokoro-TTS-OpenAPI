"""OpenAI-compatible speech endpoint, so existing OpenAI TTS clients work as-is."""
from fastapi import APIRouter, Depends, Response

from app.audio import SUPPORTED_FORMATS
from app.config import Settings
from app.deps import get_service, get_settings_dep
from app.errors import ApiError
from app.schemas import SpeechRequest
from app.service import SynthesisService
from app.validation import clean_text, encode_or_400, resolve_voice

router = APIRouter(prefix="/v1")


@router.post("/audio/speech")
async def create_speech(
    request: SpeechRequest,
    service: SynthesisService = Depends(get_service),
    settings: Settings = Depends(get_settings_dep),
):
    if request.response_format not in SUPPORTED_FORMATS:
        raise ApiError(
            400,
            f"response_format '{request.response_format}' is not supported. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}",
        )

    text = clean_text(request.input, settings.max_chars)
    voice, lang = resolve_voice(request.voice, settings.default_voice)

    result = await service.synthesize(text, voice, lang, request.speed)
    data, content_type = encode_or_400(
        result.audio, request.response_format, result.sample_rate
    )
    return Response(
        content=data,
        media_type=content_type,
        headers={"X-Audio-Duration": f"{result.duration:.3f}", "X-Voice": voice},
    )
