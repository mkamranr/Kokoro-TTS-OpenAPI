import base64

from fastapi import APIRouter, Depends, Response

from app.config import Settings
from app.deps import get_service, get_settings_dep
from app.schemas import TtsRequest
from app.service import SynthesisService
from app.validation import clean_text, encode_or_400, resolve_voice
from app.voices import DEFAULT_VOICE_ID, catalog

router = APIRouter()


@router.get("/voices")
async def list_voices() -> dict:
    rows = catalog()
    return {"voices": rows, "count": len(rows), "default": DEFAULT_VOICE_ID}


@router.post("/tts")
async def tts(
    request: TtsRequest,
    service: SynthesisService = Depends(get_service),
    settings: Settings = Depends(get_settings_dep),
):
    text = clean_text(request.text, settings.max_chars)
    voice, voice_lang = resolve_voice(request.voice, settings.default_voice)
    lang = request.lang or voice_lang

    result = await service.synthesize(text, voice, lang, request.speed)
    data, content_type = encode_or_400(
        result.audio, request.format, result.sample_rate
    )

    if not request.include_timestamps:
        return Response(
            content=data,
            media_type=content_type,
            headers={
                "X-Audio-Duration": f"{result.duration:.3f}",
                "X-Voice": voice,
            },
        )

    return {
        "audio": base64.b64encode(data).decode("ascii"),
        "format": request.format,
        "sample_rate": result.sample_rate,
        "duration": round(result.duration, 3),
        "voice": voice,
        "segments": result.segments,
        "phonemes": result.phonemes,
        "words": [w.as_dict() for w in result.words],
    }
