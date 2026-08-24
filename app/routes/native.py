import base64
import json
import logging

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from app.audio import pcm_f32_base64
from app.config import Settings
from app.deps import get_service, get_settings_dep
from app.schemas import TtsRequest
from app.service import SynthesisService
from app.validation import clean_text, encode_or_400, resolve_voice
from app.voices import DEFAULT_VOICE_ID, catalog

logger = logging.getLogger(__name__)

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


@router.post("/tts/stream")
async def tts_stream(
    request: TtsRequest,
    service: SynthesisService = Depends(get_service),
    settings: Settings = Depends(get_settings_dep),
):
    """NDJSON stream of float32 PCM chunks with absolute word timings.

    Validation runs here, before the response starts, so bad requests still get
    a normal 400 rather than an error buried in the stream.
    """
    text = clean_text(request.text, settings.max_chars)
    voice, voice_lang = resolve_voice(request.voice, settings.default_voice)
    lang = request.lang or voice_lang

    async def lines():
        yield json.dumps(
            {
                "type": "meta",
                "sample_rate": service.sample_rate,
                "voice": voice,
                "lang": lang,
                "format": "pcm_f32le",
            }
        ) + "\n"

        total = 0.0
        count = 0
        try:
            async for segment in service.stream_segments(
                text, voice, lang, request.speed
            ):
                total += segment.duration
                count += 1
                yield json.dumps(
                    {
                        "type": "chunk",
                        "index": segment.index,
                        "audio": pcm_f32_base64(segment.audio),
                        "duration": round(segment.duration, 3),
                        "words": [w.as_dict() for w in segment.words],
                    }
                ) + "\n"
        except Exception as exc:  # noqa: BLE001 - surfaced to the client below
            logger.exception("streaming synthesis failed")
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
            return

        yield json.dumps(
            {
                "type": "done",
                "duration": round(total, 3),
                "segments": count,
            }
        ) + "\n"

    return StreamingResponse(
        lines(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
