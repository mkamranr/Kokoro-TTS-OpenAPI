from typing import Literal, Optional

from pydantic import BaseModel, Field


class TtsRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    lang: Optional[Literal["a", "b"]] = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    format: Literal["wav", "mp3"] = "wav"
    include_timestamps: bool = False


class SpeechRequest(BaseModel):
    """OpenAI's /v1/audio/speech body. `model` is accepted and ignored."""

    input: str
    model: str = "kokoro"
    voice: Optional[str] = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    response_format: str = "wav"
