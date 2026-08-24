"""Encoding float32 mono audio to WAV/MP3 bytes — no ffmpeg involved."""
import base64
import io

import lameenc
import numpy as np
import soundfile as sf

from app.types import SAMPLE_RATE

CONTENT_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg"}
SUPPORTED_FORMATS = tuple(CONTENT_TYPES)
MP3_BITRATE = 128


def to_int16(audio: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, to_int16(audio), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def to_mp3_bytes(
    audio: np.ndarray, sample_rate: int = SAMPLE_RATE, bitrate: int = MP3_BITRATE
) -> bytes:
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)  # 2 = high quality, still fast
    return bytes(encoder.encode(to_int16(audio).tobytes()) + encoder.flush())


def encode(
    audio: np.ndarray, fmt: str, sample_rate: int = SAMPLE_RATE
) -> tuple[bytes, str]:
    if fmt == "wav":
        return to_wav_bytes(audio, sample_rate), CONTENT_TYPES["wav"]
    if fmt == "mp3":
        return to_mp3_bytes(audio, sample_rate), CONTENT_TYPES["mp3"]
    raise ValueError(
        f"Unsupported format '{fmt}'. Supported: {', '.join(SUPPORTED_FORMATS)}"
    )


def pcm_f32_base64(audio: np.ndarray) -> str:
    """Raw little-endian float32 PCM, base64 encoded — the streaming chunk format."""
    return base64.b64encode(
        np.asarray(audio, dtype="<f4").tobytes()
    ).decode("ascii")


def concat(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([np.asarray(c, dtype=np.float32) for c in chunks])
