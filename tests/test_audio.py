import base64
import io

import numpy as np
import pytest
import soundfile as sf

from app.audio import (
    CONTENT_TYPES,
    concat,
    encode,
    pcm_f32_base64,
    to_int16,
    to_mp3_bytes,
    to_wav_bytes,
)
from app.types import SAMPLE_RATE


def tone(seconds=0.25, freq=440.0):
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_wav_bytes_are_a_readable_riff_file():
    data = to_wav_bytes(tone())
    assert data[:4] == b"RIFF"
    audio, rate = sf.read(io.BytesIO(data), dtype="float32")
    assert rate == SAMPLE_RATE
    assert audio.ndim == 1
    assert len(audio) == len(tone())


def test_wav_round_trip_preserves_the_signal():
    original = tone()
    audio, _ = sf.read(io.BytesIO(to_wav_bytes(original)), dtype="float32")
    # PCM_16 quantization only; 1/32768 of full scale is the error bound.
    assert np.max(np.abs(audio - original)) < 1e-3


def test_mp3_bytes_start_with_a_frame_sync_or_id3_tag():
    data = to_mp3_bytes(tone(seconds=0.5))
    assert len(data) > 100
    is_id3 = data[:3] == b"ID3"
    is_frame_sync = data[0] == 0xFF and (data[1] & 0xE0) == 0xE0
    assert is_id3 or is_frame_sync


def test_to_int16_clips_instead_of_wrapping():
    loud = np.array([2.0, -2.0, 0.0], dtype=np.float32)
    assert to_int16(loud).tolist() == [32767, -32767, 0]
    assert to_int16(loud).dtype == np.int16


def test_encode_returns_bytes_and_content_type():
    for fmt in ("wav", "mp3"):
        data, content_type = encode(tone(), fmt)
        assert isinstance(data, bytes) and data
        assert content_type == CONTENT_TYPES[fmt]
    assert CONTENT_TYPES["wav"] == "audio/wav"
    assert CONTENT_TYPES["mp3"] == "audio/mpeg"


def test_encode_rejects_unknown_formats():
    with pytest.raises(ValueError) as excinfo:
        encode(tone(), "flac")
    assert "flac" in str(excinfo.value)


def test_pcm_f32_base64_round_trips_little_endian_float32():
    original = tone(seconds=0.05)
    decoded = np.frombuffer(
        base64.b64decode(pcm_f32_base64(original)), dtype="<f4"
    )
    assert np.array_equal(decoded, original)


def test_concat_joins_chunks_and_handles_the_empty_case():
    joined = concat([tone(0.1), tone(0.1)])
    assert len(joined) == len(tone(0.1)) * 2
    assert joined.dtype == np.float32
    empty = concat([])
    assert len(empty) == 0
    assert empty.dtype == np.float32
