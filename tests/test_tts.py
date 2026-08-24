import base64
import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.config import Settings
from app.deps import get_settings_dep
from app.main import create_app


def test_wav_response_is_playable_audio(client):
    resp = client.post("/tts", json={"text": "hello there", "voice": "af_heart"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content[:4] == b"RIFF"
    audio, rate = sf.read(io.BytesIO(resp.content), dtype="float32")
    assert rate == 24000
    assert len(audio) > 0


def test_duration_header_is_present(client):
    resp = client.post("/tts", json={"text": "hello there"})
    assert float(resp.headers["x-audio-duration"]) > 0


def test_mp3_format(client):
    resp = client.post("/tts", json={"text": "hello", "format": "mp3"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"


def test_default_voice_is_used_when_omitted(client, fake_engine):
    client.post("/tts", json={"text": "hello"})
    assert fake_engine.calls[0][1] == "af_heart"


def test_timestamps_response_carries_words_and_base64_audio(client):
    resp = client.post(
        "/tts",
        json={"text": "one two\nthree four", "include_timestamps": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["sample_rate"] == 24000
    assert body["voice"] == "af_heart"
    assert body["format"] == "wav"
    assert [w["word"] for w in body["words"]] == ["one", "two", "three", "four"]

    starts = [w["start"] for w in body["words"]]
    assert starts == sorted(starts)
    # Segment two must be offset past segment one, not restarted at zero.
    assert body["words"][2]["start"] > body["words"][1]["start"]

    decoded = base64.b64decode(body["audio"])
    assert decoded[:4] == b"RIFF"
    audio, _ = sf.read(io.BytesIO(decoded), dtype="float32")
    assert abs(len(audio) / 24000 - body["duration"]) < 0.01


def test_blend_spec_is_passed_through_normalized(client, fake_engine):
    resp = client.post(
        "/tts", json={"text": "hi", "voice": "af_bella:3,af_sky:1"}
    )
    assert resp.status_code == 200
    assert fake_engine.calls[0][1] == "af_bella:0.7500,af_sky:0.2500"


def test_lang_defaults_from_the_voice_but_can_be_overridden(client, fake_engine):
    client.post("/tts", json={"text": "hi", "voice": "bm_george"})
    assert fake_engine.calls[0][2] == "b"
    client.post("/tts", json={"text": "hi", "voice": "bm_george", "lang": "a"})
    assert fake_engine.calls[1][2] == "a"


def test_empty_text_is_rejected(client):
    resp = client.post("/tts", json={"text": "   "})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"
    assert "empty" in resp.json()["error"]["message"].lower()


def test_text_over_the_limit_is_rejected(app_with_fake_engine):
    app_with_fake_engine.dependency_overrides[get_settings_dep] = lambda: Settings(
        max_chars=10
    )
    client = TestClient(app_with_fake_engine, raise_server_exceptions=False)
    resp = client.post("/tts", json={"text": "x" * 11})
    assert resp.status_code == 400
    assert "limit is 10" in resp.json()["error"]["message"]


def test_unknown_voice_is_rejected(client):
    resp = client.post("/tts", json={"text": "hi", "voice": "af_nope"})
    assert resp.status_code == 400
    assert "Unknown voice" in resp.json()["error"]["message"]


def test_out_of_range_speed_is_rejected_in_the_error_envelope(client):
    resp = client.post("/tts", json={"text": "hi", "speed": 9.0})
    assert resp.status_code == 400
    assert "error" in resp.json()
    assert "speed" in resp.json()["error"]["message"]


def test_unsupported_format_is_rejected(client):
    resp = client.post("/tts", json={"text": "hi", "format": "flac"})
    assert resp.status_code == 400


def test_requests_before_the_model_loads_get_503():
    client = TestClient(create_app(), raise_server_exceptions=False)
    resp = client.post("/tts", json={"text": "hi"})
    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "service_unavailable"
