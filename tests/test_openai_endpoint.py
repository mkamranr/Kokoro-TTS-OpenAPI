def test_openai_shaped_request_returns_wav(client):
    resp = client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": "Hello from OpenAI clients", "voice": "af_heart"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content[:4] == b"RIFF"


def test_openai_voice_names_are_aliased(client, fake_engine):
    client.post("/v1/audio/speech", json={"input": "hi", "voice": "shimmer"})
    assert fake_engine.calls[0][1] == "af_sky"


def test_real_kokoro_ids_still_work_here(client, fake_engine):
    client.post("/v1/audio/speech", json={"input": "hi", "voice": "bm_george"})
    assert fake_engine.calls[0][1] == "bm_george"
    assert fake_engine.calls[0][2] == "b"


def test_model_field_is_accepted_and_ignored(client):
    resp = client.post(
        "/v1/audio/speech",
        json={"model": "tts-1-hd", "input": "hi", "voice": "nova"},
    )
    assert resp.status_code == 200


def test_voice_is_optional_and_falls_back_to_the_default(client, fake_engine):
    resp = client.post("/v1/audio/speech", json={"input": "hi"})
    assert resp.status_code == 200
    assert fake_engine.calls[0][1] == "af_heart"


def test_mp3_response_format(client):
    resp = client.post(
        "/v1/audio/speech",
        json={"input": "hi", "voice": "nova", "response_format": "mp3"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"


def test_unsupported_openai_formats_are_rejected_by_name(client):
    for fmt in ("opus", "aac", "flac", "pcm"):
        resp = client.post(
            "/v1/audio/speech",
            json={"input": "hi", "response_format": fmt},
        )
        assert resp.status_code == 400, fmt
        message = resp.json()["error"]["message"]
        assert "wav" in message and "mp3" in message


def test_missing_input_is_a_400_in_the_error_envelope(client):
    resp = client.post("/v1/audio/speech", json={"voice": "nova"})
    assert resp.status_code == 400
    assert "input" in resp.json()["error"]["message"]


def test_empty_input_is_rejected(client):
    resp = client.post("/v1/audio/speech", json={"input": "  "})
    assert resp.status_code == 400


def test_speed_bounds_are_enforced(client):
    assert client.post("/v1/audio/speech", json={"input": "hi", "speed": 0.4}).status_code == 400
    assert client.post("/v1/audio/speech", json={"input": "hi", "speed": 2.5}).status_code == 400
    assert client.post("/v1/audio/speech", json={"input": "hi", "speed": 1.5}).status_code == 200


def test_unknown_voice_is_rejected(client):
    resp = client.post("/v1/audio/speech", json={"input": "hi", "voice": "sparkle"})
    assert resp.status_code == 400
    assert "Unknown voice" in resp.json()["error"]["message"]
