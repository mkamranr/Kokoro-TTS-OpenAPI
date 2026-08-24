import base64
import json

import numpy as np


def read_ndjson(resp):
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def test_stream_emits_meta_chunks_and_done(client):
    resp = client.post("/tts/stream", json={"text": "one two\nthree four"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")

    events = read_ndjson(resp)
    assert events[0]["type"] == "meta"
    assert events[0]["sample_rate"] == 24000
    assert events[0]["format"] == "pcm_f32le"
    assert events[0]["voice"] == "af_heart"

    chunks = [e for e in events if e["type"] == "chunk"]
    assert [c["index"] for c in chunks] == [0, 1]

    assert events[-1]["type"] == "done"
    assert events[-1]["segments"] == 2
    assert events[-1]["duration"] > 0


def test_chunk_audio_decodes_to_float32_pcm(client):
    events = read_ndjson(client.post("/tts/stream", json={"text": "hello world"}))
    chunk = next(e for e in events if e["type"] == "chunk")
    audio = np.frombuffer(base64.b64decode(chunk["audio"]), dtype="<f4")
    assert len(audio) > 0
    assert np.max(np.abs(audio)) <= 1.0
    assert abs(len(audio) / 24000 - chunk["duration"]) < 0.01


def test_chunk_words_are_absolute_not_segment_relative(client):
    events = read_ndjson(client.post("/tts/stream", json={"text": "one two\nthree four"}))
    chunks = [e for e in events if e["type"] == "chunk"]
    first_words = chunks[0]["words"]
    second_words = chunks[1]["words"]

    assert [w["word"] for w in first_words] == ["one", "two"]
    assert [w["word"] for w in second_words] == ["three", "four"]
    # The bug this guards: segment two restarting at 0.0.
    assert second_words[0]["start"] >= first_words[-1]["end"]


def test_done_duration_matches_the_sum_of_chunks(client):
    events = read_ndjson(client.post("/tts/stream", json={"text": "a b\nc d\ne f"}))
    chunk_total = sum(e["duration"] for e in events if e["type"] == "chunk")
    done = events[-1]
    assert abs(done["duration"] - chunk_total) < 0.01
    assert done["segments"] == 3


def test_format_field_is_ignored_by_the_stream(client):
    events = read_ndjson(
        client.post("/tts/stream", json={"text": "hello", "format": "mp3"})
    )
    assert events[0]["format"] == "pcm_f32le"


def test_validation_errors_happen_before_streaming_starts(client):
    resp = client.post("/tts/stream", json={"text": "hi", "voice": "af_nope"})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_empty_text_is_rejected_before_streaming(client):
    resp = client.post("/tts/stream", json={"text": ""})
    assert resp.status_code == 400


def test_engine_failure_mid_stream_is_reported_as_an_error_line(app_with_fake_engine):
    from fastapi.testclient import TestClient

    class ExplodingEngine:
        sample_rate = 24000

        def info(self):
            return {"device": "fake"}

        def iter_segments(self, text, voice, lang, speed):
            raise RuntimeError("boom")

    from app.service import SynthesisService

    app_with_fake_engine.state.service = SynthesisService(ExplodingEngine(), 1)
    client = TestClient(app_with_fake_engine, raise_server_exceptions=False)
    events = read_ndjson(client.post("/tts/stream", json={"text": "hi"}))
    assert events[-1]["type"] == "error"
    assert "boom" in events[-1]["message"]
