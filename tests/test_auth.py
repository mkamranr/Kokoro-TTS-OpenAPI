import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.deps import get_settings_dep


@pytest.fixture
def secured(app_with_fake_engine):
    app_with_fake_engine.dependency_overrides[get_settings_dep] = lambda: Settings(
        api_key="s3cret"
    )
    return TestClient(app_with_fake_engine, raise_server_exceptions=False)


def test_no_key_configured_means_open_access(client):
    assert client.post("/tts", json={"text": "hi"}).status_code == 200


def test_missing_header_is_401(secured):
    resp = secured.post("/tts", json={"text": "hi"})
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


def test_wrong_key_is_401(secured):
    resp = secured.post(
        "/tts", json={"text": "hi"}, headers={"Authorization": "Bearer nope"}
    )
    assert resp.status_code == 401


def test_wrong_scheme_is_401(secured):
    resp = secured.post(
        "/tts", json={"text": "hi"}, headers={"Authorization": "Basic s3cret"}
    )
    assert resp.status_code == 401


def test_correct_key_is_accepted(secured):
    resp = secured.post(
        "/tts", json={"text": "hi"}, headers={"Authorization": "Bearer s3cret"}
    )
    assert resp.status_code == 200


def test_non_ascii_key_is_401_not_500(secured):
    """Regression: hmac.compare_digest used to TypeError on non-ASCII str.

    ASGI decodes header bytes as latin-1, so the 0xe9 below arrives as "é" and
    reached compare_digest, which refuses non-ASCII str and raised -- yielding
    a 500 server_error plus a logged stack trace that any unauthenticated
    caller could trigger on demand.
    """
    # Sent as raw bytes: httpx refuses to encode a non-ASCII str header, but a
    # real client can put any byte on the wire, which is the point.
    resp = secured.post(
        "/tts", json={"text": "hi"}, headers=[(b"authorization", b"Bearer \xe9")]
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


def test_a_multibyte_non_ascii_key_is_also_401(secured):
    """A whole UTF-8 sequence, not just one stray high byte."""
    resp = secured.post(
        "/tts",
        json={"text": "hi"},
        headers=[(b"authorization", "Bearer s3creté☃".encode("utf-8"))],
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


def test_trailing_whitespace_in_the_token_is_tolerated(secured):
    """HTTP permits optional whitespace around a header value.

    The transport may strip it before we ever see it, so treating "s3cret " as
    a different key would be arbitrary. This pins the strip() in app/auth.py as
    intentional rather than incidental.
    """
    resp = secured.post(
        "/tts", json={"text": "hi"}, headers={"Authorization": "Bearer s3cret  "}
    )
    assert resp.status_code == 200


def test_internal_whitespace_still_fails(secured):
    """Stripping the ends must not make the comparison lenient in general."""
    resp = secured.post(
        "/tts", json={"text": "hi"}, headers={"Authorization": "Bearer s3c ret"}
    )
    assert resp.status_code == 401


def test_all_synthesis_routes_are_protected(secured):
    assert secured.post("/tts/stream", json={"text": "hi"}).status_code == 401
    assert secured.post("/v1/audio/speech", json={"input": "hi"}).status_code == 401
    assert secured.get("/voices").status_code == 401


def test_health_is_always_reachable(secured):
    assert secured.get("/health").status_code == 200
