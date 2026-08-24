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


def test_all_synthesis_routes_are_protected(secured):
    assert secured.post("/tts/stream", json={"text": "hi"}).status_code == 401
    assert secured.post("/v1/audio/speech", json={"input": "hi"}).status_code == 401
    assert secured.get("/voices").status_code == 401


def test_health_is_always_reachable(secured):
    assert secured.get("/health").status_code == 200
