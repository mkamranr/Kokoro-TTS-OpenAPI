import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app, validate_default_voice


@pytest.fixture
def fresh_settings():
    """get_settings is lru_cached; a test that changes the env must not leak."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_factory_defaults_to_not_loading_the_model():
    """The fast suite must never pull in torch via the lifespan handler."""
    app = create_app()
    assert app.state.load_model is False
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "loading"


def test_routes_are_registered():
    # This installed FastAPI (0.141.1) resolves `include_router` lazily: the
    # objects appended to `app.routes` are `_IncludedRouter` wrappers with no
    # `.path`, so the brief's literal `{route.path for route in app.routes}`
    # can't see included-router paths on this version. The OpenAPI schema
    # reflects the fully-resolved route table (it's what /docs renders from),
    # so it is used here as the version-independent way to assert the same
    # intent: that these paths are registered.
    paths = set(create_app().openapi()["paths"].keys())
    assert {"/health", "/tts", "/tts/stream", "/voices", "/v1/audio/speech"} <= paths


def test_a_typo_in_the_default_voice_fails_fast(monkeypatch, fresh_settings):
    """Otherwise the misconfiguration hides behind a per-request 400.

    An unvalidated KOKORO_DEFAULT_VOICE makes every request that omits a voice
    fail with "Unknown voice", which reads like the caller's fault. Startup is
    where the bad value should surface, named.
    """
    monkeypatch.setenv("KOKORO_DEFAULT_VOICE", "af_hart")
    with pytest.raises(RuntimeError) as excinfo:
        with TestClient(create_app()):
            pass
    message = str(excinfo.value)
    assert "af_hart" in message
    assert "KOKORO_DEFAULT_VOICE" in message
    assert "af_heart" in message  # the valid ids are listed


def test_the_default_default_voice_is_valid(fresh_settings):
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200


def test_validate_default_voice_accepts_every_catalog_id():
    from app.voices import VOICES

    for voice in VOICES:
        validate_default_voice(voice.id)


def test_validate_default_voice_rejects_a_blend_spec():
    """KOKORO_DEFAULT_VOICE is a single id; blends are a per-request feature."""
    with pytest.raises(RuntimeError):
        validate_default_voice("af_bella:0.6,af_sky:0.4")
