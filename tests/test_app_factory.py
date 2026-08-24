from fastapi.testclient import TestClient

from app.main import create_app


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
