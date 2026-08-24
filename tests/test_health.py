from fastapi.testclient import TestClient

from app.main import create_app


def test_health_reports_loading_before_the_engine_is_ready():
    client = TestClient(create_app())
    body = client.get("/health").json()
    assert body["status"] == "loading"
    assert body["model_loaded"] is False


def test_error_envelope_shape():
    from fastapi import FastAPI

    from app.errors import ApiError, install_error_handlers

    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    def boom():
        raise ApiError(400, "bad thing", "invalid_request_error")

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/boom")
    assert resp.status_code == 400
    assert resp.json() == {
        "error": {"message": "bad thing", "type": "invalid_request_error"}
    }
