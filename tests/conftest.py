import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.service import SynthesisService
from tests.fakes import FakeEngine


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def app_with_fake_engine(fake_engine):
    app = create_app()
    app.state.service = SynthesisService(fake_engine, max_concurrency=2)
    return app


@pytest.fixture
def client(app_with_fake_engine) -> TestClient:
    return TestClient(app_with_fake_engine, raise_server_exceptions=False)
