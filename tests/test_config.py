import os

import pytest

from app.config import Settings, resolve_concurrency


@pytest.fixture(autouse=True)
def no_ambient_kokoro_env(monkeypatch):
    """Settings() reads the real environment, so the tests must own it.

    Without this, any exported KOKORO_* variable -- one left over from running
    the server by hand, say -- breaks this file: test_defaults_match_spec would
    be asserting the shell's values, not the defaults. monkeypatch restores
    everything afterwards.
    """
    for name in [n for n in os.environ if n.upper().startswith("KOKORO_")]:
        monkeypatch.delenv(name, raising=False)


def test_defaults_match_spec():
    s = Settings()
    assert s.device == "auto"
    assert s.default_voice == "af_heart"
    assert s.max_chars == 5000
    assert s.api_key == ""
    assert s.voice_cache_size == 32
    assert s.host == "127.0.0.1"
    assert s.port == 8080


def test_env_prefix_is_honored(monkeypatch):
    monkeypatch.setenv("KOKORO_DEVICE", "cuda")
    monkeypatch.setenv("KOKORO_MAX_CHARS", "120")
    s = Settings()
    assert s.device == "cuda"
    assert s.max_chars == 120


@pytest.mark.parametrize(
    "device,configured,expected",
    [("cpu", 0, 1), ("cuda", 0, 2), ("cpu", 4, 4), ("cuda", 1, 1)],
)
def test_resolve_concurrency(device, configured, expected):
    assert resolve_concurrency(device, configured) == expected


def test_allow_origins_splits_into_list():
    s = Settings(allow_origins="http://a.test, http://b.test")
    assert s.origin_list() == ["http://a.test", "http://b.test"]


def test_allow_origins_empty_is_no_origins():
    assert Settings().origin_list() == []
