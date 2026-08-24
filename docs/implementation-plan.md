# Kokoro TTS API Implementation Plan


**Goal:** A self-hosted HTTP API + web UI that synthesizes English speech with Kokoro-82M, running CPU-native on an Intel Mac and as a CUDA container on a Windows/RTX 2070 host.

**Architecture:** FastAPI service where `app/engine.py` is the only module importing `torch`/`kokoro`; every other module handles plain numpy arrays and dataclasses, so the whole route/encode/timing layer is tested against a fake engine with no model load. Synthesis runs in worker threads behind a semaphore; long text streams segment-by-segment as NDJSON, which is also how the browser UI drives karaoke-style word highlighting.

**Tech Stack:** Python 3.10, FastAPI, uvicorn, pydantic v2 + pydantic-settings, PyTorch (2.2.2 on Mac / 2.5.1 in CUDA image), kokoro 0.9.4, misaki[en], soundfile (WAV), lameenc (MP3), numpy, pytest + httpx. Frontend is vanilla HTML/CSS/JS — no npm, no CDN.

**Spec:** `docs/design.md`

## Global Constraints

These apply to every task. Values are copied verbatim from the spec.

- **Python** 3.10–3.12 (`kokoro` requires `>=3.10,<3.13`). The Mac has 3.10.4.
- **Mac dependency pins are load-bearing:** `torch==2.2.2` (last `macosx_10_9_x86_64` cp310 wheel; macOS here is 12.7.6, x86_64) and `numpy<2` (torch 2.2.2 was built against numpy 1.x). Do not float either.
- **No ffmpeg** anywhere. WAV via `soundfile`, MP3 via `lameenc`. Both have Intel-macOS cp310 wheels.
- **No ONNX backend.** `onnxruntime` macOS wheels require macOS 13+ from 1.20 onward and `kokoro-onnx` needs `>=1.20.1`; Monterey cannot satisfy it.
- **Sample rate is always 24000 Hz**, mono, float32 internally.
- **Only `app/engine.py` may import `torch` or `kokoro`.** Any other module doing so is a review rejection.
- **The default test suite must not load the model** and must run in seconds. Model-loading tests are marked `slow` and gated behind `KOKORO_RUN_SLOW=1`.
- **All errors use the OpenAI envelope:** `{"error": {"message": ..., "type": ...}}`.
- **Config is env-driven with prefix `KOKORO_`**, every setting has a default.
- **English only:** 28 voices, lang codes `a` (US) and `b` (UK).
- **UI has no build step**, no npm, no CDN — it must work offline inside the container.
- **Commit after every task.** Conventional-commit prefixes (`feat:`, `test:`, `chore:`, `docs:`).

## File Structure

| File | Responsibility |
|---|---|
| `app/__init__.py` | Package marker. |
| `app/config.py` | `Settings` (pydantic-settings), `get_settings()`, concurrency/thread resolution. |
| `app/errors.py` | `ApiError` + handlers producing the OpenAI error envelope. |
| `app/types.py` | `WordTiming`, `Segment`, `Synthesis`, `EngineProtocol`. No torch. |
| `app/timing.py` | `words_from_tokens()`, `TimelineAccumulator` — segment-relative → absolute timings. No torch. |
| `app/audio.py` | float32 → WAV/MP3 bytes, base64 PCM, content types. No torch. |
| `app/voices.py` | 28-voice catalog, OpenAI aliases, blend-spec parsing. No torch. |
| `app/engine.py` | `KokoroEngine` — the only torch/kokoro importer. |
| `app/service.py` | `SynthesisService` — semaphore + thread offloading around any engine. |
| `app/auth.py` | Optional bearer-token dependency. |
| `app/deps.py` | `get_service()`, `get_settings_dep()` FastAPI dependencies. |
| `app/routes/health.py` | `GET /health`. |
| `app/routes/native.py` | `POST /tts`, `POST /tts/stream`, `GET /voices`. |
| `app/routes/openai.py` | `POST /v1/audio/speech`. |
| `app/main.py` | App factory, lifespan model load, router + static mounting. |
| `app/__main__.py` | `python -m app` launcher that honors `KOKORO_HOST`/`KOKORO_PORT`. |
| `app/schemas.py` | `TtsRequest`, `SpeechRequest` pydantic request models. |
| `app/validation.py` | Text/voice/format validation raising `ApiError`. |
| `web/index.html`, `web/styles.css`, `web/app.js` | The UI. |
| `tests/fakes.py` | `FakeEngine` + token stubs used by every fast test. |
| `tests/conftest.py` | `client` fixture with the fake engine injected. |
| `scripts/setup_mac.sh` | Idempotent Mac CPU install + verification. |
| `scripts/bake_assets.py` | Pre-download weights + all English voices (used by Docker build and Mac setup). |
| `docker/Dockerfile.cuda`, `docker/docker-compose.gpu.yml` | GPU deployment. |
| `requirements-base.txt`, `requirements-dev.txt`, `requirements-mac-cpu.txt`, `requirements-gpu.txt` | Dependency sets. |

---

### Task 1: Scaffold, config, error envelope, health endpoint

**Files:**
- Create: `app/__init__.py`, `app/config.py`, `app/errors.py`, `app/deps.py`, `app/routes/__init__.py`, `app/routes/health.py`, `app/main.py`
- Create: `requirements-base.txt`, `requirements-dev.txt`, `pytest.ini`
- Test: `tests/__init__.py`, `tests/test_config.py`, `tests/test_health.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `Settings` with fields `device, default_voice, max_chars, max_concurrency, torch_threads, api_key, voice_cache_size, host, port, allow_origins`; `get_settings() -> Settings`; `resolve_concurrency(device: str, configured: int) -> int`; `ApiError(status_code: int, message: str, type_: str = "invalid_request_error")`; `install_error_handlers(app)`; `create_app() -> FastAPI`; `app` module-level instance in `app.main`.

- [ ] **Step 1: Create the virtualenv and dependency files**

`requirements-base.txt` (no torch — the fast suite must not need it):

```
fastapi>=0.115,<1
uvicorn[standard]>=0.30
pydantic>=2.7,<3
pydantic-settings>=2.3,<3
numpy<2
soundfile>=0.12
lameenc>=1.7
huggingface-hub>=0.25
```

`requirements-dev.txt`:

```
-r requirements-base.txt
pytest>=8
pytest-asyncio>=0.23
httpx>=0.27
```

Run:

```bash
cd .
python3.10 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-dev.txt
```

`pytest.ini`:

```ini
[pytest]
testpaths = tests
markers =
    slow: loads the real Kokoro model; runs only when KOKORO_RUN_SLOW=1
filterwarnings =
    ignore::DeprecationWarning
```

- [ ] **Step 2: Write the failing config test**

`tests/test_config.py`:

```python
import pytest

from app.config import Settings, resolve_concurrency


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
```

- [ ] **Step 3: Run it to confirm it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 4: Implement config**

`app/__init__.py`: empty file.

`app/config.py`:

```python
"""Environment-driven settings. Every value has a default."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KOKORO_", case_sensitive=False, extra="ignore"
    )

    device: str = "auto"
    default_voice: str = "af_heart"
    max_chars: int = 5000
    # 0 means "decide from the device": 1 on CPU, 2 on CUDA.
    max_concurrency: int = 0
    # 0 means "leave torch's default alone".
    torch_threads: int = 0
    api_key: str = ""
    voice_cache_size: int = 32
    host: str = "127.0.0.1"
    port: int = 8080
    allow_origins: str = ""

    def origin_list(self) -> list[str]:
        return [o.strip() for o in self.allow_origins.split(",") if o.strip()]


def resolve_concurrency(device: str, configured: int) -> int:
    if configured > 0:
        return configured
    return 2 if device == "cuda" else 1


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 5: Run the config test**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS (5 tests, one parametrized into 4 cases)

- [ ] **Step 6: Write the failing health test**

`tests/__init__.py`: empty file.

`tests/test_health.py`:

```python
from fastapi.testclient import TestClient

from app.main import create_app


def test_health_reports_loading_before_the_engine_is_ready():
    client = TestClient(create_app())
    body = client.get("/health").json()
    assert body["status"] == "loading"
    assert body["model_loaded"] is False


def test_error_envelope_shape():
    app = create_app()

    from app.errors import ApiError

    @app.get("/boom")
    def boom():
        raise ApiError(400, "bad thing", "invalid_request_error")

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/boom")
    assert resp.status_code == 400
    assert resp.json() == {
        "error": {"message": "bad thing", "type": "invalid_request_error"}
    }
```

- [ ] **Step 7: Run it to confirm it fails**

Run: `.venv/bin/pytest tests/test_health.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 8: Implement errors, deps, health route, and the app factory**

`app/errors.py`:

```python
"""All error responses use the OpenAI envelope: {"error": {"message", "type"}}."""
import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(
        self, status_code: int, message: str, type_: str = "invalid_request_error"
    ):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.type = type_


def envelope(message: str, type_: str) -> dict:
    return {"error": {"message": message, "type": type_}}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return JSONResponse(
            status_code=exc.status_code, content=envelope(exc.message, exc.type)
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ())[1:]) or "body"
        return JSONResponse(
            status_code=400,
            content=envelope(f"{where}: {first.get('msg')}", "invalid_request_error"),
        )

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        request_id = uuid.uuid4().hex[:12]
        logger.exception("unhandled error request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content=envelope(
                f"Internal error (request_id={request_id})", "server_error"
            ),
        )
```

`app/deps.py`:

```python
from fastapi import Request

from app.config import Settings, get_settings
from app.errors import ApiError


def get_settings_dep() -> Settings:
    return get_settings()


def get_service(request: Request):
    """The SynthesisService installed by the lifespan handler (or by tests)."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise ApiError(
            503, "Model is still loading; retry shortly", "service_unavailable"
        )
    return service
```

`app/routes/__init__.py`: empty file.

`app/routes/health.py`:

```python
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    service = getattr(request.app.state, "service", None)
    if service is None:
        return {"status": "loading", "model_loaded": False}
    info = service.info()
    return {"status": "ok", "model_loaded": True, **info}
```

`app/main.py`:

```python
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.errors import install_error_handlers
from app.routes import health

logging.basicConfig(level=logging.INFO)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Kokoro TTS API", version="1.0.0")
    install_error_handlers(app)
    origins = settings.origin_list()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.include_router(health.router)
    return app


app = create_app()
```

- [ ] **Step 9: Run the health test**

Run: `.venv/bin/pytest tests/test_health.py -v`
Expected: PASS (2 tests)

- [ ] **Step 10: Confirm the server actually boots**

Run:

```bash
.venv/bin/python -c "
from fastapi.testclient import TestClient
from app.main import app
print(TestClient(app).get('/health').json())
"
```

Expected: `{'status': 'loading', 'model_loaded': False}`

- [ ] **Step 11: Commit**

```bash
git add app tests requirements-base.txt requirements-dev.txt pytest.ini
git commit -m "feat: scaffold FastAPI app with settings, error envelope, health endpoint"
```

---

### Task 2: Voice catalog, OpenAI aliases, blend-spec parsing

**Files:**
- Create: `app/voices.py`
- Test: `tests/test_voices.py`

**Interfaces:**
- Consumes: `ApiError` is *not* used here — this module raises `ValueError`; routes translate.
- Produces: `Voice(id, name, gender, accent, lang, grade, default)`; `VOICES: tuple[Voice, ...]` (28 entries); `VOICES_BY_ID: dict[str, Voice]`; `OPENAI_ALIASES: dict[str, str]`; `DEFAULT_VOICE_ID = "af_heart"`; `MAX_BLEND_COMPONENTS = 4`; `BlendComponent(voice_id: str, weight: float)`; `resolve_alias(name: str) -> str`; `parse_voice_spec(spec: str) -> list[BlendComponent]`; `canonical_spec(components) -> str`; `lang_for(components) -> str`; `catalog() -> list[dict]`.

- [ ] **Step 1: Write the failing test**

`tests/test_voices.py`:

```python
import pytest

from app.voices import (
    MAX_BLEND_COMPONENTS,
    OPENAI_ALIASES,
    VOICES,
    VOICES_BY_ID,
    canonical_spec,
    catalog,
    lang_for,
    parse_voice_spec,
    resolve_alias,
)


def test_catalog_has_all_28_english_voices():
    assert len(VOICES) == 28
    assert len({v.id for v in VOICES}) == 28
    assert sum(1 for v in VOICES if v.lang == "a") == 20
    assert sum(1 for v in VOICES if v.lang == "b") == 8


def test_af_heart_is_the_single_default():
    defaults = [v.id for v in VOICES if v.default]
    assert defaults == ["af_heart"]


def test_every_id_matches_its_lang_and_gender_prefix():
    for v in VOICES:
        assert v.id[0] == v.lang
        assert v.id[1] == ("f" if v.gender == "female" else "m")
        assert v.accent == ("American" if v.lang == "a" else "British")


def test_catalog_dicts_are_json_ready():
    rows = catalog()
    assert len(rows) == 28
    assert set(rows[0]) == {
        "id", "name", "gender", "accent", "lang", "grade", "default",
    }


def test_openai_aliases_point_at_real_voices():
    assert set(OPENAI_ALIASES) == {
        "alloy", "echo", "fable", "onyx", "nova", "shimmer",
    }
    for target in OPENAI_ALIASES.values():
        assert target in VOICES_BY_ID
    assert resolve_alias("shimmer") == "af_sky"
    assert resolve_alias("af_bella") == "af_bella"


def test_single_voice_spec():
    comps = parse_voice_spec("af_bella")
    assert len(comps) == 1
    assert comps[0].voice_id == "af_bella"
    assert comps[0].weight == pytest.approx(1.0)


def test_unweighted_blend_averages_equally():
    comps = parse_voice_spec("af_bella,af_sky")
    assert [c.voice_id for c in comps] == ["af_bella", "af_sky"]
    assert [c.weight for c in comps] == pytest.approx([0.5, 0.5])


def test_weighted_blend_normalizes_to_one():
    comps = parse_voice_spec("af_bella:3,af_sky:1")
    assert [c.weight for c in comps] == pytest.approx([0.75, 0.25])


def test_weights_that_already_sum_to_one_are_preserved():
    comps = parse_voice_spec("af_bella:0.6,af_sky:0.4")
    assert [c.weight for c in comps] == pytest.approx([0.6, 0.4])


def test_alias_inside_a_blend_is_resolved():
    assert [c.voice_id for c in parse_voice_spec("nova,onyx")] == [
        "af_nova", "am_onyx",
    ]


def test_canonical_spec_is_stable_for_caching():
    assert canonical_spec(parse_voice_spec("af_bella:0.6,af_sky:0.4")) == (
        "af_bella:0.6000,af_sky:0.4000"
    )


def test_lang_comes_from_the_first_component():
    assert lang_for(parse_voice_spec("af_bella")) == "a"
    assert lang_for(parse_voice_spec("bm_george,af_bella")) == "b"


@pytest.mark.parametrize(
    "spec,fragment",
    [
        ("", "empty"),
        ("   ", "empty"),
        ("nope", "Unknown voice"),
        ("af_bella,nope", "Unknown voice"),
        ("af_bella:0.5,af_sky", "either all"),
        ("af_bella:0", "greater than 0"),
        ("af_bella:-1", "greater than 0"),
        ("af_bella:abc", "not a number"),
        ("af_bella,af_sky,af_nova,am_puck,am_echo", "at most 4"),
        ("zf_xiaobei", "Unknown voice"),
    ],
)
def test_invalid_specs_raise_valueerror(spec, fragment):
    with pytest.raises(ValueError) as excinfo:
        parse_voice_spec(spec)
    assert fragment in str(excinfo.value)


def test_blend_component_cap_constant():
    assert MAX_BLEND_COMPONENTS == 4
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `.venv/bin/pytest tests/test_voices.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.voices'`

- [ ] **Step 3: Implement the catalog and parser**

Grades below are transcribed from upstream `VOICES.md` (Overall Grade column).
`af_heart` is absent from that table but ships as `voices/af_heart.pt`; it is
Kokoro's own README default, recorded here as grade `A` and the service default.

`app/voices.py`:

```python
"""The 28 English Kokoro voices, OpenAI name aliases, and blend-spec parsing.

A blend spec is either a single voice id ("af_bella"), a comma-separated list
to average equally ("af_bella,af_sky"), or weighted components
("af_bella:0.6,af_sky:0.4"). Weights are normalized to sum to 1.
"""
from dataclasses import dataclass

DEFAULT_VOICE_ID = "af_heart"
MAX_BLEND_COMPONENTS = 4


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    gender: str
    accent: str
    lang: str
    grade: str
    default: bool = False


def _v(vid: str, name: str, grade: str, default: bool = False) -> Voice:
    gender = "female" if vid[1] == "f" else "male"
    lang = vid[0]
    accent = "American" if lang == "a" else "British"
    return Voice(vid, name, gender, accent, lang, grade, default)


VOICES: tuple[Voice, ...] = (
    _v("af_heart", "Heart", "A", default=True),
    _v("af_bella", "Bella", "A-"),
    _v("af_nicole", "Nicole", "B-"),
    _v("af_aoede", "Aoede", "C+"),
    _v("af_kore", "Kore", "C+"),
    _v("af_sarah", "Sarah", "C+"),
    _v("af_alloy", "Alloy", "C"),
    _v("af_nova", "Nova", "C"),
    _v("af_sky", "Sky", "C-"),
    _v("af_jessica", "Jessica", "D"),
    _v("af_river", "River", "D"),
    _v("am_fenrir", "Fenrir", "C+"),
    _v("am_michael", "Michael", "C+"),
    _v("am_puck", "Puck", "C+"),
    _v("am_echo", "Echo", "D"),
    _v("am_eric", "Eric", "D"),
    _v("am_liam", "Liam", "D"),
    _v("am_onyx", "Onyx", "D"),
    _v("am_santa", "Santa", "D-"),
    _v("am_adam", "Adam", "F+"),
    _v("bf_emma", "Emma", "B-"),
    _v("bf_isabella", "Isabella", "C"),
    _v("bf_alice", "Alice", "D"),
    _v("bf_lily", "Lily", "D"),
    _v("bm_fable", "Fable", "C"),
    _v("bm_george", "George", "C"),
    _v("bm_lewis", "Lewis", "D+"),
    _v("bm_daniel", "Daniel", "D"),
)

VOICES_BY_ID: dict[str, Voice] = {v.id: v for v in VOICES}

# OpenAI's six voice names mapped onto the closest Kokoro voice, so existing
# OpenAI TTS clients work against /v1/audio/speech unchanged.
OPENAI_ALIASES: dict[str, str] = {
    "alloy": "af_alloy",
    "echo": "am_echo",
    "fable": "bm_fable",
    "onyx": "am_onyx",
    "nova": "af_nova",
    "shimmer": "af_sky",
}


@dataclass(frozen=True)
class BlendComponent:
    voice_id: str
    weight: float


def resolve_alias(name: str) -> str:
    return OPENAI_ALIASES.get(name.strip().lower(), name.strip())


def catalog() -> list[dict]:
    return [
        {
            "id": v.id,
            "name": v.name,
            "gender": v.gender,
            "accent": v.accent,
            "lang": v.lang,
            "grade": v.grade,
            "default": v.default,
        }
        for v in VOICES
    ]


def parse_voice_spec(spec: str) -> list[BlendComponent]:
    if not spec or not spec.strip():
        raise ValueError("Voice spec is empty")

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("Voice spec is empty")
    if len(parts) > MAX_BLEND_COMPONENTS:
        raise ValueError(
            f"A blend may combine at most {MAX_BLEND_COMPONENTS} voices, got {len(parts)}"
        )

    ids: list[str] = []
    weights: list[float | None] = []
    for part in parts:
        name, sep, raw_weight = part.partition(":")
        voice_id = resolve_alias(name)
        if voice_id not in VOICES_BY_ID:
            raise ValueError(
                f"Unknown voice '{name.strip()}'. See GET /voices for the 28 supported ids."
            )
        ids.append(voice_id)
        if not sep:
            weights.append(None)
            continue
        try:
            weight = float(raw_weight)
        except ValueError:
            raise ValueError(
                f"Weight for '{voice_id}' is not a number: '{raw_weight.strip()}'"
            ) from None
        if weight <= 0:
            raise ValueError(f"Weight for '{voice_id}' must be greater than 0")
        weights.append(weight)

    explicit = [w for w in weights if w is not None]
    if explicit and len(explicit) != len(weights):
        raise ValueError(
            "Blend weights must be either all present or all absent, e.g. "
            "'af_bella:0.6,af_sky:0.4' or 'af_bella,af_sky'"
        )

    if not explicit:
        share = 1.0 / len(ids)
        return [BlendComponent(v, share) for v in ids]

    total = sum(explicit)
    return [BlendComponent(v, w / total) for v, w in zip(ids, explicit)]


def canonical_spec(components: list[BlendComponent]) -> str:
    """Stable key for the voice-pack cache."""
    return ",".join(f"{c.voice_id}:{c.weight:.4f}" for c in components)


def lang_for(components: list[BlendComponent]) -> str:
    return components[0].voice_id[0]
```

- [ ] **Step 4: Run the test**

Run: `.venv/bin/pytest tests/test_voices.py -v`
Expected: PASS (all cases, including the 10 parametrized invalid specs)

- [ ] **Step 5: Commit**

```bash
git add app/voices.py tests/test_voices.py
git commit -m "feat: add English voice catalog, OpenAI aliases, blend-spec parsing"
```

---

### Task 3: Core types and the timestamp offset accumulator

This is the highest-risk logic in the project. Kokoro writes `start_ts`/`end_ts`
onto each token, but the values **restart near zero in every segment**. Getting
the offsetting wrong produces word highlights that drift further out of sync the
longer the text — so it is tested here, with no model involved.

**Files:**
- Create: `app/types.py`, `app/timing.py`
- Test: `tests/test_timing.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `WordTiming(word: str, start: float, end: float)`; `Segment(index: int, audio: np.ndarray, words: list[WordTiming], phonemes: str)`; `Synthesis(audio, sample_rate, duration, words, phonemes, voice, segments)`; `EngineProtocol` with `sample_rate: int`, `info() -> dict`, `iter_segments(text, voice, lang, speed) -> Iterator[Segment]`; `words_from_tokens(tokens) -> list[WordTiming]`; `TimelineAccumulator` with `.add(words, duration) -> list[WordTiming]` and `.total` property.

- [ ] **Step 1: Write the failing test**

`tests/test_timing.py`:

```python
from dataclasses import dataclass

import pytest

from app.timing import TimelineAccumulator, words_from_tokens
from app.types import WordTiming


@dataclass
class StubToken:
    """Stands in for misaki.en.MToken — same attribute names, no torch."""

    text: str
    phonemes: str | None = "x"
    start_ts: float | None = None
    end_ts: float | None = None
    whitespace: str = " "


def test_words_from_tokens_extracts_word_and_bounds():
    tokens = [
        StubToken("Hello", start_ts=0.0, end_ts=0.4),
        StubToken("world", start_ts=0.45, end_ts=0.9),
    ]
    assert words_from_tokens(tokens) == [
        WordTiming("Hello", 0.0, 0.4),
        WordTiming("world", 0.45, 0.9),
    ]


def test_tokens_without_timings_are_dropped():
    tokens = [
        StubToken("Hi", start_ts=0.0, end_ts=0.3),
        StubToken(".", phonemes=None),
        StubToken("there", start_ts=None, end_ts=0.9),
    ]
    assert [w.word for w in words_from_tokens(tokens)] == ["Hi"]


def test_surrounding_whitespace_is_stripped_from_words():
    tokens = [StubToken("  Hello\n", start_ts=0.0, end_ts=0.4)]
    assert words_from_tokens(tokens)[0].word == "Hello"


def test_blank_text_tokens_are_dropped():
    tokens = [StubToken("   ", start_ts=0.0, end_ts=0.4)]
    assert words_from_tokens(tokens) == []


def test_end_never_precedes_start():
    tokens = [StubToken("Odd", start_ts=0.5, end_ts=0.2)]
    assert words_from_tokens(tokens) == [WordTiming("Odd", 0.5, 0.5)]


def test_accumulator_leaves_the_first_segment_alone():
    acc = TimelineAccumulator()
    words = acc.add([WordTiming("one", 0.0, 0.5)], duration=1.0)
    assert words == [WordTiming("one", 0.0, 0.5)]
    assert acc.total == pytest.approx(1.0)


def test_accumulator_shifts_later_segments_by_elapsed_audio():
    acc = TimelineAccumulator()
    acc.add([WordTiming("one", 0.0, 0.5)], duration=1.0)
    second = acc.add([WordTiming("two", 0.0, 0.6)], duration=1.5)
    assert second == [WordTiming("two", 1.0, 1.6)]
    assert acc.total == pytest.approx(2.5)


def test_timings_stay_monotonic_across_many_segments():
    """The regression this whole class exists to prevent."""
    acc = TimelineAccumulator()
    collected: list[WordTiming] = []
    for _ in range(5):
        # Every segment reports the same segment-relative timings.
        collected += acc.add(
            [WordTiming("a", 0.0, 0.4), WordTiming("b", 0.5, 0.9)], duration=1.0
        )

    assert len(collected) == 10
    for earlier, later in zip(collected, collected[1:]):
        assert later.start >= earlier.start
        assert earlier.end <= later.end
    assert collected[-1].end == pytest.approx(4.9)
    assert acc.total == pytest.approx(5.0)


def test_accumulator_counts_duration_even_with_no_words():
    acc = TimelineAccumulator()
    assert acc.add([], duration=0.8) == []
    assert acc.add([WordTiming("x", 0.0, 0.2)], duration=0.5) == [
        WordTiming("x", 0.8, 1.0)
    ]
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `.venv/bin/pytest tests/test_timing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.timing'`

- [ ] **Step 3: Implement types**

`app/types.py`:

```python
"""Plain data types shared by every layer. Deliberately torch-free."""
from dataclasses import dataclass, field
from typing import Iterator, Protocol

import numpy as np

SAMPLE_RATE = 24000


@dataclass(frozen=True)
class WordTiming:
    word: str
    start: float
    end: float

    def as_dict(self) -> dict:
        return {"word": self.word, "start": round(self.start, 3), "end": round(self.end, 3)}


@dataclass
class Segment:
    index: int
    audio: np.ndarray  # float32, mono, SAMPLE_RATE
    words: list[WordTiming] = field(default_factory=list)
    phonemes: str = ""

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


@dataclass
class Synthesis:
    audio: np.ndarray
    sample_rate: int
    duration: float
    words: list[WordTiming]
    phonemes: str
    voice: str
    segments: int


class EngineProtocol(Protocol):
    """What the routes need from a synthesizer. KokoroEngine and FakeEngine both satisfy it."""

    sample_rate: int

    def info(self) -> dict: ...

    def iter_segments(
        self, text: str, voice: str, lang: str | None, speed: float
    ) -> Iterator[Segment]: ...
```

- [ ] **Step 4: Implement timing**

`app/timing.py`:

```python
"""Turning Kokoro's segment-relative token timings into one absolute timeline.

Kokoro populates MToken.start_ts / .end_ts via KPipeline.join_timestamps, but
each yielded segment starts its clock over near zero. TimelineAccumulator adds
the elapsed audio duration so the words form a single monotonic timeline.
"""
from typing import Iterable

from app.types import WordTiming


def words_from_tokens(tokens: Iterable) -> list[WordTiming]:
    """Extract timed words from misaki MTokens (duck-typed for testability).

    Tokens without phonemes (punctuation, whitespace) never receive timestamps
    and are skipped.
    """
    words: list[WordTiming] = []
    for token in tokens:
        start = getattr(token, "start_ts", None)
        end = getattr(token, "end_ts", None)
        if start is None or end is None:
            continue
        text = (getattr(token, "text", "") or "").strip()
        if not text:
            continue
        start = float(start)
        end = float(end)
        words.append(WordTiming(text, start, max(start, end)))
    return words


class TimelineAccumulator:
    """Shifts each segment's timings by the audio duration already emitted."""

    def __init__(self) -> None:
        self._offset = 0.0

    @property
    def total(self) -> float:
        return self._offset

    def add(self, words: list[WordTiming], duration: float) -> list[WordTiming]:
        offset = self._offset
        shifted = [
            WordTiming(w.word, w.start + offset, w.end + offset) for w in words
        ]
        self._offset = offset + duration
        return shifted
```

- [ ] **Step 5: Run the test**

Run: `.venv/bin/pytest tests/test_timing.py -v`
Expected: PASS (10 tests)

- [ ] **Step 6: Commit**

```bash
git add app/types.py app/timing.py tests/test_timing.py
git commit -m "feat: add core types and absolute-timeline word timing"
```

---

### Task 4: Audio encoding (WAV, MP3, base64 PCM)

**Files:**
- Create: `app/audio.py`
- Test: `tests/test_audio.py`

**Interfaces:**
- Consumes: `SAMPLE_RATE` from `app.types`.
- Produces: `CONTENT_TYPES: dict[str, str]`; `SUPPORTED_FORMATS: tuple[str, ...]`; `to_int16(audio) -> np.ndarray`; `to_wav_bytes(audio, sample_rate=SAMPLE_RATE) -> bytes`; `to_mp3_bytes(audio, sample_rate=SAMPLE_RATE, bitrate=128) -> bytes`; `encode(audio, fmt, sample_rate=SAMPLE_RATE) -> tuple[bytes, str]`; `pcm_f32_base64(audio) -> str`; `concat(chunks) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

`tests/test_audio.py`:

```python
import base64
import io

import numpy as np
import pytest
import soundfile as sf

from app.audio import (
    CONTENT_TYPES,
    concat,
    encode,
    pcm_f32_base64,
    to_int16,
    to_mp3_bytes,
    to_wav_bytes,
)
from app.types import SAMPLE_RATE


def tone(seconds=0.25, freq=440.0):
    t = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_wav_bytes_are_a_readable_riff_file():
    data = to_wav_bytes(tone())
    assert data[:4] == b"RIFF"
    audio, rate = sf.read(io.BytesIO(data), dtype="float32")
    assert rate == SAMPLE_RATE
    assert audio.ndim == 1
    assert len(audio) == len(tone())


def test_wav_round_trip_preserves_the_signal():
    original = tone()
    audio, _ = sf.read(io.BytesIO(to_wav_bytes(original)), dtype="float32")
    # PCM_16 quantization only; 1/32768 of full scale is the error bound.
    assert np.max(np.abs(audio - original)) < 1e-3


def test_mp3_bytes_start_with_a_frame_sync_or_id3_tag():
    data = to_mp3_bytes(tone(seconds=0.5))
    assert len(data) > 100
    is_id3 = data[:3] == b"ID3"
    is_frame_sync = data[0] == 0xFF and (data[1] & 0xE0) == 0xE0
    assert is_id3 or is_frame_sync


def test_to_int16_clips_instead_of_wrapping():
    loud = np.array([2.0, -2.0, 0.0], dtype=np.float32)
    assert to_int16(loud).tolist() == [32767, -32767, 0]
    assert to_int16(loud).dtype == np.int16


def test_encode_returns_bytes_and_content_type():
    for fmt in ("wav", "mp3"):
        data, content_type = encode(tone(), fmt)
        assert isinstance(data, bytes) and data
        assert content_type == CONTENT_TYPES[fmt]
    assert CONTENT_TYPES["wav"] == "audio/wav"
    assert CONTENT_TYPES["mp3"] == "audio/mpeg"


def test_encode_rejects_unknown_formats():
    with pytest.raises(ValueError) as excinfo:
        encode(tone(), "flac")
    assert "flac" in str(excinfo.value)


def test_pcm_f32_base64_round_trips_little_endian_float32():
    original = tone(seconds=0.05)
    decoded = np.frombuffer(
        base64.b64decode(pcm_f32_base64(original)), dtype="<f4"
    )
    assert np.array_equal(decoded, original)


def test_concat_joins_chunks_and_handles_the_empty_case():
    joined = concat([tone(0.1), tone(0.1)])
    assert len(joined) == len(tone(0.1)) * 2
    assert joined.dtype == np.float32
    empty = concat([])
    assert len(empty) == 0
    assert empty.dtype == np.float32
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `.venv/bin/pytest tests/test_audio.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.audio'`

- [ ] **Step 3: Implement audio encoding**

`app/audio.py`:

```python
"""Encoding float32 mono audio to WAV/MP3 bytes — no ffmpeg involved."""
import base64
import io

import lameenc
import numpy as np
import soundfile as sf

from app.types import SAMPLE_RATE

CONTENT_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg"}
SUPPORTED_FORMATS = tuple(CONTENT_TYPES)
MP3_BITRATE = 128


def to_int16(audio: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, to_int16(audio), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def to_mp3_bytes(
    audio: np.ndarray, sample_rate: int = SAMPLE_RATE, bitrate: int = MP3_BITRATE
) -> bytes:
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)  # 2 = high quality, still fast
    return bytes(encoder.encode(to_int16(audio).tobytes()) + encoder.flush())


def encode(
    audio: np.ndarray, fmt: str, sample_rate: int = SAMPLE_RATE
) -> tuple[bytes, str]:
    if fmt == "wav":
        return to_wav_bytes(audio, sample_rate), CONTENT_TYPES["wav"]
    if fmt == "mp3":
        return to_mp3_bytes(audio, sample_rate), CONTENT_TYPES["mp3"]
    raise ValueError(
        f"Unsupported format '{fmt}'. Supported: {', '.join(SUPPORTED_FORMATS)}"
    )


def pcm_f32_base64(audio: np.ndarray) -> str:
    """Raw little-endian float32 PCM, base64 encoded — the streaming chunk format."""
    return base64.b64encode(
        np.asarray(audio, dtype="<f4").tobytes()
    ).decode("ascii")


def concat(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([np.asarray(c, dtype=np.float32) for c in chunks])
```

- [ ] **Step 4: Run the test**

Run: `.venv/bin/pytest tests/test_audio.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Verify MP3 output is actually playable**

Run:

```bash
.venv/bin/python -c "
import numpy as np
from app.audio import to_mp3_bytes, to_wav_bytes
from app.types import SAMPLE_RATE
t = np.arange(SAMPLE_RATE, dtype=np.float32)/SAMPLE_RATE
a = (0.4*np.sin(2*np.pi*440*t)).astype(np.float32)
open('/tmp/kokoro_check.mp3','wb').write(to_mp3_bytes(a))
open('/tmp/kokoro_check.wav','wb').write(to_wav_bytes(a))
print('wrote', len(to_mp3_bytes(a)), 'mp3 bytes')
" && afplay /tmp/kokoro_check.mp3 && echo "MP3 played OK"
```

Expected: a one-second 440 Hz tone plays; `afplay` exits 0.

- [ ] **Step 6: Commit**

```bash
git add app/audio.py tests/test_audio.py
git commit -m "feat: add WAV/MP3/base64-PCM encoding without ffmpeg"
```

---

### Task 5: Synthesis service, fake engine, shared test fixtures

The service owns concurrency (semaphore + thread offload) so routes stay thin
and so every later task can be tested without torch.

**Files:**
- Create: `app/service.py`, `tests/fakes.py`, `tests/conftest.py`
- Test: `tests/test_service.py`

**Interfaces:**
- Consumes: `EngineProtocol`, `Segment`, `Synthesis`, `SAMPLE_RATE` from `app.types`; `concat` from `app.audio`.
- Produces: `SynthesisService(engine, max_concurrency=1)` with `.sample_rate`, `.info() -> dict`, `async .stream_segments(text, voice, lang, speed) -> AsyncIterator[Segment]`, `async .synthesize(text, voice, lang, speed) -> Synthesis`; `FakeEngine(...)` with `.calls` list; `client` and `app_with_fake_engine` pytest fixtures.

- [ ] **Step 1: Write the failing test**

`tests/test_service.py`:

```python
import asyncio

import pytest

from app.service import SynthesisService
from tests.fakes import FakeEngine


@pytest.mark.asyncio
async def test_synthesize_concatenates_segments_and_words():
    service = SynthesisService(FakeEngine(), max_concurrency=1)
    result = await service.synthesize("one two\nthree four", "af_heart", "a", 1.0)

    assert result.segments == 2
    assert [w.word for w in result.words] == ["one", "two", "three", "four"]
    assert result.sample_rate == 24000
    assert result.duration == pytest.approx(len(result.audio) / 24000)
    assert result.voice == "af_heart"


@pytest.mark.asyncio
async def test_word_timings_are_absolute_across_segments():
    service = SynthesisService(FakeEngine(), max_concurrency=1)
    result = await service.synthesize("one two\nthree four", "af_heart", "a", 1.0)

    starts = [w.start for w in result.words]
    assert starts == sorted(starts)
    # The second segment's first word must start after the first segment's audio.
    assert result.words[2].start >= result.words[1].end


@pytest.mark.asyncio
async def test_stream_segments_yields_progressively():
    service = SynthesisService(FakeEngine(), max_concurrency=1)
    seen = [seg.index async for seg in service.stream_segments("a\nb\nc", "af_heart", "a", 1.0)]
    assert seen == [0, 1, 2]


@pytest.mark.asyncio
async def test_engine_receives_the_arguments_it_was_given():
    engine = FakeEngine()
    service = SynthesisService(engine, max_concurrency=1)
    await service.synthesize("hello", "af_bella:0.5,af_sky:0.5", "b", 1.25)
    assert engine.calls == [("hello", "af_bella:0.5,af_sky:0.5", "b", 1.25)]


@pytest.mark.asyncio
async def test_semaphore_serializes_concurrent_synthesis():
    engine = FakeEngine(segment_delay=0.05)
    service = SynthesisService(engine, max_concurrency=1)

    async def run():
        return await service.synthesize("x", "af_heart", "a", 1.0)

    started = asyncio.get_running_loop().time()
    await asyncio.gather(run(), run())
    elapsed = asyncio.get_running_loop().time() - started
    # Serialized: two 50ms synthesis calls cannot finish in under 100ms.
    assert elapsed >= 0.09


@pytest.mark.asyncio
async def test_the_event_loop_stays_responsive_during_synthesis():
    """Synthesis must run in a thread, not block the loop."""
    engine = FakeEngine(segment_delay=0.2)
    service = SynthesisService(engine, max_concurrency=1)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    await service.synthesize("x", "af_heart", "a", 1.0)
    task.cancel()
    assert ticks > 5


def test_info_includes_engine_details_and_concurrency():
    service = SynthesisService(FakeEngine(), max_concurrency=3)
    info = service.info()
    assert info["device"] == "fake"
    assert info["max_concurrency"] == 3
```

- [ ] **Step 2: Write the fake engine and fixtures**

`tests/fakes.py`:

```python
"""A deterministic stand-in for KokoroEngine. No torch, no model, no downloads."""
import time
from typing import Iterator

import numpy as np

from app.types import SAMPLE_RATE, Segment, WordTiming

WORD_SECONDS = 0.3
GAP_SECONDS = 0.05


class FakeEngine:
    """Splits text on newlines into segments and on whitespace into timed words.

    Emits a 220 Hz tone whose length matches the words, so audio duration and
    word timings stay consistent with each other.
    """

    sample_rate = SAMPLE_RATE

    def __init__(self, segment_delay: float = 0.0, device: str = "fake"):
        self.segment_delay = segment_delay
        self.device = device
        self.calls: list[tuple] = []

    def info(self) -> dict:
        return {
            "device": self.device,
            "backend": "fake",
            "torch_version": "n/a",
            "warmup_seconds": 0.0,
            "voices": 28,
        }

    def iter_segments(
        self, text: str, voice: str, lang: str | None, speed: float
    ) -> Iterator[Segment]:
        self.calls.append((text, voice, lang, speed))
        lines = [line for line in text.split("\n") if line.strip()]
        for index, line in enumerate(lines):
            if self.segment_delay:
                time.sleep(self.segment_delay)
            words: list[WordTiming] = []
            cursor = 0.0
            for word in line.split():
                # Segment-relative on purpose: the service must make these absolute.
                words.append(WordTiming(word, cursor, cursor + WORD_SECONDS))
                cursor += WORD_SECONDS + GAP_SECONDS
            samples = max(1, int(cursor * SAMPLE_RATE))
            t = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
            audio = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
            yield Segment(
                index=index, audio=audio, words=words, phonemes=f"seg{index}"
            )
```

`tests/conftest.py`:

```python
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
```

- [ ] **Step 3: Run the test to confirm it fails**

Run: `.venv/bin/pytest tests/test_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.service'`

- [ ] **Step 4: Implement the service**

`app/service.py`:

```python
"""Concurrency and threading around any EngineProtocol implementation.

Torch inference here is not safely reentrant and the Mac has four cores, so
synthesis is serialized by a semaphore and executed in worker threads, keeping
the event loop free to serve /health and static assets.
"""
import asyncio
from typing import AsyncIterator

from app.audio import concat
from app.timing import TimelineAccumulator
from app.types import EngineProtocol, Segment, Synthesis


class SynthesisService:
    def __init__(self, engine: EngineProtocol, max_concurrency: int = 1):
        self._engine = engine
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def sample_rate(self) -> int:
        return self._engine.sample_rate

    def info(self) -> dict:
        return {**self._engine.info(), "max_concurrency": self._max_concurrency}

    async def stream_segments(
        self, text: str, voice: str, lang: str | None, speed: float
    ) -> AsyncIterator[Segment]:
        """Yield segments as they finish, with absolute word timings."""
        sentinel = object()
        async with self._semaphore:
            generator = self._engine.iter_segments(text, voice, lang, speed)
            timeline = TimelineAccumulator()

            def pull():
                return next(generator, sentinel)

            while True:
                segment = await asyncio.to_thread(pull)
                if segment is sentinel:
                    return
                segment.words = timeline.add(segment.words, segment.duration)
                yield segment

    async def synthesize(
        self, text: str, voice: str, lang: str | None, speed: float
    ) -> Synthesis:
        chunks = []
        words = []
        phonemes = []
        count = 0
        async for segment in self.stream_segments(text, voice, lang, speed):
            chunks.append(segment.audio)
            words.extend(segment.words)
            if segment.phonemes:
                phonemes.append(segment.phonemes)
            count += 1

        audio = concat(chunks)
        return Synthesis(
            audio=audio,
            sample_rate=self.sample_rate,
            duration=len(audio) / self.sample_rate,
            words=words,
            phonemes=" ".join(phonemes),
            voice=voice,
            segments=count,
        )
```

- [ ] **Step 5: Run the test**

Run: `.venv/bin/pytest tests/test_service.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Run the whole suite and check it is still fast**

Run: `.venv/bin/pytest -q --durations=5`
Expected: all tests pass; total runtime under ~3 seconds; no torch import anywhere.

- [ ] **Step 7: Commit**

```bash
git add app/service.py tests/fakes.py tests/conftest.py tests/test_service.py
git commit -m "feat: add synthesis service with semaphore and thread offloading"
```

---

### Task 6: Native `POST /tts` and `GET /voices`

**Files:**
- Create: `app/validation.py`, `app/schemas.py`, `app/routes/native.py`
- Modify: `app/main.py` (include the native router)
- Test: `tests/test_tts.py`, `tests/test_voices_endpoint.py`

**Interfaces:**
- Consumes: `get_service`, `get_settings_dep` from `app.deps`; `clean_text`/`resolve_voice` are defined here; `encode` from `app.audio`; `parse_voice_spec`, `canonical_spec`, `lang_for`, `catalog`, `DEFAULT_VOICE_ID` from `app.voices`.
- Produces: `clean_text(text: str, max_chars: int) -> str`; `resolve_voice(spec: str | None, default: str) -> tuple[str, str]` returning `(engine_spec, lang)`; `encode_or_400(audio, fmt, sample_rate) -> tuple[bytes, str]`; `TtsRequest` pydantic model; `router` in `app.routes.native` serving `POST /tts` and `GET /voices`.

- [ ] **Step 1: Write the failing tests**

`tests/test_tts.py`:

```python
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
```

`tests/test_voices_endpoint.py`:

```python
def test_voices_endpoint_lists_all_28(client):
    body = client.get("/voices").json()
    assert body["count"] == 28
    assert len(body["voices"]) == 28
    assert body["default"] == "af_heart"


def test_voice_rows_carry_display_metadata(client):
    row = next(v for v in client.get("/voices").json()["voices"] if v["id"] == "bm_george")
    assert row["accent"] == "British"
    assert row["gender"] == "male"
    assert row["lang"] == "b"
    assert row["grade"] == "C"
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `.venv/bin/pytest tests/test_tts.py tests/test_voices_endpoint.py -v`
Expected: FAIL — 404 responses (routes not registered) / `ModuleNotFoundError: No module named 'app.validation'`

- [ ] **Step 3: Implement validation helpers**

`app/validation.py`:

```python
"""Request validation that raises ApiError, keeping routes free of error shaping."""
import numpy as np

from app.audio import encode
from app.errors import ApiError
from app.voices import DEFAULT_VOICE_ID, canonical_spec, lang_for, parse_voice_spec


def clean_text(text: str, max_chars: int) -> str:
    stripped = (text or "").strip()
    if not stripped:
        raise ApiError(400, "Text is empty")
    if len(stripped) > max_chars:
        raise ApiError(
            400,
            f"Text is {len(stripped)} characters; the limit is {max_chars}",
        )
    return stripped


def resolve_voice(spec: str | None, default: str) -> tuple[str, str]:
    """Normalize a voice spec and derive its language code.

    Returns (engine_spec, lang). A single voice keeps its plain id so Kokoro's
    own voice cache is used; blends become the canonical weighted form, which
    parse_voice_spec can read back.
    """
    raw = (spec or default or DEFAULT_VOICE_ID).strip()
    try:
        components = parse_voice_spec(raw)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from None
    engine_spec = (
        components[0].voice_id if len(components) == 1 else canonical_spec(components)
    )
    return engine_spec, lang_for(components)


def encode_or_400(
    audio: np.ndarray, fmt: str, sample_rate: int
) -> tuple[bytes, str]:
    try:
        return encode(audio, fmt, sample_rate)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from None
```

- [ ] **Step 4: Implement request schemas**

`app/schemas.py`:

```python
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TtsRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    lang: Optional[Literal["a", "b"]] = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    format: Literal["wav", "mp3"] = "wav"
    include_timestamps: bool = False


class SpeechRequest(BaseModel):
    """OpenAI's /v1/audio/speech body. `model` is accepted and ignored."""

    input: str
    model: str = "kokoro"
    voice: Optional[str] = None
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    response_format: str = "wav"
```

- [ ] **Step 5: Implement the native router**

`app/routes/native.py`:

```python
import base64

from fastapi import APIRouter, Depends, Response

from app.config import Settings
from app.deps import get_service, get_settings_dep
from app.schemas import TtsRequest
from app.service import SynthesisService
from app.validation import clean_text, encode_or_400, resolve_voice
from app.voices import DEFAULT_VOICE_ID, catalog

router = APIRouter()


@router.get("/voices")
async def list_voices() -> dict:
    rows = catalog()
    return {"voices": rows, "count": len(rows), "default": DEFAULT_VOICE_ID}


@router.post("/tts")
async def tts(
    request: TtsRequest,
    service: SynthesisService = Depends(get_service),
    settings: Settings = Depends(get_settings_dep),
):
    text = clean_text(request.text, settings.max_chars)
    voice, voice_lang = resolve_voice(request.voice, settings.default_voice)
    lang = request.lang or voice_lang

    result = await service.synthesize(text, voice, lang, request.speed)
    data, content_type = encode_or_400(
        result.audio, request.format, result.sample_rate
    )

    if not request.include_timestamps:
        return Response(
            content=data,
            media_type=content_type,
            headers={
                "X-Audio-Duration": f"{result.duration:.3f}",
                "X-Voice": voice,
            },
        )

    return {
        "audio": base64.b64encode(data).decode("ascii"),
        "format": request.format,
        "sample_rate": result.sample_rate,
        "duration": round(result.duration, 3),
        "voice": voice,
        "segments": result.segments,
        "phonemes": result.phonemes,
        "words": [w.as_dict() for w in result.words],
    }
```

- [ ] **Step 6: Register the router**

In `app/main.py`, add the import and include it after `health`:

```python
from app.routes import health, native
```

```python
    app.include_router(health.router)
    app.include_router(native.router)
```

- [ ] **Step 7: Run the tests**

Run: `.venv/bin/pytest tests/test_tts.py tests/test_voices_endpoint.py -v`
Expected: PASS (15 tests)

- [ ] **Step 8: Commit**

```bash
git add app/validation.py app/schemas.py app/routes/native.py app/main.py tests/test_tts.py tests/test_voices_endpoint.py
git commit -m "feat: add POST /tts with word timestamps and GET /voices"
```

---

### Task 7: Streaming `POST /tts/stream` (NDJSON)

Chunk audio is **always** raw float32 PCM base64 — the request's `format` field
is ignored here, because per-segment WAV headers or MP3 frames would have to be
stripped and rejoined by the client for no gain.

**Files:**
- Modify: `app/routes/native.py` (add the streaming route)
- Test: `tests/test_stream.py`

**Interfaces:**
- Consumes: `SynthesisService.stream_segments`, `pcm_f32_base64` from `app.audio`.
- Produces: `POST /tts/stream` emitting `application/x-ndjson` lines of type `meta`, `chunk`, `done`, or `error`.

- [ ] **Step 1: Write the failing test**

`tests/test_stream.py`:

```python
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
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `.venv/bin/pytest tests/test_stream.py -v`
Expected: FAIL — 404 Not Found on `/tts/stream`

- [ ] **Step 3: Implement the streaming route**

Add to `app/routes/native.py` — imports first:

```python
import json
import logging

from fastapi.responses import StreamingResponse

from app.audio import pcm_f32_base64

logger = logging.getLogger(__name__)
```

Then the route:

```python
@router.post("/tts/stream")
async def tts_stream(
    request: TtsRequest,
    service: SynthesisService = Depends(get_service),
    settings: Settings = Depends(get_settings_dep),
):
    """NDJSON stream of float32 PCM chunks with absolute word timings.

    Validation runs here, before the response starts, so bad requests still get
    a normal 400 rather than an error buried in the stream.
    """
    text = clean_text(request.text, settings.max_chars)
    voice, voice_lang = resolve_voice(request.voice, settings.default_voice)
    lang = request.lang or voice_lang

    async def lines():
        yield json.dumps(
            {
                "type": "meta",
                "sample_rate": service.sample_rate,
                "voice": voice,
                "lang": lang,
                "format": "pcm_f32le",
            }
        ) + "\n"

        total = 0.0
        count = 0
        try:
            async for segment in service.stream_segments(
                text, voice, lang, request.speed
            ):
                total += segment.duration
                count += 1
                yield json.dumps(
                    {
                        "type": "chunk",
                        "index": segment.index,
                        "audio": pcm_f32_base64(segment.audio),
                        "duration": round(segment.duration, 3),
                        "words": [w.as_dict() for w in segment.words],
                    }
                ) + "\n"
        except Exception as exc:  # noqa: BLE001 - surfaced to the client below
            logger.exception("streaming synthesis failed")
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
            return

        yield json.dumps(
            {
                "type": "done",
                "duration": round(total, 3),
                "segments": count,
            }
        ) + "\n"

    return StreamingResponse(
        lines(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 4: Run the test**

Run: `.venv/bin/pytest tests/test_stream.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Confirm chunks actually arrive progressively**

The TestClient buffers, so verify streaming behaviour against a live server:

```bash
.venv/bin/python -c "
import uvicorn, threading, time, json, urllib.request
from app.main import create_app
from app.service import SynthesisService
import sys; sys.path.insert(0, '.')
from tests.fakes import FakeEngine
app = create_app()
app.state.service = SynthesisService(FakeEngine(segment_delay=0.4), 1)
cfg = uvicorn.Config(app, host='127.0.0.1', port=8099, log_level='error')
server = uvicorn.Server(cfg)
threading.Thread(target=server.run, daemon=True).start()
time.sleep(1.5)
req = urllib.request.Request('http://127.0.0.1:8099/tts/stream',
    data=json.dumps({'text':'one\ntwo\nthree'}).encode(),
    headers={'Content-Type':'application/json'})
start = time.time()
with urllib.request.urlopen(req) as resp:
    for raw in resp:
        event = json.loads(raw)
        print(f\"{time.time()-start:5.2f}s {event['type']}\")
server.should_exit = True
"
```

Expected: `meta` near 0.00s, then `chunk` lines roughly 0.4s apart — not all at once at the end.

- [ ] **Step 6: Commit**

```bash
git add app/routes/native.py tests/test_stream.py
git commit -m "feat: add NDJSON streaming endpoint with per-chunk word timings"
```

---

### Task 8: OpenAI-compatible `POST /v1/audio/speech`

**Files:**
- Create: `app/routes/openai.py`
- Modify: `app/main.py` (include the router)
- Test: `tests/test_openai_endpoint.py`

**Interfaces:**
- Consumes: `SpeechRequest` from `app.schemas`; `clean_text`, `resolve_voice`, `encode_or_400`; `SUPPORTED_FORMATS` from `app.audio`.
- Produces: `POST /v1/audio/speech` returning raw audio bytes.

- [ ] **Step 1: Write the failing test**

`tests/test_openai_endpoint.py`:

```python
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
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `.venv/bin/pytest tests/test_openai_endpoint.py -v`
Expected: FAIL — 404 Not Found on `/v1/audio/speech`

- [ ] **Step 3: Implement the router**

`app/routes/openai.py`:

```python
"""OpenAI-compatible speech endpoint, so existing OpenAI TTS clients work as-is."""
from fastapi import APIRouter, Depends, Response

from app.audio import SUPPORTED_FORMATS
from app.config import Settings
from app.deps import get_service, get_settings_dep
from app.errors import ApiError
from app.schemas import SpeechRequest
from app.service import SynthesisService
from app.validation import clean_text, encode_or_400, resolve_voice

router = APIRouter(prefix="/v1")


@router.post("/audio/speech")
async def create_speech(
    request: SpeechRequest,
    service: SynthesisService = Depends(get_service),
    settings: Settings = Depends(get_settings_dep),
):
    if request.response_format not in SUPPORTED_FORMATS:
        raise ApiError(
            400,
            f"response_format '{request.response_format}' is not supported. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}",
        )

    text = clean_text(request.input, settings.max_chars)
    voice, lang = resolve_voice(request.voice, settings.default_voice)

    result = await service.synthesize(text, voice, lang, request.speed)
    data, content_type = encode_or_400(
        result.audio, request.response_format, result.sample_rate
    )
    return Response(
        content=data,
        media_type=content_type,
        headers={"X-Audio-Duration": f"{result.duration:.3f}", "X-Voice": voice},
    )
```

- [ ] **Step 4: Register the router**

In `app/main.py`:

```python
from app.routes import health, native, openai
```

```python
    app.include_router(health.router)
    app.include_router(native.router)
    app.include_router(openai.router)
```

- [ ] **Step 5: Run the test**

Run: `.venv/bin/pytest tests/test_openai_endpoint.py -v`
Expected: PASS (11 tests)

- [ ] **Step 6: Commit**

```bash
git add app/routes/openai.py app/main.py tests/test_openai_endpoint.py
git commit -m "feat: add OpenAI-compatible /v1/audio/speech endpoint"
```

---

### Task 9: Optional bearer-token auth

**Files:**
- Create: `app/auth.py`
- Modify: `app/main.py` (apply the dependency to the synthesis routers)
- Test: `tests/test_auth.py`

**Interfaces:**
- Consumes: `Settings`, `get_settings_dep`, `ApiError`.
- Produces: `require_api_key(request, settings)` FastAPI dependency, applied to the native and OpenAI routers but never to `/health` or the static UI.

- [ ] **Step 1: Write the failing test**

`tests/test_auth.py`:

```python
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
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `.venv/bin/pytest tests/test_auth.py -v`
Expected: FAIL — protected routes return 200 instead of 401

- [ ] **Step 3: Implement auth**

`app/auth.py`:

```python
"""Optional bearer-token auth. Unset KOKORO_API_KEY means open access."""
import hmac

from fastapi import Depends, Request

from app.config import Settings
from app.deps import get_settings_dep
from app.errors import ApiError


async def require_api_key(
    request: Request, settings: Settings = Depends(get_settings_dep)
) -> None:
    expected = settings.api_key
    if not expected:
        return

    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        token.strip(), expected
    ):
        raise ApiError(401, "Invalid or missing API key", "authentication_error")
```

- [ ] **Step 4: Apply it to the synthesis routers**

In `app/main.py`, add the import:

```python
from fastapi import Depends, FastAPI

from app.auth import require_api_key
```

and change the two `include_router` calls (leaving `health` unguarded):

```python
    app.include_router(health.router)
    app.include_router(native.router, dependencies=[Depends(require_api_key)])
    app.include_router(openai.router, dependencies=[Depends(require_api_key)])
```

- [ ] **Step 5: Run the test**

Run: `.venv/bin/pytest tests/test_auth.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/pytest -q`
Expected: everything passes, still no torch import, still a few seconds.

- [ ] **Step 7: Commit**

```bash
git add app/auth.py app/main.py tests/test_auth.py
git commit -m "feat: add optional bearer-token auth for synthesis routes"
```

---

### Task 10: The real Kokoro engine, lifespan wiring, and Mac setup

This is the first task that installs torch. Everything before it must still run
without the model afterwards — verify that at the end.

**Files:**
- Create: `app/engine.py`, `scripts/bake_assets.py`, `scripts/setup_mac.sh`, `requirements-mac-cpu.txt`
- Modify: `app/main.py` (lifespan + `create_app(load_model=...)`)
- Test: `tests/test_engine_slow.py`, `tests/test_app_factory.py`

**Interfaces:**
- Consumes: `words_from_tokens` from `app.timing`; `Segment`, `SAMPLE_RATE` from `app.types`; `parse_voice_spec`, `VOICES`, `DEFAULT_VOICE_ID` from `app.voices`; `SynthesisService`; `resolve_concurrency`.
- Produces: `resolve_device(requested: str) -> str`; `KokoroEngine(device="auto", torch_threads=0, voice_cache_size=32)` satisfying `EngineProtocol`, with `.device`, `.warmup_seconds`, `.info()`, `.iter_segments(...)`; `create_app(load_model: bool = False) -> FastAPI`; module-level `app = create_app(load_model=True)`.

- [ ] **Step 1: Write the failing app-factory test**

`tests/test_app_factory.py`:

```python
from fastapi.testclient import TestClient

from app.main import create_app


def test_factory_defaults_to_not_loading_the_model():
    """The fast suite must never pull in torch via the lifespan handler."""
    app = create_app()
    assert app.state.load_model is False
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "loading"


def test_routes_are_registered():
    paths = {route.path for route in create_app().routes}
    assert {"/health", "/tts", "/tts/stream", "/voices", "/v1/audio/speech"} <= paths
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `.venv/bin/pytest tests/test_app_factory.py -v`
Expected: FAIL — `AttributeError: 'State' object has no attribute 'load_model'`

- [ ] **Step 3: Rewrite `app/main.py` with the lifespan handler**

The `KokoroEngine` import stays *inside* the lifespan function — importing it at
module scope would pull torch into every test.

```python
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth import require_api_key
from app.config import get_settings, resolve_concurrency
from app.errors import install_error_handlers
from app.routes import health, native, openai
from app.service import SynthesisService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if app.state.load_model:
        from app.engine import KokoroEngine  # local: keeps torch out of tests

        settings = get_settings()
        logger.info("loading Kokoro (device=%s)", settings.device)
        engine = await asyncio.to_thread(
            KokoroEngine,
            settings.device,
            settings.torch_threads,
            settings.voice_cache_size,
        )
        app.state.service = SynthesisService(
            engine, resolve_concurrency(engine.device, settings.max_concurrency)
        )
        logger.info(
            "ready on %s (warmup %.2fs)", engine.device, engine.warmup_seconds
        )
    yield
    app.state.service = None


def create_app(load_model: bool = False) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Kokoro TTS API", version="1.0.0", lifespan=lifespan)
    app.state.load_model = load_model
    app.state.service = None

    install_error_handlers(app)
    origins = settings.origin_list()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(health.router)
    app.include_router(native.router, dependencies=[Depends(require_api_key)])
    app.include_router(openai.router, dependencies=[Depends(require_api_key)])
    return app


app = create_app(load_model=True)
```

- [ ] **Step 4: Run the app-factory test and the whole fast suite**

Run: `.venv/bin/pytest -q`
Expected: PASS — all previous tests still green, still no torch installed.

- [ ] **Step 5: Commit the wiring before installing torch**

```bash
git add app/main.py tests/test_app_factory.py
git commit -m "feat: add lifespan model loading behind create_app(load_model=)"
```

- [ ] **Step 6: Write the Mac requirements file**

`requirements-mac-cpu.txt` — the pins here are load-bearing, see Global Constraints:

```
-r requirements-base.txt

# 2.2.2 is the last torch release with a macosx_10_9_x86_64 cp310 wheel.
torch==2.2.2
# Current transformers releases assume newer torch; this one works with 2.2.
transformers==4.44.2
kokoro==0.9.4
misaki[en]>=0.9.4
```

- [ ] **Step 7: Write the asset pre-download script**

`scripts/bake_assets.py`:

```python
"""Pre-download the model, config, and all 28 English voice packs.

Used by the Mac setup script and by the Docker build, so a container can start
with no network access.
"""
import sys

from huggingface_hub import hf_hub_download

sys.path.insert(0, ".")
from app.voices import VOICES  # noqa: E402

REPO_ID = "hexgrad/Kokoro-82M"


def main() -> None:
    for filename in ("config.json", "kokoro-v1_0.pth"):
        print(f"fetching {filename}")
        hf_hub_download(REPO_ID, filename)
    for voice in VOICES:
        print(f"fetching voices/{voice.id}.pt")
        hf_hub_download(REPO_ID, f"voices/{voice.id}.pt")
    print(f"done: model + {len(VOICES)} English voices cached")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Write the failing slow engine test**

`tests/test_engine_slow.py`:

```python
"""Real-model tests. Gated: run with KOKORO_RUN_SLOW=1."""
import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("KOKORO_RUN_SLOW") != "1",
        reason="set KOKORO_RUN_SLOW=1 to run model-loading tests",
    ),
]

SENTENCE = "The quick brown fox jumps over the lazy dog."


@pytest.fixture(scope="module")
def engine():
    from app.engine import KokoroEngine

    return KokoroEngine(device="auto")


def test_engine_reports_its_backend(engine):
    info = engine.info()
    assert info["backend"] == "pytorch"
    assert info["device"] in {"cpu", "cuda"}
    assert info["voices"] == 28
    assert info["warmup_seconds"] > 0


def test_synthesis_produces_non_silent_audio_of_plausible_length(engine):
    segments = list(engine.iter_segments(SENTENCE, "af_heart", "a", 1.0))
    assert segments
    audio = np.concatenate([s.audio for s in segments])
    duration = len(audio) / engine.sample_rate

    assert audio.dtype == np.float32
    assert np.max(np.abs(audio)) > 0.05, "audio is silent"
    assert 1.5 < duration < 6.0, f"implausible duration {duration:.2f}s"


def test_word_timings_cover_the_sentence_in_order(engine):
    segments = list(engine.iter_segments(SENTENCE, "af_heart", "a", 1.0))
    words = [w for s in segments for w in s.words]

    spoken = [w.word.lower().strip(".,") for w in words]
    assert spoken == SENTENCE.lower().rstrip(".").split()

    for earlier, later in zip(words, words[1:]):
        assert earlier.start <= earlier.end
        assert later.start >= earlier.start


def test_timings_stay_inside_the_audio_duration(engine):
    segments = list(engine.iter_segments(SENTENCE, "af_heart", "a", 1.0))
    duration = sum(s.duration for s in segments)
    words = [w for s in segments for w in s.words]
    # Segment-relative here; the service adds offsets. Per-segment bound only.
    for segment in segments:
        for word in segment.words:
            assert 0 <= word.start <= segment.duration + 0.2
    assert duration > 0


def test_multi_paragraph_text_yields_multiple_segments(engine):
    text = "First paragraph here.\nSecond paragraph here.\nThird one too."
    segments = list(engine.iter_segments(text, "af_heart", "a", 1.0))
    assert len(segments) >= 2
    assert [s.index for s in segments] == list(range(len(segments)))


def test_british_voice_and_lang_code(engine):
    segments = list(engine.iter_segments("Good afternoon.", "bm_george", "b", 1.0))
    audio = np.concatenate([s.audio for s in segments])
    assert np.max(np.abs(audio)) > 0.05


def test_weighted_blend_synthesizes(engine):
    segments = list(
        engine.iter_segments("Blended voice test.", "af_bella:0.7000,af_sky:0.3000", "a", 1.0)
    )
    audio = np.concatenate([s.audio for s in segments])
    assert np.max(np.abs(audio)) > 0.05


def test_blend_differs_from_either_component(engine):
    text = "Comparing voices now."
    def render(voice):
        return np.concatenate(
            [s.audio for s in engine.iter_segments(text, voice, "a", 1.0)]
        )

    bella = render("af_bella")
    blend = render("af_bella:0.5000,af_sky:0.5000")
    shortest = min(len(bella), len(blend))
    assert not np.allclose(bella[:shortest], blend[:shortest], atol=1e-4)


def test_speed_affects_duration(engine):
    slow = list(engine.iter_segments(SENTENCE, "af_heart", "a", 0.8))
    fast = list(engine.iter_segments(SENTENCE, "af_heart", "a", 1.5))
    assert sum(s.duration for s in fast) < sum(s.duration for s in slow)


@pytest.mark.asyncio
async def test_service_end_to_end_with_the_real_engine(engine):
    from app.service import SynthesisService

    service = SynthesisService(engine, max_concurrency=1)
    result = await service.synthesize(
        "One two three.\nFour five six.", "af_heart", "a", 1.0
    )
    assert result.segments >= 2
    starts = [w.start for w in result.words]
    assert starts == sorted(starts)
    assert result.words[-1].end <= result.duration + 0.5
```

- [ ] **Step 9: Implement the engine**

`app/engine.py`:

```python
"""Kokoro-82M inference. The ONLY module permitted to import torch or kokoro."""
import logging
import time
from collections import OrderedDict
from typing import Iterator

import numpy as np

from app.timing import words_from_tokens
from app.types import SAMPLE_RATE, Segment
from app.voices import DEFAULT_VOICE_ID, VOICES, parse_voice_spec

logger = logging.getLogger(__name__)

REPO_ID = "hexgrad/Kokoro-82M"
WARMUP_TEXT = "Kokoro is ready."
LANG_CODES = ("a", "b")


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "KOKORO_DEVICE=cuda but torch reports no CUDA device available"
        )
    return requested


class KokoroEngine:
    sample_rate = SAMPLE_RATE

    def __init__(
        self,
        device: str = "auto",
        torch_threads: int = 0,
        voice_cache_size: int = 32,
    ):
        import torch
        from kokoro import KModel, KPipeline

        self._torch = torch
        self.device = resolve_device(device)
        if torch_threads > 0:
            torch.set_num_threads(torch_threads)

        self._model = KModel(repo_id=REPO_ID).to(self.device).eval()
        # One model, two pipelines: US and UK G2P share the same weights.
        self._pipelines = {
            code: KPipeline(lang_code=code, repo_id=REPO_ID, model=self._model)
            for code in LANG_CODES
        }
        self._cache_size = voice_cache_size
        self._blend_cache: "OrderedDict[str, object]" = OrderedDict()
        self.warmup_seconds = self._warmup()

    def _warmup(self) -> float:
        started = time.perf_counter()
        for _ in self.iter_segments(WARMUP_TEXT, DEFAULT_VOICE_ID, "a", 1.0):
            pass
        elapsed = time.perf_counter() - started
        logger.info("warmup synthesis took %.2fs on %s", elapsed, self.device)
        return elapsed

    def info(self) -> dict:
        return {
            "device": self.device,
            "backend": "pytorch",
            "torch_version": self._torch.__version__,
            "warmup_seconds": round(self.warmup_seconds, 3),
            "voices": len(VOICES),
        }

    def _voice_argument(self, spec: str, lang: str):
        """A plain voice id, or a CPU float32 blend tensor.

        Kokoro's load_voice averages voices equally; weighted blends are ours.
        The tensor MUST be CPU float32: KPipeline.load_voice only passes a
        tensor through when isinstance(voice, torch.FloatTensor), which is
        False for CUDA tensors.
        """
        components = parse_voice_spec(spec)
        if len(components) == 1:
            return components[0].voice_id

        cached = self._blend_cache.get(spec)
        if cached is not None:
            self._blend_cache.move_to_end(spec)
            return cached

        pipeline = self._pipelines[lang]
        packs = [
            pipeline.load_single_voice(c.voice_id).detach().cpu().float()
            for c in components
        ]
        stacked = self._torch.stack(packs)
        weights = self._torch.tensor(
            [c.weight for c in components], dtype=self._torch.float32
        ).view(-1, 1, 1, 1)
        blend = (stacked * weights).sum(dim=0).cpu().float()

        self._blend_cache[spec] = blend
        while len(self._blend_cache) > self._cache_size:
            self._blend_cache.popitem(last=False)
        return blend

    def iter_segments(
        self, text: str, voice: str, lang: str | None, speed: float
    ) -> Iterator[Segment]:
        code = lang if lang in LANG_CODES else "a"
        pipeline = self._pipelines[code]
        voice_argument = self._voice_argument(voice, code)

        index = 0
        for result in pipeline(text, voice=voice_argument, speed=speed):
            if result.audio is None:
                continue
            audio = result.audio.detach().cpu().numpy().astype(np.float32)
            # Timestamps here are segment-relative; SynthesisService offsets them.
            yield Segment(
                index=index,
                audio=audio,
                words=words_from_tokens(result.tokens or []),
                phonemes=result.phonemes or "",
            )
            index += 1
```

- [ ] **Step 10: Write the Mac setup script**

`scripts/setup_mac.sh`:

```bash
#!/usr/bin/env bash
# Idempotent CPU-only setup for Intel macOS. Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3.10}"

echo "==> checking the environment"
arch="$(uname -m)"
echo "    arch:   $arch"
"$PYTHON" -c 'import sys; v=sys.version_info; \
  sys.exit(0 if (3,10) <= (v.major,v.minor) < (3,13) else 1)' || {
  echo "ERROR: need Python 3.10-3.12, got $("$PYTHON" -V 2>&1)"; exit 1; }
echo "    python: $("$PYTHON" -V 2>&1)"

free_gb="$(df -g . | awk 'NR==2 {print $4}')"
echo "    free:   ${free_gb}Gi"
if [ "$free_gb" -lt 4 ]; then
  echo "ERROR: need at least 4Gi free for torch + weights, have ${free_gb}Gi"
  exit 1
fi

echo "==> creating .venv"
[ -d .venv ] || "$PYTHON" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip

echo "==> installing dependencies (torch 2.2.2 is the last Intel-Mac wheel)"
.venv/bin/pip install -r requirements-mac-cpu.txt
.venv/bin/pip install -r requirements-dev.txt

echo "==> downloading the spaCy tagger misaki's English G2P needs"
.venv/bin/python -m spacy download en_core_web_sm

echo "==> pre-downloading Kokoro weights and all 28 English voices"
.venv/bin/python scripts/bake_assets.py

echo "==> verifying with a real synthesis"
.venv/bin/python - <<'PY'
import time
import numpy as np
import soundfile as sf
from app.engine import KokoroEngine

engine = KokoroEngine(device="cpu")
text = "Kokoro is running locally on this machine."
started = time.perf_counter()
segments = list(engine.iter_segments(text, "af_heart", "a", 1.0))
elapsed = time.perf_counter() - started

audio = np.concatenate([s.audio for s in segments])
duration = len(audio) / engine.sample_rate
sf.write("/tmp/kokoro_hello.wav", audio, engine.sample_rate)
words = [w.word for s in segments for w in s.words]

print(f"    audio:   {duration:.2f}s written to /tmp/kokoro_hello.wav")
print(f"    compute: {elapsed:.2f}s  (real-time factor {elapsed / duration:.2f})")
print(f"    words:   {len(words)} timed -> {' '.join(words)}")
assert np.max(np.abs(audio)) > 0.05, "synthesis produced silence"
PY

echo
echo "Setup complete. Listen:  afplay /tmp/kokoro_hello.wav"
echo "Start the API:           .venv/bin/python -m app"
echo "Open the UI:             http://127.0.0.1:8080/"
```

Then: `chmod +x scripts/setup_mac.sh`

- [ ] **Step 11: Run the setup script**

Run: `./scripts/setup_mac.sh`
Expected: installs finish; the verification block prints a duration, a real-time
factor (likely 1.0–2.5 on this CPU), and the timed words. `/tmp/kokoro_hello.wav`
plays intelligible speech via `afplay /tmp/kokoro_hello.wav`.

If `kokoro` or `transformers` raises on torch 2.2.2, that is the risk called out
in the spec: record the exact error, try `transformers==4.49.0`, and if it still
fails, stop and report — the Mac then becomes a client of the GPU box and the API
contract is unchanged.

- [ ] **Step 12: Run the slow suite**

Run: `KOKORO_RUN_SLOW=1 .venv/bin/pytest tests/test_engine_slow.py -v -s`
Expected: PASS (10 tests). Slow — several minutes on this CPU is normal.

- [ ] **Step 13: Confirm the fast suite is still fast and torch-free**

Run: `.venv/bin/pytest -q --durations=5`
Expected: all non-slow tests pass in a few seconds; the slow module reports as
skipped without importing torch.

- [ ] **Step 14: Commit**

```bash
git add app/engine.py scripts requirements-mac-cpu.txt tests/test_engine_slow.py
git commit -m "feat: add Kokoro PyTorch engine, asset baking, and Mac setup script"
```

---

### Task 11: Web UI with karaoke word highlighting

Vanilla HTML/CSS/JS, no build step, no CDN — it must work offline inside the
container. It drives `/tts/stream` and schedules PCM chunks on the WebAudio
timeline, which both starts playback as soon as chunk 0 lands and gives an exact
clock for the highlight.

**Files:**
- Create: `web/index.html`, `web/styles.css`, `web/app.js`
- Modify: `app/main.py` (mount static files last)
- Test: `tests/test_web_ui.py`

**Interfaces:**
- Consumes: `GET /voices`, `POST /tts/stream`, `POST /tts`, `GET /health`.
- Produces: the UI at `/`. No Python interfaces.

- [ ] **Step 1: Write the failing test**

`tests/test_web_ui.py`:

```python
import re
from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def test_index_is_served_at_the_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Kokoro" in resp.text


def test_assets_are_served(client):
    css = client.get("/styles.css")
    js = client.get("/app.js")
    assert css.status_code == 200
    assert js.status_code == 200
    assert "/tts/stream" in js.text


def test_every_referenced_asset_exists_on_disk():
    html = (WEB_DIR / "index.html").read_text()
    referenced = re.findall(r'(?:src|href)="(?!https?:|//)([^"]+)"', html)
    assert referenced, "expected local asset references"
    for asset in referenced:
        assert (WEB_DIR / asset.lstrip("/")).is_file(), asset


def test_the_ui_pulls_in_no_remote_resources():
    """The container has no network; a CDN reference would break it."""
    html = (WEB_DIR / "index.html").read_text()
    assert "http://" not in html
    assert "https://" not in html


def test_mounting_the_ui_does_not_shadow_the_api(client):
    assert client.get("/voices").status_code == 200
    assert client.post("/tts", json={"text": "hi"}).status_code == 200
    assert client.get("/health").status_code == 200
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `.venv/bin/pytest tests/test_web_ui.py -v`
Expected: FAIL — 404 on `/` and `FileNotFoundError` for `web/index.html`

- [ ] **Step 3: Write `web/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kokoro TTS</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header>
    <h1>Kokoro TTS</h1>
    <span id="badge" class="badge">connecting…</span>
  </header>

  <main>
    <textarea id="text" rows="5" placeholder="Type something to speak. Separate paragraphs with newlines — each one streams as it finishes."></textarea>
    <div class="counter"><span id="count">0</span> characters</div>

    <div class="controls">
      <label>Voice
        <select id="voice"></select>
      </label>
      <label>Accent
        <select id="lang">
          <option value="">from voice</option>
          <option value="a">American</option>
          <option value="b">British</option>
        </select>
      </label>
      <label>Speed <output id="speedOut">1.00</output>
        <input id="speed" type="range" min="0.5" max="2" step="0.05" value="1">
      </label>
      <label>Format
        <select id="format">
          <option value="wav">WAV</option>
          <option value="mp3">MP3</option>
        </select>
      </label>
    </div>

    <details>
      <summary>Advanced</summary>
      <div class="advanced">
        <label class="check">
          <input id="blendOn" type="checkbox"> Blend with a second voice
        </label>
        <label>Second voice
          <select id="voiceB"></select>
        </label>
        <label>Mix <output id="mixOut">50%</output>
          <input id="mix" type="range" min="5" max="95" step="5" value="50">
        </label>
        <label>API key (only if the server sets one)
          <input id="apiKey" type="password" autocomplete="off" placeholder="leave empty for local use">
        </label>
      </div>
    </details>

    <div class="actions">
      <button id="speak" type="button">Speak</button>
      <button id="stop" type="button" class="secondary">Stop</button>
      <a id="download" class="secondary button" download="kokoro.wav" hidden>Download WAV</a>
    </div>

    <p id="status" class="status">Ready.</p>
    <div id="transcript" class="transcript" aria-live="polite"></div>
    <audio id="player" hidden></audio>
  </main>

  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Write `web/styles.css`**

```css
:root {
  --bg: #ffffff;
  --fg: #16181d;
  --muted: #5c6370;
  --line: #d9dce3;
  --accent: #2f6feb;
  --accent-fg: #ffffff;
  --highlight: #ffe08a;
  --panel: #f6f7f9;
  --error: #b42318;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161a;
    --fg: #e8eaed;
    --muted: #9aa1ac;
    --line: #2c3038;
    --accent: #5b8cff;
    --accent-fg: #0d1117;
    --highlight: #7a5c00;
    --panel: #1b1e24;
    --error: #ff7b72;
  }
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1.25rem 1.5rem;
  border-bottom: 1px solid var(--line);
}

h1 { font-size: 1.15rem; margin: 0; }

.badge {
  font-size: 0.75rem;
  color: var(--muted);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0.15rem 0.6rem;
}

main { max-width: 62rem; margin: 0 auto; padding: 1.5rem; }

textarea {
  width: 100%;
  padding: 0.9rem;
  font: inherit;
  color: var(--fg);
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  resize: vertical;
}

.counter { font-size: 0.8rem; color: var(--muted); margin: 0.35rem 0 1rem; }

.controls, .advanced {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr));
  gap: 0.9rem;
}

.advanced { margin-top: 0.9rem; }

label { display: flex; flex-direction: column; gap: 0.3rem; font-size: 0.85rem; color: var(--muted); }
label.check { flex-direction: row; align-items: center; gap: 0.5rem; }

select, input[type="password"] {
  font: inherit;
  padding: 0.45rem;
  color: var(--fg);
  background: var(--bg);
  border: 1px solid var(--line);
  border-radius: 6px;
}

details { margin-top: 1rem; }
summary { cursor: pointer; font-size: 0.85rem; color: var(--muted); }

.actions { display: flex; gap: 0.6rem; align-items: center; margin: 1.4rem 0 0.6rem; }

button, .button {
  font: inherit;
  padding: 0.55rem 1.1rem;
  border: 1px solid transparent;
  border-radius: 6px;
  background: var(--accent);
  color: var(--accent-fg);
  cursor: pointer;
  text-decoration: none;
}

button.secondary, .button.secondary {
  background: transparent;
  color: var(--fg);
  border-color: var(--line);
}

button[disabled] { opacity: 0.55; cursor: default; }

.status { font-size: 0.85rem; color: var(--muted); min-height: 1.4em; }
.status.error { color: var(--error); }

.transcript {
  margin-top: 1rem;
  padding: 1rem;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  min-height: 3rem;
  font-size: 1.1rem;
  line-height: 2;
}

.transcript span { padding: 0.1rem 0.15rem; border-radius: 4px; }
.transcript span.active { background: var(--highlight); font-weight: 600; }
```

- [ ] **Step 5: Write `web/app.js`**

```javascript
"use strict";

const $ = (selector) => document.querySelector(selector);

const state = {
  ctx: null,
  words: [],
  chunks: [],
  sources: [],
  sampleRate: 24000,
  scheduleAt: 0,
  playStart: 0,
  playing: false,
  raf: 0,
};

/* ---------- helpers ---------- */

function setStatus(message, isError) {
  const el = $("#status");
  el.textContent = message;
  el.classList.toggle("error", Boolean(isError));
}

async function api(path, options) {
  const settings = Object.assign({}, options);
  const headers = Object.assign(
    { "Content-Type": "application/json" },
    settings.headers || {}
  );
  const key = $("#apiKey").value.trim();
  if (key) headers.Authorization = "Bearer " + key;
  settings.headers = headers;
  return fetch(path, settings);
}

async function errorMessage(resp) {
  try {
    const body = await resp.json();
    if (body && body.error && body.error.message) return body.error.message;
  } catch (err) {
    /* fall through to the status line below */
  }
  return "Request failed with status " + resp.status;
}

function decodePcm(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function wavBlob(chunks, sampleRate) {
  let length = 0;
  chunks.forEach((c) => { length += c.length; });
  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);
  const ascii = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + length * 2, true);
  ascii(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, length * 2, true);

  let offset = 44;
  chunks.forEach((chunk) => {
    for (let i = 0; i < chunk.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, chunk[i]));
      view.setInt16(offset, sample * 32767, true);
      offset += 2;
    }
  });
  return new Blob([view], { type: "audio/wav" });
}

/* ---------- voices ---------- */

function fillVoiceSelect(select, voices, selectedId) {
  const groups = new Map();
  voices.forEach((voice) => {
    const label = voice.accent + " " + voice.gender;
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label).push(voice);
  });
  select.innerHTML = "";
  Array.from(groups.keys()).sort().forEach((label) => {
    const group = document.createElement("optgroup");
    group.label = label;
    groups.get(label).forEach((voice) => {
      const option = document.createElement("option");
      option.value = voice.id;
      option.textContent = voice.name + " (" + voice.grade + ")";
      if (voice.id === selectedId) option.selected = true;
      group.appendChild(option);
    });
    select.appendChild(group);
  });
}

async function loadVoices() {
  const resp = await api("/voices");
  if (!resp.ok) {
    setStatus(await errorMessage(resp), true);
    return;
  }
  const body = await resp.json();
  const saved = localStorage.getItem("kokoro.voice");
  fillVoiceSelect($("#voice"), body.voices, saved || body.default);
  fillVoiceSelect($("#voiceB"), body.voices, "af_sky");
}

async function loadHealth() {
  try {
    const resp = await fetch("/health");
    const body = await resp.json();
    $("#badge").textContent =
      body.status === "ok"
        ? body.device + " · warmup " + Number(body.warmup_seconds || 0).toFixed(1) + "s"
        : "model loading…";
  } catch (err) {
    $("#badge").textContent = "server unreachable";
  }
}

function voiceSpec() {
  const primary = $("#voice").value;
  if (!$("#blendOn").checked) return primary;
  const mix = Number($("#mix").value) / 100;
  return primary + ":" + (1 - mix).toFixed(2) + "," + $("#voiceB").value + ":" + mix.toFixed(2);
}

function requestBody() {
  return {
    text: $("#text").value,
    voice: voiceSpec(),
    lang: $("#lang").value || null,
    speed: Number($("#speed").value),
    format: $("#format").value,
  };
}

/* ---------- highlighting ---------- */

function findWordIndex(words, seconds) {
  let low = 0;
  let high = words.length - 1;
  let found = -1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (words[mid].start <= seconds) {
      found = mid;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  if (found >= 0 && seconds > words[found].end + 0.15) return -1;
  return found;
}

function appendWord(word) {
  const span = document.createElement("span");
  span.textContent = word.word;
  $("#transcript").appendChild(span);
  $("#transcript").appendChild(document.createTextNode(" "));
}

function startHighlighting(clock) {
  cancelAnimationFrame(state.raf);
  const tick = () => {
    const index = findWordIndex(state.words, clock());
    const spans = $("#transcript").querySelectorAll("span");
    for (let i = 0; i < spans.length; i += 1) {
      spans[i].classList.toggle("active", i === index);
    }
    if (state.playing) state.raf = requestAnimationFrame(tick);
  };
  state.raf = requestAnimationFrame(tick);
}

/* ---------- playback ---------- */

function ensureContext() {
  if (!state.ctx) {
    const Ctor = window.AudioContext || window.webkitAudioContext;
    state.ctx = new Ctor();
  }
  if (state.ctx.state === "suspended") state.ctx.resume();
  return state.ctx;
}

function scheduleChunk(event) {
  const samples = decodePcm(event.audio);
  state.chunks.push(samples);

  const ctx = state.ctx;
  const buffer = ctx.createBuffer(1, samples.length, state.sampleRate);
  buffer.copyToChannel(samples, 0);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);

  if (state.scheduleAt === 0) {
    // Small lead-in so the first chunk is not clipped by scheduling latency.
    state.scheduleAt = ctx.currentTime + 0.12;
    state.playStart = state.scheduleAt;
    state.playing = true;
    startHighlighting(() => state.ctx.currentTime - state.playStart);
  }
  source.start(state.scheduleAt);
  state.scheduleAt += buffer.duration;
  state.sources.push(source);

  (event.words || []).forEach((word) => {
    state.words.push(word);
    appendWord(word);
  });
}

function stop() {
  state.playing = false;
  cancelAnimationFrame(state.raf);
  state.sources.forEach((source) => {
    try { source.stop(); } catch (err) { /* already finished */ }
  });
  state.sources = [];
  state.scheduleAt = 0;
  const player = $("#player");
  player.pause();
  $("#transcript").querySelectorAll("span.active").forEach((span) => {
    span.classList.remove("active");
  });
}

function offerDownload(blob, filename) {
  const link = $("#download");
  if (link.dataset.url) URL.revokeObjectURL(link.dataset.url);
  const url = URL.createObjectURL(blob);
  link.href = url;
  link.dataset.url = url;
  link.download = filename;
  link.hidden = false;
}

/* ---------- synthesis ---------- */

async function speak() {
  const text = $("#text").value.trim();
  if (!text) {
    setStatus("Type something first.", true);
    return;
  }

  stop();
  state.words = [];
  state.chunks = [];
  $("#transcript").innerHTML = "";
  $("#download").hidden = true;
  $("#speak").disabled = true;
  localStorage.setItem("kokoro.voice", $("#voice").value);
  localStorage.setItem("kokoro.key", $("#apiKey").value);

  const started = performance.now();
  let firstAudioAt = null;
  setStatus("Synthesizing…");

  let resp;
  try {
    resp = await api("/tts/stream", {
      method: "POST",
      body: JSON.stringify(requestBody()),
    });
  } catch (err) {
    $("#speak").disabled = false;
    return fallback("stream request failed");
  }

  if (!resp.ok) {
    setStatus(await errorMessage(resp), true);
    $("#speak").disabled = false;
    return;
  }
  if (!resp.body || !(window.AudioContext || window.webkitAudioContext)) {
    $("#speak").disabled = false;
    return fallback("streaming unsupported in this browser");
  }

  ensureContext();
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffered = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffered += decoder.decode(value, { stream: true });
      const lines = buffered.split("\n");
      buffered = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "meta") {
          state.sampleRate = event.sample_rate;
        } else if (event.type === "chunk") {
          if (firstAudioAt === null) firstAudioAt = performance.now();
          scheduleChunk(event);
        } else if (event.type === "error") {
          setStatus("Synthesis failed: " + event.message, true);
          $("#speak").disabled = false;
          return;
        } else if (event.type === "done") {
          reportStats(started, firstAudioAt, event.duration);
        }
      }
    }
  } catch (err) {
    setStatus("Stream interrupted: " + err, true);
  }

  $("#speak").disabled = false;
  if (state.chunks.length) {
    offerDownload(wavBlob(state.chunks, state.sampleRate), "kokoro.wav");
    const tail = (state.scheduleAt - state.ctx.currentTime + 0.4) * 1000;
    setTimeout(() => { state.playing = false; }, Math.max(0, tail));
  }
}

function reportStats(started, firstAudioAt, audioDuration) {
  const total = (performance.now() - started) / 1000;
  const ttfa = firstAudioAt ? (firstAudioAt - started) / 1000 : total;
  const rtf = audioDuration ? total / audioDuration : 0;
  setStatus(
    "audio " + audioDuration.toFixed(2) + "s · first sound " + ttfa.toFixed(2) +
    "s · total " + total.toFixed(2) + "s · RTF " + rtf.toFixed(2)
  );
}

/** Non-streaming path: one /tts call, plain <audio> playback. */
async function fallback(reason) {
  setStatus("Falling back to non-streaming mode (" + reason + ")…");
  const started = performance.now();
  const body = Object.assign(requestBody(), { include_timestamps: true });

  let resp;
  try {
    resp = await api("/tts", { method: "POST", body: JSON.stringify(body) });
  } catch (err) {
    setStatus("Request failed: " + err, true);
    return;
  }
  if (!resp.ok) {
    setStatus(await errorMessage(resp), true);
    return;
  }

  const payload = await resp.json();
  state.words = payload.words || [];
  $("#transcript").innerHTML = "";
  state.words.forEach(appendWord);

  const player = $("#player");
  const mime = payload.format === "mp3" ? "audio/mpeg" : "audio/wav";
  player.src = "data:" + mime + ";base64," + payload.audio;
  player.hidden = false;
  state.playing = true;
  startHighlighting(() => player.currentTime);
  player.onended = () => { state.playing = false; };
  await player.play();

  const bytes = atob(payload.audio);
  const array = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i += 1) array[i] = bytes.charCodeAt(i);
  offerDownload(new Blob([array], { type: mime }), "kokoro." + payload.format);
  reportStats(started, performance.now(), payload.duration);
}

/* ---------- wiring ---------- */

function init() {
  $("#apiKey").value = localStorage.getItem("kokoro.key") || "";
  $("#text").value = "Kokoro is an open weight text to speech model with eighty two million parameters.";
  $("#count").textContent = $("#text").value.length;

  $("#text").addEventListener("input", () => {
    $("#count").textContent = $("#text").value.length;
  });
  $("#speed").addEventListener("input", () => {
    $("#speedOut").textContent = Number($("#speed").value).toFixed(2);
  });
  $("#mix").addEventListener("input", () => {
    $("#mixOut").textContent = $("#mix").value + "%";
  });
  $("#speak").addEventListener("click", speak);
  $("#stop").addEventListener("click", stop);
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") speak();
  });

  loadVoices();
  loadHealth();
}

document.addEventListener("DOMContentLoaded", init);
```

- [ ] **Step 6: Mount the UI in `app/main.py`**

Add the import:

```python
from fastapi.staticfiles import StaticFiles
```

Add the constant below the logger:

```python
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
```

And mount it as the **last** thing in `create_app`, after all `include_router`
calls — a mount at `/` registered earlier would shadow the API:

```python
    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
```

- [ ] **Step 7: Run the test**

Run: `.venv/bin/pytest tests/test_web_ui.py -v`
Expected: PASS (5 tests)

- [ ] **Step 8: Verify the UI by hand against the real model**

Run: `.venv/bin/python -m app`

Then open `http://127.0.0.1:8080/` and check:

1. The badge shows `cpu · warmup Ns`; the voice picker lists 28 voices in four groups.
2. Clicking **Speak** starts audio before the whole paragraph is finished (watch the status line).
3. Words highlight roughly in time with the speech, and the highlight does not drift on the **second and third** paragraphs — this is the segment-offset behaviour under test.
4. **Stop** silences playback immediately.
5. **Download WAV** saves a file that plays in QuickTime.
6. Enabling **Blend with a second voice** audibly changes the timbre.
7. Reloading keeps the chosen voice (localStorage).

- [ ] **Step 9: Commit**

```bash
git add web app/main.py tests/test_web_ui.py
git commit -m "feat: add web UI with streaming playback and karaoke highlighting"
```

---

### Task 12: CUDA Docker image, compose file, and README

The image build and GPU run can only be *verified* on the Windows 11 / RTX 2070
machine — the Mac has no running Docker daemon and no NVIDIA GPU. So this task
splits into checks that run here and checks that run there; the task is not done
until the Windows box reports `"device": "cuda"`.

**Files:**
- Create: `app/__main__.py`, `requirements-gpu.txt`, `docker/Dockerfile.cuda`, `docker/docker-compose.gpu.yml`, `.dockerignore`, `README.md`
- Test: `tests/test_deployment_files.py`

**Interfaces:**
- Consumes: `get_settings` from `app.config`; `scripts/bake_assets.py`, `app/`, `web/`, `requirements-base.txt`.
- Produces: `python -m app` launcher honoring `KOKORO_HOST`/`KOKORO_PORT`; an image serving the same API on port 8080 with `KOKORO_DEVICE=cuda`.

- [ ] **Step 1: Write the failing deployment-file test**

These assertions are cheap and catch the mistakes that otherwise surface as a
five-minute failed build on the other machine.

`tests/test_deployment_files.py`:

```python
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "docker" / "Dockerfile.cuda"
COMPOSE = ROOT / "docker" / "docker-compose.gpu.yml"


def test_dockerfile_copies_only_paths_that_exist():
    for line in DOCKERFILE.read_text().splitlines():
        if not line.startswith("COPY "):
            continue
        parts = line.split()[1:]
        for source in parts[:-1]:
            if source.startswith("--"):
                continue
            assert (ROOT / source).exists(), source


def test_dockerfile_does_not_reinstall_torch():
    """torch comes from the CUDA base image; reinstalling risks a CPU-only wheel."""
    text = DOCKERFILE.read_text()
    assert "pip install" in text
    assert not re.search(r"pip install[^\n]*\btorch\b", text)


def test_gpu_requirements_exclude_torch_and_include_kokoro():
    text = (ROOT / "requirements-gpu.txt").read_text()
    assert "kokoro==0.9.4" in text
    assert not re.search(r"^torch[=<>]", text, re.MULTILINE)


def test_image_binds_all_interfaces():
    """A 127.0.0.1 bind inside the container is unreachable from the host."""
    assert "KOKORO_HOST=0.0.0.0" in DOCKERFILE.read_text()
    assert "KOKORO_HOST: 0.0.0.0" in COMPOSE.read_text()


def test_compose_reserves_the_gpu():
    text = COMPOSE.read_text()
    assert "driver: nvidia" in text
    assert "capabilities: [gpu]" in text
    assert "KOKORO_DEVICE: cuda" in text


def test_compose_is_valid_yaml_with_one_service():
    try:
        import yaml
    except ImportError:
        import pytest

        pytest.skip("pyyaml not installed")
    config = yaml.safe_load(COMPOSE.read_text())
    assert list(config["services"]) == ["kokoro"]
    assert config["services"]["kokoro"]["ports"] == ["8080:8080"]


def test_dockerignore_excludes_the_local_venv():
    text = (ROOT / ".dockerignore").read_text()
    assert ".venv" in text


def test_readme_documents_both_deployments():
    text = (ROOT / "README.md").read_text()
    assert "setup_mac.sh" in text
    assert "docker-compose.gpu.yml" in text
    assert "/v1/audio/speech" in text
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `.venv/bin/pytest tests/test_deployment_files.py -v`
Expected: FAIL — `FileNotFoundError` for `docker/Dockerfile.cuda`

- [ ] **Step 3: Add the `python -m app` launcher**

`KOKORO_HOST` and `KOKORO_PORT` exist in `Settings` but nothing reads them yet —
a `uvicorn` command line bypasses them. This makes them real, and gives the
image a single entrypoint that respects its own environment.

`app/__main__.py`:

```python
"""Launcher that honors KOKORO_HOST / KOKORO_PORT: python -m app"""
import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
```

Verify it reads the environment:

```bash
KOKORO_PORT=8123 .venv/bin/python -m app &
sleep 8 && curl -fsS http://127.0.0.1:8123/health && kill %1
```

Expected: the health JSON comes back on 8123, not 8080.

- [ ] **Step 5: Write `requirements-gpu.txt`**

```
-r requirements-base.txt

# torch is provided by the CUDA base image (2.5.1+cu121). Do NOT add it here:
# pip would fetch a CPU-only wheel and silently disable the GPU.
transformers>=4.44,<5
kokoro==0.9.4
misaki[en]>=0.9.4
```

- [ ] **Step 4: Write `docker/Dockerfile.cuda`**

```dockerfile
# RTX 2070 is Turing (SM 7.5) — fully supported by cu121 builds.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    KOKORO_DEVICE=cuda \
    KOKORO_HOST=0.0.0.0 \
    KOKORO_PORT=8080

# espeak-ng as belt and braces alongside the espeakng-loader wheel; curl for HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-base.txt requirements-gpu.txt ./
RUN pip install --no-cache-dir -r requirements-gpu.txt

# misaki's English G2P needs the spaCy tagger.
RUN python -m spacy download en_core_web_sm

COPY app ./app
COPY web ./web
COPY scripts ./scripts

# Bake the weights and all 28 English voices so the container starts offline.
RUN python scripts/bake_assets.py

RUN useradd --create-home --uid 10001 kokoro \
    && chown -R kokoro:kokoro /opt/hf /app
USER kokoro

EXPOSE 8080
# start-period covers model load + warm-up on first boot.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

# Uses the launcher so KOKORO_HOST / KOKORO_PORT above are honored.
CMD ["python", "-m", "app"]
```

- [ ] **Step 6: Write `docker/docker-compose.gpu.yml`**

```yaml
services:
  kokoro:
    build:
      context: ..
      dockerfile: docker/Dockerfile.cuda
    image: kokoro-tts-api:latest
    container_name: kokoro-tts
    ports:
      - "8080:8080"
    environment:
      KOKORO_DEVICE: cuda
      KOKORO_HOST: 0.0.0.0
      KOKORO_MAX_CONCURRENCY: "2"
      # Uncomment before exposing this beyond localhost:
      # KOKORO_API_KEY: change-me
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

- [ ] **Step 7: Write `.dockerignore`**

```
.venv
.git
.gitignore
__pycache__
**/__pycache__
*.pyc
.pytest_cache
docs
tests
pytest.ini
requirements-mac-cpu.txt
requirements-dev.txt
scripts/setup_mac.sh
```

- [ ] **Step 8: Write `README.md`**

````markdown
# Kokoro TTS API

Self-hosted English text-to-speech built on [Kokoro-82M](https://github.com/hexgrad/kokoro),
with an OpenAI-compatible endpoint, word-level timestamps, streaming, and a
browser UI. Runs CPU-native on Intel macOS or GPU-accelerated in Docker.

## Quick start — macOS (CPU)

```bash
./scripts/setup_mac.sh
.venv/bin/python -m app
```

Open <http://127.0.0.1:8080/>.

The pins in `requirements-mac-cpu.txt` are deliberate: `torch==2.2.2` is the last
release with an Intel-Mac wheel, and `numpy<2` is what that torch was built
against. Do not float them.

Expect a real-time factor around 1.0–2.5 on a 2016 quad-core CPU — roughly 10–25
seconds of compute per 10 seconds of audio. The streaming endpoint and the UI
start playing before synthesis finishes, which hides most of that.

## Quick start — Windows 11 + NVIDIA GPU (Docker)

Prerequisites: Docker Desktop with the WSL2 backend, and a current NVIDIA
driver. No CUDA toolkit is needed on the host.

```powershell
cd docker
docker compose -f docker-compose.gpu.yml up --build -d
curl http://localhost:8080/health
```

`"device": "cuda"` in the response means the GPU is in use. Open
<http://localhost:8080/> for the UI.

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/audio/speech` | OpenAI-compatible. Drop-in for OpenAI TTS clients. |
| `POST /tts` | Native. Voice blending, word timestamps. |
| `POST /tts/stream` | NDJSON stream of PCM chunks + timings. |
| `GET /voices` | The 28 English voices with quality grades. |
| `GET /health` | Device, backend, warm-up time. |

### OpenAI-compatible

```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","input":"Hello from Kokoro.","voice":"nova","response_format":"mp3"}' \
  --output hello.mp3
```

OpenAI voice names map onto Kokoro voices: `alloy`, `echo`, `fable`, `onyx`,
`nova`, `shimmer`. Any real Kokoro id works too. `model` is accepted and ignored.

### Native, with word timestamps

```bash
curl -X POST http://127.0.0.1:8080/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello there.","voice":"af_heart","include_timestamps":true}'
```

```json
{
  "audio": "<base64 wav>",
  "format": "wav",
  "sample_rate": 24000,
  "duration": 1.35,
  "voice": "af_heart",
  "words": [{"word": "Hello", "start": 0.05, "end": 0.48},
            {"word": "there", "start": 0.52, "end": 0.94}]
}
```

Without `include_timestamps`, the response is raw audio bytes.

### Voice blending

```bash
curl -X POST http://127.0.0.1:8080/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"A blended voice.","voice":"af_bella:0.6,af_sky:0.4"}' \
  --output blend.wav
```

Up to four voices. Weights are normalized; omitting them averages equally
(`af_bella,af_sky`).

### Streaming

```bash
curl -N -X POST http://127.0.0.1:8080/tts/stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"First line.\nSecond line."}'
```

One JSON object per line: a `meta` line, a `chunk` line per segment (base64
float32 PCM at 24 kHz plus absolute word timings), then `done`. The `format`
field is ignored here — chunks are always raw PCM.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `KOKORO_DEVICE` | `auto` | `auto` / `cpu` / `cuda` |
| `KOKORO_DEFAULT_VOICE` | `af_heart` | Voice when a request omits one |
| `KOKORO_MAX_CHARS` | `5000` | Per-request input cap |
| `KOKORO_MAX_CONCURRENCY` | 1 (CPU) / 2 (CUDA) | Concurrent synthesis permits |
| `KOKORO_TORCH_THREADS` | torch default | `torch.set_num_threads` |
| `KOKORO_API_KEY` | unset | Requires `Authorization: Bearer` when set |
| `KOKORO_HOST` / `KOKORO_PORT` | `127.0.0.1` / `8080` | Bind address |
| `KOKORO_ALLOW_ORIGINS` | unset | Comma-separated CORS origins |
| `HF_HOME` | platform default | Weights/voices cache location |

Binding to the LAN? Set `KOKORO_API_KEY` as well — `/health` stays public, every
synthesis route requires the bearer token.

## Tests

```bash
.venv/bin/pytest                              # fast: no model load, seconds
KOKORO_RUN_SLOW=1 .venv/bin/pytest -m slow    # loads the real model
```

## Troubleshooting

- **`KOKORO_DEVICE=cuda but torch reports no CUDA device`** — the container did
  not get the GPU. Check `nvidia-smi` on the host and that Docker Desktop's WSL2
  backend is enabled.
- **Synthesis feels slow on the Mac** — expected; see the real-time factor note
  above. Use `/tts/stream`, or point the UI at the GPU box.
- **`OSError: [E050] Can't find model 'en_core_web_sm'`** — run
  `.venv/bin/python -m spacy download en_core_web_sm`.
- **Timestamps drift slightly** — Kokoro derives them from predicted phoneme
  durations, so they are approximate by design. Whole-word highlighting absorbs it.
````

- [ ] **Step 9: Run the deployment test and the full suite**

Run: `.venv/bin/pytest -q`
Expected: PASS — every fast test including the 8 deployment-file checks.

- [ ] **Step 10: Verify on the Windows 11 / RTX 2070 machine**

Copy the repo over (or `git clone` it), then in PowerShell:

```powershell
cd docker
docker compose -f docker-compose.gpu.yml up --build -d
docker compose -f docker-compose.gpu.yml logs -f    # watch for "ready on cuda"
```

Then confirm, expecting `"device": "cuda"` and a warm-up well under a second:

```powershell
curl http://localhost:8080/health
curl -X POST http://localhost:8080/tts -H "Content-Type: application/json" `
  -d '{\"text\":\"Running on the GPU now.\"}' --output gpu.wav
```

Finally open <http://localhost:8080/> and repeat the UI checks from Task 11
Step 8. If the build or GPU handoff fails, report the exact error — do not
paper over it by falling back to `KOKORO_DEVICE=cpu` in the image.

- [ ] **Step 11: Commit**

```bash
git add app/__main__.py requirements-gpu.txt docker .dockerignore README.md tests/test_deployment_files.py
git commit -m "feat: add CUDA Docker deployment, module launcher, and project README"
```

---

## Acceptance

The project is done when all of these hold:

1. `.venv/bin/pytest -q` passes with no torch import and no model download.
2. `KOKORO_RUN_SLOW=1 .venv/bin/pytest -m slow` passes on the Mac.
3. `./scripts/setup_mac.sh` completes on a clean checkout and prints a real-time factor.
4. The UI at `http://127.0.0.1:8080/` plays audio with word highlighting that stays in sync **past the first paragraph**.
5. `curl` examples in the README work as written against the Mac instance.
6. `docker compose -f docker/docker-compose.gpu.yml up --build` on the Windows box serves `/health` with `"device": "cuda"`, and the UI works there too.
