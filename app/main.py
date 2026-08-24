import asyncio
import base64
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

import app as app_package
from app.auth import require_api_key
from app.config import get_settings, resolve_concurrency
from app.errors import install_error_handlers
from app.routes import health, native, openai
from app.service import SynthesisService
from app.voices import VOICES_BY_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# Swagger UI / ReDoc, vendored by scripts/fetch_docs_assets.py. FastAPI's
# defaults point at cdn.jsdelivr.net, which renders /docs and /redoc blank in
# the offline container. Served by the StaticFiles mount below -- no new mount.
SWAGGER_JS_URL = "/vendor/swagger-ui-bundle.js"
SWAGGER_CSS_URL = "/vendor/swagger-ui.css"
REDOC_JS_URL = "/vendor/redoc.standalone.js"

# A data: URI favicon instead of FastAPI's fastapi.tiangolo.com default: no
# remote request, and no extra file to ship. Base64 so nothing in the SVG has
# to be escaped for the HTML attribute it lands in.
_FAVICON_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    "<rect width='32' height='32' rx='7' fill='#2b6cb0'/>"
    "<text x='16' y='23' text-anchor='middle' font-family='sans-serif'"
    " font-size='19' font-weight='700' fill='#ffffff'>K</text></svg>"
)
FAVICON_DATA_URI = "data:image/svg+xml;base64," + base64.b64encode(
    _FAVICON_SVG.encode("utf-8")
).decode("ascii")


def validate_default_voice(voice: str) -> None:
    """Fail fast on a typo'd KOKORO_DEFAULT_VOICE.

    Unvalidated, a bad value is invisible until traffic arrives: every request
    that omits a voice 400s with "Unknown voice", which points at the caller
    rather than at the misconfiguration.
    """
    if voice in VOICES_BY_ID:
        return
    raise RuntimeError(
        f"KOKORO_DEFAULT_VOICE={voice!r} is not a known voice id. "
        f"Valid ids: {', '.join(sorted(VOICES_BY_ID))}"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Before the model load, so a typo costs a second rather than a minute.
    validate_default_voice(settings.default_voice)
    if app.state.load_model:
        from app.engine import KokoroEngine  # local: keeps torch out of tests

        # Logged (and WARNED about, if it leaked outside the project) before
        # anything can download into it.
        app_package.log_cache_location()
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


def _register_offline_docs(app: FastAPI) -> None:
    """/docs and /redoc, served entirely from vendored assets.

    Registered before the StaticFiles mount (which must stay last) so the
    mount cannot shadow them.
    """

    @app.get("/docs", include_in_schema=False)
    async def swagger_ui() -> HTMLResponse:
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} — Swagger UI",
            swagger_js_url=SWAGGER_JS_URL,
            swagger_css_url=SWAGGER_CSS_URL,
            swagger_favicon_url=FAVICON_DATA_URI,
        )

    @app.get("/redoc", include_in_schema=False)
    async def redoc() -> HTMLResponse:
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} — ReDoc",
            redoc_js_url=REDOC_JS_URL,
            redoc_favicon_url=FAVICON_DATA_URI,
            # Otherwise ReDoc injects a fonts.googleapis.com stylesheet.
            with_google_fonts=False,
        )


def create_app(load_model: bool = False) -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Kokoro TTS API",
        version="1.0.0",
        lifespan=lifespan,
        # Replaced by _register_offline_docs below; FastAPI's built-ins are
        # CDN-backed and therefore blank in the offline container.
        docs_url=None,
        redoc_url=None,
    )
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
    _register_offline_docs(app)

    # MUST stay last: a mount at "/" matches everything under it, so anything
    # registered after it is unreachable. tests/test_web_ui.py pins this.
    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app(load_model=True)
