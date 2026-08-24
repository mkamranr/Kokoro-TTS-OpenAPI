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
