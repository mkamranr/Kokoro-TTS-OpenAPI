import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth import require_api_key
from app.config import get_settings
from app.errors import install_error_handlers
from app.routes import health, native, openai

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
    app.include_router(native.router, dependencies=[Depends(require_api_key)])
    app.include_router(openai.router, dependencies=[Depends(require_api_key)])
    return app


app = create_app()
