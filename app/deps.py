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
