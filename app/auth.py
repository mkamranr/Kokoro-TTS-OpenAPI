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
