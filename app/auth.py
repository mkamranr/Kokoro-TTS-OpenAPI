"""Optional bearer-token auth. Unset KOKORO_API_KEY means open access."""
import hmac

from fastapi import Depends, Request

from app.config import Settings
from app.deps import get_settings_dep
from app.errors import ApiError


def _digest(value: str) -> bytes:
    """Encode for hmac.compare_digest without ever raising.

    compare_digest refuses str arguments that contain non-ASCII characters,
    and ASGI decodes header bytes as latin-1 -- so a single 0xe9 byte in an
    Authorization header used to reach it as "é" and blow up with a TypeError,
    turning what should be a 401 into a 500 plus a logged stack trace that any
    unauthenticated caller could trigger at will.

    utf-8 with surrogateescape encodes every str Python can hold: latin-1
    decoded header values, and lone surrogates from os.environ on POSIX. The
    comparison stays constant-time because both sides end up as bytes.
    """
    return value.encode("utf-8", "surrogateescape")


async def require_api_key(
    request: Request, settings: Settings = Depends(get_settings_dep)
) -> None:
    expected = settings.api_key
    if not expected:
        return

    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    # token.strip(): HTTP allows optional whitespace around a header value, so
    # a token differing only in surrounding whitespace is the same token.
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        _digest(token.strip()), _digest(expected)
    ):
        raise ApiError(401, "Invalid or missing API key", "authentication_error")
