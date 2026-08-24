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
