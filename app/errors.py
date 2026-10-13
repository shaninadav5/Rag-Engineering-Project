from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class AppError(Exception):
    status_code = 500
    code = "INTERNAL_ERROR"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class EmptyDocumentError(AppError):
    status_code = 400
    code = "EMPTY_DOCUMENT"


class UnreadableDocumentError(AppError):
    status_code = 400
    code = "UNREADABLE_DOCUMENT"


class FileTooLargeError(AppError):
    status_code = 413
    code = "FILE_TOO_LARGE"


class TooManyPagesError(AppError):
    status_code = 413
    code = "TOO_MANY_PAGES"


class UnsupportedFileTypeError(AppError):
    status_code = 415
    code = "UNSUPPORTED_FILE_TYPE"


class DocumentNotFoundError(AppError):
    status_code = 404
    code = "DOCUMENT_NOT_FOUND"


class DocumentNotReadyError(AppError):
    status_code = 409
    code = "DOCUMENT_NOT_READY"


class LLMError(AppError):
    status_code = 502
    code = "LLM_ERROR"


class LLMUnavailableError(AppError):
    status_code = 503
    code = "LLM_UNAVAILABLE"


class EmbeddingUnavailableError(AppError):
    status_code = 503
    code = "EMBEDDING_UNAVAILABLE"


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "details": details or {}}},
        headers=headers,
    )


async def _handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
    return error_response(exc.status_code, exc.code, exc.message, exc.details)


async def _handle_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {"field": ".".join(str(part) for part in err["loc"]), "message": err["msg"]}
        for err in exc.errors()
    ]
    return error_response(422, "VALIDATION_ERROR", "Request validation failed.", {"errors": errors})


async def _handle_http_exception(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return error_response(
        exc.status_code, "HTTP_ERROR", str(exc.detail), headers=getattr(exc, "headers", None)
    )


async def _handle_unexpected_error(_request: Request, _exc: Exception) -> JSONResponse:
    # Starlette re-raises the exception after this handler returns, so the server
    # still logs the full traceback; the client only gets a generic message.
    return error_response(500, "INTERNAL_ERROR", "An unexpected error occurred.")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _handle_app_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(Exception, _handle_unexpected_error)
