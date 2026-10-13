from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.errors import FileTooLargeError, error_response

# Multipart framing (boundaries, part headers, the name field) adds a little on top of the
# file itself; the upload handler then checks the file size exactly.
MULTIPART_ALLOWANCE_BYTES = 64 * 1024


class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Rejects request bodies over the upload limit with 413 before they are fully received.

    A plain ASGI middleware rather than BaseHTTPMiddleware, because it must see the body as it
    streams in. FastAPI turns any exception raised while it parses a body into a generic 400,
    so on overflow this middleware sends the 413 itself and discards what the app sends after.
    """

    def __init__(self, app: ASGIApp, max_upload_mb: int) -> None:
        self.app = app
        self.max_upload_mb = max_upload_mb
        self.max_body_bytes = max_upload_mb * 1024 * 1024 + MULTIPART_ALLOWANCE_BYTES

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared is not None and declared > self.max_body_bytes:
            await self._reject(scope, receive, send)
            return

        received = 0
        rejected = False

        async def limited_receive() -> Message:
            nonlocal received, rejected
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    rejected = True
                    await self._reject(scope, receive, send)
                    raise _BodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            if not rejected:
                await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _BodyTooLarge:
            pass

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        error = FileTooLargeError.for_limit(self.max_upload_mb)
        response = error_response(error.status_code, error.code, error.message, error.details)
        await response(scope, receive, send)


def _content_length(scope: Scope) -> int | None:
    for name, value in scope["headers"]:
        if name == b"content-length":
            return int(value) if value.isdigit() else None
    return None
