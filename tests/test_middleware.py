import asyncio
import json
from typing import Any

from starlette.responses import PlainTextResponse
from starlette.types import Message, Receive, Scope, Send

from app.middleware import MULTIPART_ALLOWANCE_BYTES, BodySizeLimitMiddleware

MB = 1024 * 1024
LIMIT = MB + MULTIPART_ALLOWANCE_BYTES


async def body_reading_app(scope: Scope, receive: Receive, send: Send) -> None:
    """Reads the whole body like FastAPI does, turning read errors into a 400 like FastAPI."""
    body = b""
    try:
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
    except Exception:
        await PlainTextResponse("error parsing the body", status_code=400)(scope, receive, send)
        return
    await PlainTextResponse(f"received {len(body)}")(scope, receive, send)


def run(chunks: list[bytes], content_length: int | None = None) -> dict[str, Any]:
    """Drives the middleware with raw ASGI messages and records what it sends back."""
    headers = [] if content_length is None else [(b"content-length", str(content_length).encode())]
    scope = {"type": "http", "method": "POST", "path": "/documents", "headers": headers}
    incoming = [
        {"type": "http.request", "body": chunk, "more_body": index < len(chunks) - 1}
        for index, chunk in enumerate(chunks)
    ]
    reads = 0
    sent: list[Message] = []

    async def receive() -> Message:
        nonlocal reads
        reads += 1
        return incoming.pop(0) if incoming else {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    asyncio.run(BodySizeLimitMiddleware(body_reading_app, max_upload_mb=1)(scope, receive, send))
    starts = [message for message in sent if message["type"] == "http.response.start"]
    body = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    return {"starts": starts, "body": body, "reads": reads}


def test_body_within_limit_passes_through() -> None:
    result = run([b"a" * 1000, b"b" * 1000], content_length=2000)

    assert [start["status"] for start in result["starts"]] == [200]
    assert result["body"] == b"received 2000"


def test_declared_oversized_body_is_rejected_without_reading_it() -> None:
    result = run([b"a"], content_length=LIMIT + 1)

    assert [start["status"] for start in result["starts"]] == [413]
    assert json.loads(result["body"])["error"] == {
        "code": "FILE_TOO_LARGE",
        "message": "Upload exceeds the maximum size of 1 MB.",
        "details": {"max_bytes": MB},
    }
    assert result["reads"] == 0


def test_streamed_body_without_length_is_cut_off_once_over_the_limit() -> None:
    chunk = b"a" * (256 * 1024)

    result = run([chunk] * 8)

    # Exactly one response, the 413; the app's own 400 after the failed read is discarded.
    assert [start["status"] for start in result["starts"]] == [413]
    assert json.loads(result["body"])["error"]["code"] == "FILE_TOO_LARGE"
    assert result["reads"] == 5  # 5 x 256 KB is the first total over 1 MB + 64 KB


def test_non_http_scopes_are_passed_through() -> None:
    seen: list[str] = []

    async def lifespan_app(scope: Scope, _receive: Receive, _send: Send) -> None:
        seen.append(scope["type"])

    async def noop_receive() -> Message:
        return {}

    async def noop_send(_message: Message) -> None:
        return None

    asyncio.run(
        BodySizeLimitMiddleware(lifespan_app, max_upload_mb=1)(
            {"type": "lifespan"}, noop_receive, noop_send
        )
    )

    assert seen == ["lifespan"]
