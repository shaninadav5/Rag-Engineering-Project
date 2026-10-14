import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.errors import LLMError, LLMUnavailableError
from app.services.llm import Generation, OllamaClient

Handler = Callable[[httpx.Request], httpx.Response]


def make_client(tmp_path: Path, handler: Handler) -> OllamaClient:
    settings = Settings(_env_file=None, data_dir=tmp_path, llm_timeout_seconds=30)
    return OllamaClient(settings, transport=httpx.MockTransport(handler))


def ollama_reply(content: str = "Revenue grew 12% [1].", **extra: Any) -> Handler:
    body = {"model": "llama3.2:3b", "message": {"role": "assistant", "content": content}}
    body.update({"prompt_eval_count": 120, "eval_count": 30, "done": True})
    body.update(extra)
    return lambda _request: httpx.Response(200, json=body)


def raise_error(error: Exception) -> Handler:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise error

    return handler


def test_request_sends_messages_without_streaming_and_with_explicit_context(
    tmp_path: Path,
) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return ollama_reply()(request)

    make_client(tmp_path, handler).generate("system rules", "sources and question")

    [request] = captured
    assert (request.method, request.url.path) == ("POST", "/api/chat")
    assert json.loads(request.content) == {
        "model": "llama3.2:3b",
        "messages": [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "sources and question"},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": 512},
    }


def test_successful_reply_is_mapped_to_a_generation(tmp_path: Path) -> None:
    client = make_client(tmp_path, ollama_reply("  Revenue grew 12% [1].\n"))

    assert client.generate("s", "u") == Generation("Revenue grew 12% [1].", "llama3.2:3b", 150)


def test_prompt_near_the_context_limit_logs_a_truncation_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Default num_ctx=8192, num_predict=512; 7680 + 512 == 8192.
    with caplog.at_level(logging.WARNING, logger="app.services.llm"):
        make_client(tmp_path, ollama_reply(prompt_eval_count=7680)).generate("s", "u")

    assert "truncated" in caplog.text
    assert "7680" in caplog.text


def test_prompt_well_under_the_context_limit_does_not_warn(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="app.services.llm"):
        make_client(tmp_path, ollama_reply(prompt_eval_count=120)).generate("s", "u")

    assert caplog.text == ""


def test_missing_prompt_token_count_counts_as_zero(tmp_path: Path) -> None:
    client = make_client(tmp_path, ollama_reply(prompt_eval_count=None))

    assert client.generate("s", "u").tokens_used == 30


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (httpx.ConnectError("refused"), "service at http://localhost:11434 is unreachable"),
        (httpx.ReadTimeout("slow"), "did not respond within 30 seconds"),
        (httpx.ConnectTimeout("slow"), "did not respond within 30 seconds"),
    ],
)
def test_connection_problems_mean_the_llm_is_unavailable(
    tmp_path: Path, error: Exception, message: str
) -> None:
    with pytest.raises(LLMUnavailableError, match=message):
        make_client(tmp_path, raise_error(error)).generate("s", "u")


def test_model_not_pulled_names_the_pull_command(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'llama3.2:3b' not found"})

    with pytest.raises(LLMUnavailableError) as error:
        make_client(tmp_path, handler).generate("s", "u")

    assert error.value.message == (
        "Model 'llama3.2:3b' is not available. Run: ollama pull llama3.2:3b"
    )
    assert error.value.status_code == 503


def test_other_http_errors_are_llm_errors(tmp_path: Path) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "out of memory"})

    with pytest.raises(LLMError) as error:
        make_client(tmp_path, handler).generate("s", "u")

    assert error.value.details == {"status": 500}
    assert error.value.status_code == 502


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"model": "llama3.2:3b"}),
        httpx.Response(200, json={"message": None}),
        httpx.Response(200, json={"message": {"content": 42}}),
        httpx.Response(200, json=["unexpected", "list"]),
    ],
)
def test_malformed_replies_are_llm_errors(tmp_path: Path, response: httpx.Response) -> None:
    with pytest.raises(LLMError, match="invalid response"):
        make_client(tmp_path, lambda _request: response).generate("s", "u")


def test_empty_answer_is_an_llm_error(tmp_path: Path) -> None:
    with pytest.raises(LLMError, match="empty answer"):
        make_client(tmp_path, ollama_reply("   ")).generate("s", "u")
