from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.config import Settings
from app.errors import LLMError, LLMUnavailableError

_CONNECT_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class Generation:
    text: str
    model: str
    tokens_used: int


class LLM(Protocol):
    def generate(self, system: str, user: str) -> Generation: ...


class OllamaClient:
    """Calls Ollama's /api/chat and maps every failure to LLMUnavailableError or LLMError."""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self._base_url = settings.ollama_base_url
        self._model = settings.llm_model
        self._timeout_seconds = settings.llm_timeout_seconds
        # num_ctx must always be sent: Ollama's default context window is small, and it silently
        # drops the start of longer prompts, which is where the sources are.
        self._options = {
            "temperature": settings.llm_temperature,
            "num_ctx": settings.llm_num_ctx,
            "num_predict": settings.llm_num_predict,
        }
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout_seconds, connect=_CONNECT_TIMEOUT_SECONDS),
            transport=transport,
        )

    def generate(self, system: str, user: str) -> Generation:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": self._options,
        }
        try:
            response = self._client.post("/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise LLMUnavailableError(
                f"The language model did not respond within {self._timeout_seconds:g} seconds."
            ) from exc
        except httpx.TransportError as exc:
            raise LLMUnavailableError(
                f"The language model service at {self._base_url} is unreachable."
            ) from exc

        if response.status_code == 404:
            raise LLMUnavailableError(
                f"Model '{self._model}' is not available. Run: ollama pull {self._model}"
            )
        if response.status_code != 200:
            raise LLMError(
                f"The language model returned an error (HTTP {response.status_code}).",
                {"status": response.status_code},
            )
        return self._parse(response)

    def close(self) -> None:
        self._client.close()

    def _parse(self, response: httpx.Response) -> Generation:
        try:
            body: dict[str, Any] = response.json()
            text = body["message"]["content"].strip()
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise LLMError("The language model returned an invalid response.") from exc
        if not text:
            raise LLMError("The language model returned an empty answer.")
        # Ollama omits prompt_eval_count when the prompt was served from its cache.
        tokens_used = int(body.get("prompt_eval_count") or 0) + int(body.get("eval_count") or 0)
        return Generation(
            text=text, model=str(body.get("model") or self._model), tokens_used=tokens_used
        )
