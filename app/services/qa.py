from collections.abc import Callable
from dataclasses import dataclass

from app.config import Settings
from app.services.llm import LLM
from app.services.prompts import REFUSAL, build_prompt, drop_invalid_citations, is_refusal
from app.services.retrieval import RetrievedChunk, Retriever


@dataclass(frozen=True)
class Answer:
    text: str
    # Exactly the chunks shown to the model: citation [n] refers to sources[n - 1].
    sources: list[RetrievedChunk]
    model: str
    tokens_used: int


class QAService:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        llm: LLM,
        count_tokens: Callable[[str], int],
    ) -> None:
        self._settings = settings
        self._retriever = retriever
        self._llm = llm
        self._count_tokens = count_tokens

    def ask(self, question: str, top_k: int, document_ids: list[str] | None = None) -> Answer:
        chunks = self._retriever.retrieve(question, top_k, document_ids)
        if not chunks:
            # Nothing relevant enough: refuse without calling the LLM, so nothing can be invented.
            return Answer(REFUSAL, [], self._settings.llm_model, 0)

        prompt = build_prompt(
            question, chunks, self._settings.max_context_tokens, self._count_tokens
        )
        if not prompt.sources:
            # Question / labels / system prompt filled the budget; calling the LLM would
            # send a prompt with no sources, or one Ollama would silently truncate.
            return Answer(REFUSAL, [], self._settings.llm_model, 0)

        generation = self._llm.generate(prompt.system, prompt.user)
        text = drop_invalid_citations(generation.text, len(prompt.sources))
        if not text or is_refusal(text):
            return Answer(REFUSAL, [], generation.model, generation.tokens_used)
        return Answer(text, prompt.sources, generation.model, generation.tokens_used)
