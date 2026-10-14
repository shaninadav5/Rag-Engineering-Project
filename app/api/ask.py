from fastapi import APIRouter
from fastapi.exceptions import RequestValidationError

from app.config import Settings
from app.dependencies import QAServiceDep, SettingsDep
from app.schemas import AskRequest, AskResponse, Source

router = APIRouter(tags=["questions"])


# A plain `def` endpoint: embedding the question and waiting for the LLM run in a worker thread.
@router.post("/ask")
def ask(request: AskRequest, qa: QAServiceDep, settings: SettingsDep) -> AskResponse:
    _check_limits(request, settings)
    top_k = request.top_k if request.top_k is not None else settings.top_k
    answer = qa.ask(request.question, top_k, request.document_ids)
    return AskResponse(
        answer=answer.text,
        sources=[
            Source(
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                chunk_text=chunk.text,
                page=chunk.page_start,
                page_end=chunk.page_end,
                relevance_score=round(chunk.score, 4),
            )
            for chunk in answer.sources
        ],
        model=answer.model,
        tokens_used=answer.tokens_used,
    )


def _check_limits(request: AskRequest, settings: Settings) -> None:
    """Limits that come from settings, reported in the same shape as schema validation errors."""
    errors = []
    if len(request.question) > settings.max_question_chars:
        errors.append(
            {
                "type": "string_too_long",
                "loc": ("body", "question"),
                "msg": f"String should have at most {settings.max_question_chars} characters",
            }
        )
    if request.top_k is not None and request.top_k > settings.top_k_max:
        errors.append(
            {
                "type": "less_than_equal",
                "loc": ("body", "top_k"),
                "msg": f"Input should be less than or equal to {settings.top_k_max}",
            }
        )
    if errors:
        raise RequestValidationError(errors)
