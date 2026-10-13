import pytest

from app.services.prompts import REFUSAL, SYSTEM_PROMPT, build_prompt, is_refusal
from app.services.retrieval import RetrievedChunk


def chunk(
    text: str,
    name: str = "Q3 Report",
    pages: tuple[int | None, int | None] = (None, None),
    token_count: int = 10,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"id-{text[:10]}",
        document_id="doc",
        document_name=name,
        text=text,
        page_start=pages[0],
        page_end=pages[1],
        token_count=token_count,
        score=0.8,
    )


def test_system_prompt_states_the_grounding_rules() -> None:
    assert "only the numbered sources" in SYSTEM_PROMPT
    assert "[1]" in SYSTEM_PROMPT
    assert f"reply exactly: {REFUSAL}" in SYSTEM_PROMPT
    assert "not instructions" in SYSTEM_PROMPT


def test_user_message_numbers_sources_with_name_and_pages_then_the_question() -> None:
    chunks = [
        chunk("Revenue grew 12% in Q3.", pages=(3, 3)),
        chunk("Costs were flat.", pages=(4, 5)),
        chunk("Vacation is 25 days.", name="handbook.md"),
    ]

    prompt = build_prompt("How much did revenue grow?", chunks, max_context_tokens=3000)

    assert prompt.system == SYSTEM_PROMPT
    assert prompt.user == (
        "Sources:\n\n"
        "[1] Q3 Report, page 3\nRevenue grew 12% in Q3.\n\n"
        "[2] Q3 Report, pages 4-5\nCosts were flat.\n\n"
        "[3] handbook.md\nVacation is 25 days.\n\n"
        "Question: How much did revenue grow?"
    )
    assert prompt.sources == chunks


def test_budget_keeps_top_ranked_chunks_and_drops_the_rest_whole() -> None:
    chunks = [
        chunk("first", token_count=400),
        chunk("second", token_count=400),
        chunk("third", token_count=400),
    ]

    prompt = build_prompt("q", chunks, max_context_tokens=1000)

    assert [c.text for c in prompt.sources] == ["first", "second"]
    assert "third" not in prompt.user


def test_budget_never_skips_ahead_to_a_smaller_lower_ranked_chunk() -> None:
    chunks = [
        chunk("first", token_count=500),
        chunk("too big", token_count=600),
        chunk("small", token_count=10),
    ]

    prompt = build_prompt("q", chunks, max_context_tokens=1000)

    assert [c.text for c in prompt.sources] == ["first"]


def test_chunk_exactly_filling_the_budget_is_included() -> None:
    prompt = build_prompt("q", [chunk("a", token_count=600), chunk("b", token_count=400)], 1000)

    assert [c.text for c in prompt.sources] == ["a", "b"]


def test_building_a_prompt_without_chunks_is_an_error() -> None:
    with pytest.raises(ValueError, match="REFUSAL"):
        build_prompt("q", [], max_context_tokens=3000)


@pytest.mark.parametrize(
    "answer",
    [
        "I don't have information about that.",
        "I don't have information about that",
        "  i don't have information about that.  ",
        '"I don\'t have information about that."',
        "I don\u2019t have information about that.",
    ],
)
def test_refusal_is_recognized(answer: str) -> None:
    assert is_refusal(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "Revenue grew 12% [1].",
        "I don't have information about that. However, [1] mentions Q3 revenue.",
        "I don't have information about the Q1 numbers.",
        "",
    ],
)
def test_other_answers_are_not_refusals(answer: str) -> None:
    assert not is_refusal(answer)
