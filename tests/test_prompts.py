import logging

import pytest

from app.services.prompts import (
    REFUSAL,
    SYSTEM_PROMPT,
    build_prompt,
    drop_invalid_citations,
    is_refusal,
)
from app.services.retrieval import RetrievedChunk


def no_overhead(_text: str) -> int:
    """Isolate chunk packing: only stored token_count counts against the budget."""
    return 0


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
    assert "do not reply with only a citation" in SYSTEM_PROMPT
    assert "no other words" in SYSTEM_PROMPT


def test_user_message_numbers_sources_with_name_and_pages_then_the_question() -> None:
    chunks = [
        chunk("Revenue grew 12% in Q3.", pages=(3, 3)),
        chunk("Costs were flat.", pages=(4, 5)),
        chunk("Vacation is 25 days.", name="handbook.md"),
    ]

    prompt = build_prompt(
        "How much did revenue grow?", chunks, max_context_tokens=3000, count_tokens=no_overhead
    )

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

    prompt = build_prompt("q", chunks, max_context_tokens=1000, count_tokens=no_overhead)

    assert [c.text for c in prompt.sources] == ["first", "second"]
    assert "third" not in prompt.user


def test_budget_never_skips_ahead_to_a_smaller_lower_ranked_chunk() -> None:
    chunks = [
        chunk("first", token_count=500),
        chunk("too big", token_count=600),
        chunk("small", token_count=10),
    ]

    prompt = build_prompt("q", chunks, max_context_tokens=1000, count_tokens=no_overhead)

    assert [c.text for c in prompt.sources] == ["first"]


def test_chunk_exactly_filling_the_budget_is_included() -> None:
    prompt = build_prompt(
        "q",
        [chunk("a", token_count=600), chunk("b", token_count=400)],
        1000,
        count_tokens=no_overhead,
    )

    assert [c.text for c in prompt.sources] == ["a", "b"]


def test_building_a_prompt_without_chunks_is_an_error() -> None:
    with pytest.raises(ValueError, match="REFUSAL"):
        build_prompt("q", [], max_context_tokens=3000, count_tokens=no_overhead)


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


def test_question_and_labels_consume_budget_so_a_later_chunk_is_dropped() -> None:
    chunks = [chunk("first", token_count=10), chunk("second", token_count=10)]

    def count_tokens(text: str) -> int:
        if text == SYSTEM_PROMPT:
            return 50
        if "Question:" in text:
            return 20
        return 5  # source header / separator

    # 50 system + 20 question + 5 header + 10 chunk = 85; the second chunk would be 100.
    prompt = build_prompt("q", chunks, max_context_tokens=90, count_tokens=count_tokens)

    assert [c.text for c in prompt.sources] == ["first"]
    assert "second" not in prompt.user


def test_overhead_that_leaves_no_room_for_any_chunk_returns_no_sources() -> None:
    def count_tokens(text: str) -> int:
        return 100 if text == SYSTEM_PROMPT else 10

    prompt = build_prompt("q", [chunk("first", token_count=10)], 50, count_tokens)

    assert prompt.sources == []


def test_out_of_range_citations_are_removed(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="app.services.prompts"):
        cleaned = drop_invalid_citations("Revenue grew 12% [1] and also [7][0]. See [2].", 2)

    assert cleaned == "Revenue grew 12% [1] and also. See [2]."
    assert "7" in caplog.text and "0" in caplog.text


def test_adjacent_invalid_citation_is_dropped_without_touching_valid_ones() -> None:
    assert drop_invalid_citations("Claim [1][9][2].", 2) == "Claim [1][2]."


def test_answer_that_is_only_invalid_citations_becomes_empty() -> None:
    assert drop_invalid_citations("[7]", 3) == ""
