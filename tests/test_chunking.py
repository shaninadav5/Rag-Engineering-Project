import random
import string

import pytest

from app.services.chunking import ChunkData, chunk_pages
from app.services.extraction import PageText


def count_words(text: str) -> int:
    return len(text.split())


def words(prefix: str, n: int) -> str:
    return " ".join(f"{prefix}w{i}" for i in range(n))


def sentence(prefix: str, n: int) -> str:
    """n distinct words ending with a period, e.g. 's3w0 s3w1 s3w2.'"""
    return words(prefix, n) + "."


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[ChunkData]:
    return chunk_pages([PageText(None, text)], count_words, chunk_size, chunk_overlap)


def test_empty_input_gives_no_chunks() -> None:
    assert chunk_pages([], count_words, chunk_size=100, chunk_overlap=10) == []


def test_blank_pages_produce_no_chunks_and_do_not_break_neighbours() -> None:
    pages = [
        PageText(1, "Intro sentence."),
        PageText(2, "   \n\t \r\n  "),
        PageText(3, ""),
        PageText(4, "Closing sentence."),
    ]

    chunks = chunk_pages(pages, count_words, chunk_size=100, chunk_overlap=10)

    assert chunks == [ChunkData(0, "Intro sentence.\n\nClosing sentence.", 1, 4, 4)]
    assert chunk_pages(pages[1:3], count_words, chunk_size=100, chunk_overlap=10) == []


def test_small_paragraph_is_one_chunk() -> None:
    chunks = chunk_text("Revenue grew 12% in Q3.", chunk_size=100, chunk_overlap=10)

    assert chunks == [ChunkData(0, "Revenue grew 12% in Q3.", None, None, 5)]


def test_inline_whitespace_is_collapsed() -> None:
    chunks = chunk_text("Alpha    beta\t\tgamma.  ", chunk_size=100, chunk_overlap=10)

    assert chunks[0].text == "Alpha beta gamma."


def test_paragraphs_are_packed_whole_and_chunks_break_at_paragraph_boundaries() -> None:
    p1, p2, p3 = sentence("a", 39), sentence("b", 39), sentence("c", 39)

    chunks = chunk_text(f"{p1}\n\n{p2}\n\n{p3}", chunk_size=100, chunk_overlap=10)

    assert len(chunks) == 2
    assert chunks[0].text == f"{p1}\n\n{p2}"
    assert chunks[1].text.endswith(f"\n\n{p3}")


def test_overlap_falls_back_to_trailing_words_when_last_unit_is_too_big() -> None:
    p1, p2 = sentence("a", 29), sentence("b", 29)

    chunks = chunk_text(f"{p1}\n\n{p2}", chunk_size=40, chunk_overlap=5)

    assert chunks[0].text == p1
    assert chunks[1].text.split()[:5] == p1.split()[-5:]
    assert chunks[1].text.endswith(p2)


def test_overlap_prefers_whole_sentences() -> None:
    sentences = [sentence(f"s{i}", 5) for i in range(10)]

    chunks = chunk_text(" ".join(sentences), chunk_size=20, chunk_overlap=10)

    assert chunks[0].text == " ".join(sentences[0:4])
    assert chunks[1].text == " ".join(sentences[2:6])


def test_oversized_paragraph_is_split_only_at_sentence_ends() -> None:
    paragraph = " ".join(sentence(f"s{i}", 7) for i in range(20))

    chunks = chunk_text(paragraph, chunk_size=30, chunk_overlap=8)

    assert len(chunks) > 1
    assert all(chunk.text.endswith(".") for chunk in chunks)


def test_oversized_sentence_is_split_into_word_windows_keeping_every_word() -> None:
    original = [f"w{i}" for i in range(250)]

    chunks = chunk_text(" ".join(original), chunk_size=100, chunk_overlap=10)

    assert all(chunk.token_count <= 100 for chunk in chunks)
    seen = list(dict.fromkeys(word for chunk in chunks for word in chunk.text.split()))
    assert seen == original


def test_single_word_longer_than_chunk_is_split() -> None:
    blob = string.ascii_letters * 4

    chunks = chunk_pages([PageText(None, blob)], len, chunk_size=50, chunk_overlap=5)

    assert all(len(chunk.text) <= 50 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == blob


def test_size_limit_overlap_and_progress_on_mixed_document() -> None:
    rng = random.Random(42)
    paragraphs = []
    for p in range(30):
        n_sentences = rng.randint(1, 12)
        paragraphs.append(
            " ".join(sentence(f"p{p}s{s}", rng.randint(2, 40)) for s in range(n_sentences))
        )
    document_words = " ".join(paragraphs).split()

    chunks = chunk_text("\n\n".join(paragraphs), chunk_size=100, chunk_overlap=15)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.token_count == count_words(chunk.text) <= 100
    for previous, current in zip(chunks, chunks[1:], strict=False):
        previous_words, current_words = previous.text.split(), current.text.split()
        shared = next(
            n
            for n in range(min(len(previous_words), 15), -1, -1)
            if previous_words[len(previous_words) - n :] == current_words[:n]
        )
        assert 1 <= shared <= 15, "consecutive chunks must overlap"
        assert len(current_words) > shared, "every chunk must add new text"
    seen = list(dict.fromkeys(word for chunk in chunks for word in chunk.text.split()))
    assert seen == document_words


def test_sentence_cut_by_page_break_stays_together_with_page_range() -> None:
    pages = [
        PageText(1, "Intro paragraph.\n\nRevenue grew strongly because"),
        PageText(2, "of new customers.\n\nNext topic."),
    ]

    chunks = chunk_pages(pages, count_words, chunk_size=100, chunk_overlap=10)

    assert len(chunks) == 1
    assert chunks[0].text == (
        "Intro paragraph.\n\nRevenue grew strongly because of new customers.\n\nNext topic."
    )
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_page_ranges_include_pages_carried_by_overlap() -> None:
    pages = [PageText(n, sentence(f"p{n}", 29)) for n in (1, 2, 3)]

    chunks = chunk_pages(pages, count_words, chunk_size=50, chunk_overlap=10)

    assert [(c.page_start, c.page_end) for c in chunks] == [(1, 1), (1, 2), (2, 3)]


def test_text_document_chunks_have_no_pages() -> None:
    chunks = chunk_text(words("a", 300), chunk_size=100, chunk_overlap=10)

    assert all(c.page_start is None and c.page_end is None for c in chunks)


def test_markdown_headings_become_their_own_paragraphs() -> None:
    chunks = chunk_text("# Intro\nAlpha beta.\n## Details\nGamma delta.", 100, 10)

    assert chunks[0].text == "# Intro\n\nAlpha beta.\n\n## Details\n\nGamma delta."


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"), [(100, 100), (100, 150), (0, 0), (100, -1)]
)
def test_invalid_arguments_are_rejected(chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("Some text.", chunk_size, chunk_overlap)


@pytest.mark.model
def test_chunks_fit_bge_input_limit_with_real_tokenizer() -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    assert tokenizer.model_max_length == 512

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False, verbose=False))

    prose = (
        "In Q3 2026, U.S.-based revenue grew 12.5% (vs. 9.1% in Q2), driven by enterprise "
        "contracts, e.g. multi-year SaaS deals worth $1.2M+ each. Churn fell to 3.4%! "
    )
    pages = [
        PageText(1, "\n\n".join(prose * n for n in (1, 3, 8, 2))),
        PageText(2, " ".join(f"token{i}" for i in range(2000))),
        PageText(3, "Final summary: " + prose * 20),
    ]

    chunks = chunk_pages(pages, count_tokens, chunk_size=510, chunk_overlap=50)

    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.token_count == count_tokens(chunk.text) <= 510
        assert len(tokenizer(chunk.text, verbose=False)["input_ids"]) <= 512
