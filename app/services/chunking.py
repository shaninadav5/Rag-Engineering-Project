"""Structure-aware chunking: paragraphs, then sentences, then words, packed across pages.

See specs/001-rag-document-qa/research.md section 1 for the reasoning behind each rule.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from app.services.extraction import PageText

TokenCounter = Callable[[str], int]

_BLANK_LINE = re.compile(r"\n\s*\n")
_MARKDOWN_HEADING = re.compile(r"#{1,6}\s")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_INLINE_WHITESPACE = re.compile(r"[ \t]+")
_SENTENCE_END = (".", "!", "?", ":", ";")
_CLOSING_MARKS = "\"')]"


@dataclass(frozen=True)
class ChunkData:
    chunk_index: int
    text: str
    page_start: int | None
    page_end: int | None
    token_count: int


@dataclass(frozen=True)
class _Unit:
    """Smallest piece that is packed into chunks: a paragraph, sentence, or word window."""

    text: str
    page: int | None
    paragraph_id: int


def chunk_pages(
    pages: list[PageText], count_tokens: TokenCounter, chunk_size: int, chunk_overlap: int
) -> list[ChunkData]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    units = _build_units(pages, count_tokens, chunk_size)
    groups = _pack(units, count_tokens, chunk_size, chunk_overlap)
    return [_to_chunk(index, group, count_tokens) for index, group in enumerate(groups)]


# --- Step 1: split text into units that each fit in a chunk ----------------------


def _build_units(pages: list[PageText], count_tokens: TokenCounter, chunk_size: int) -> list[_Unit]:
    units: list[_Unit] = []
    paragraph_id = -1
    for page in pages:
        for position, paragraph in enumerate(_split_paragraphs(page.text)):
            continues_previous_page = (
                position == 0 and bool(units) and _continues(units[-1].text, paragraph)
            )
            if not continues_previous_page:
                paragraph_id += 1
            for piece in _fit(paragraph, count_tokens, chunk_size):
                units.append(_Unit(piece, page.page_number, paragraph_id))
    return units


def _split_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    for block in _BLANK_LINE.split(text):
        lines: list[str] = []
        for raw_line in block.splitlines():
            line = _INLINE_WHITESPACE.sub(" ", raw_line).strip()
            if not line:
                continue
            if _MARKDOWN_HEADING.match(line):
                if lines:
                    paragraphs.append("\n".join(lines))
                    lines = []
                paragraphs.append(line)
            else:
                lines.append(line)
        if lines:
            paragraphs.append("\n".join(lines))
    return paragraphs


def _continues(previous_text: str, next_paragraph: str) -> bool:
    """True when a paragraph was cut by a page break mid-sentence."""
    ends_sentence = previous_text.rstrip(_CLOSING_MARKS).endswith(_SENTENCE_END)
    return not ends_sentence and next_paragraph[:1].islower()


def _fit(paragraph: str, count_tokens: TokenCounter, chunk_size: int) -> list[str]:
    if count_tokens(paragraph) <= chunk_size:
        return [paragraph]
    pieces: list[str] = []
    for sentence in _SENTENCE_BOUNDARY.split(paragraph):
        sentence = sentence.strip()
        if not sentence:
            continue
        if count_tokens(sentence) <= chunk_size:
            pieces.append(sentence)
        else:
            pieces.extend(_word_windows(sentence, count_tokens, chunk_size))
    return pieces


def _word_windows(sentence: str, count_tokens: TokenCounter, chunk_size: int) -> list[str]:
    windows: list[str] = []
    current: list[str] = []
    for word in sentence.split():
        if count_tokens(word) > chunk_size:
            if current:
                windows.append(" ".join(current))
                current = []
            windows.extend(_halve_until_fits(word, count_tokens, chunk_size))
            continue
        if current and count_tokens(" ".join([*current, word])) > chunk_size:
            windows.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        windows.append(" ".join(current))
    return windows


def _halve_until_fits(text: str, count_tokens: TokenCounter, chunk_size: int) -> list[str]:
    """Last resort for a single 'word' too long for a chunk (e.g. a long URL or encoded blob)."""
    if count_tokens(text) <= chunk_size or len(text) <= 1:
        return [text]
    middle = len(text) // 2
    return _halve_until_fits(text[:middle], count_tokens, chunk_size) + _halve_until_fits(
        text[middle:], count_tokens, chunk_size
    )


# --- Step 2: pack units into chunks with overlap --------------------------------


def _pack(
    units: list[_Unit], count_tokens: TokenCounter, chunk_size: int, chunk_overlap: int
) -> list[list[_Unit]]:
    groups: list[list[_Unit]] = []
    current: list[_Unit] = []
    for unit in units:
        if current and count_tokens(_join([*current, unit])) > chunk_size:
            groups.append(current)
            current = _overlap_tail(current, count_tokens, chunk_overlap)
            while current and count_tokens(_join([*current, unit])) > chunk_size:
                current = current[1:]
        current.append(unit)
    if current:
        groups.append(current)
    return groups


def _overlap_tail(
    group: list[_Unit], count_tokens: TokenCounter, chunk_overlap: int
) -> list[_Unit]:
    """Trailing whole units within the overlap budget, or trailing words of the last unit."""
    if chunk_overlap == 0:
        return []

    tail: list[_Unit] = []
    for unit in reversed(group):
        if count_tokens(_join([unit, *tail])) > chunk_overlap:
            break
        tail.insert(0, unit)
    if tail:
        return tail

    last = group[-1]
    words: list[str] = []
    for word in reversed(last.text.split()):
        if count_tokens(" ".join([word, *words])) > chunk_overlap:
            break
        words.insert(0, word)
    return [_Unit(" ".join(words), last.page, last.paragraph_id)] if words else []


def _join(units: list[_Unit]) -> str:
    parts = [units[0].text]
    for previous, unit in zip(units, units[1:], strict=False):
        parts.append(" " if unit.paragraph_id == previous.paragraph_id else "\n\n")
        parts.append(unit.text)
    return "".join(parts)


def _to_chunk(index: int, group: list[_Unit], count_tokens: TokenCounter) -> ChunkData:
    text = _join(group)
    pages = [unit.page for unit in group if unit.page is not None]
    return ChunkData(
        chunk_index=index,
        text=text,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        token_count=count_tokens(text),
    )
