from pathlib import Path
from typing import Any

import pytest
from pypdf import PageObject

from app.db.models import FileType
from app.errors import (
    EmptyDocumentError,
    TooManyPagesError,
    UnreadableDocumentError,
    UnsupportedFileTypeError,
)
from app.services.extraction import PageText, detect_file_type, extract, validate
from tests.pdf_factory import write_encrypted_pdf, write_pdf, write_pdf_without_pages

MAX_PAGES = 1000


@pytest.fixture
def count_page_extractions(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    original = PageObject.extract_text

    def counting_extract_text(self: PageObject, *args: Any, **kwargs: Any) -> str:
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PageObject, "extract_text", counting_extract_text)
    return calls


# --- File type detection -----------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("report.pdf", FileType.PDF),
        ("REPORT.PDF", FileType.PDF),
        ("notes.txt", FileType.TXT),
        ("README.md", FileType.MD),
    ],
)
def test_detect_supported_file_types(filename: str, expected: FileType) -> None:
    assert detect_file_type(filename) == expected


@pytest.mark.parametrize("filename", ["notes.docx", "archive.tar.gz", "Makefile"])
def test_detect_rejects_unsupported_types_with_supported_list(filename: str) -> None:
    with pytest.raises(UnsupportedFileTypeError) as excinfo:
        detect_file_type(filename)

    assert excinfo.value.details == {"supported_types": [".pdf", ".txt", ".md"]}
    assert ".pdf, .txt, .md" in excinfo.value.message


# --- PDF validation ----------------------------------------------------------


def test_validate_pdf_returns_page_count(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "doc.pdf", ["One.", "Two.", "Three."])

    result = validate(pdf, FileType.PDF, MAX_PAGES)

    assert result.file_type == FileType.PDF
    assert result.page_count == 3


def test_validate_stops_at_first_page_with_text(
    tmp_path: Path, count_page_extractions: list[int]
) -> None:
    pdf = write_pdf(tmp_path / "doc.pdf", ["Text on every page."] * 5)

    validate(pdf, FileType.PDF, MAX_PAGES)

    assert len(count_page_extractions) == 1


def test_validate_accepts_pdf_whose_text_starts_on_a_later_page(
    tmp_path: Path, count_page_extractions: list[int]
) -> None:
    pdf = write_pdf(tmp_path / "doc.pdf", ["", "", "Text only on page three."])

    result = validate(pdf, FileType.PDF, MAX_PAGES)

    assert result.page_count == 3
    assert len(count_page_extractions) == 3


def test_validate_rejects_pdf_without_text(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "scanned.pdf", ["", ""])

    with pytest.raises(EmptyDocumentError, match="Scanned"):
        validate(pdf, FileType.PDF, MAX_PAGES)


def test_validate_rejects_pdf_without_pages(tmp_path: Path) -> None:
    pdf = write_pdf_without_pages(tmp_path / "no-pages.pdf")

    with pytest.raises(EmptyDocumentError):
        validate(pdf, FileType.PDF, MAX_PAGES)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(b"this is not a pdf", id="not-a-pdf"),
        pytest.param(b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog", id="truncated-pdf"),
    ],
)
def test_validate_rejects_corrupt_pdf(tmp_path: Path, content: bytes) -> None:
    pdf = tmp_path / "corrupt.pdf"
    pdf.write_bytes(content)

    with pytest.raises(UnreadableDocumentError):
        validate(pdf, FileType.PDF, MAX_PAGES)


def test_validate_rejects_password_protected_pdf(tmp_path: Path) -> None:
    plain = write_pdf(tmp_path / "plain.pdf", ["Secret text."])
    locked = write_encrypted_pdf(plain, tmp_path / "locked.pdf", user_password="secret")

    with pytest.raises(UnreadableDocumentError, match="password"):
        validate(locked, FileType.PDF, MAX_PAGES)


def test_validate_accepts_pdf_encrypted_only_for_permissions(tmp_path: Path) -> None:
    plain = write_pdf(tmp_path / "plain.pdf", ["Readable text."])
    restricted = write_encrypted_pdf(plain, tmp_path / "restricted.pdf", user_password="")

    assert validate(restricted, FileType.PDF, MAX_PAGES).page_count == 1
    assert "Readable text." in extract(restricted, FileType.PDF)[0].text


def test_validate_rejects_pdf_over_page_limit(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "long.pdf", ["One.", "Two.", "Three."])

    with pytest.raises(TooManyPagesError) as excinfo:
        validate(pdf, FileType.PDF, max_pages=2)

    assert excinfo.value.details == {"pages": 3, "max_pages": 2}


# --- Empty input -------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "file_type"),
    [("empty.pdf", FileType.PDF), ("empty.txt", FileType.TXT), ("empty.md", FileType.MD)],
)
def test_validate_rejects_zero_byte_file(
    tmp_path: Path, filename: str, file_type: FileType
) -> None:
    path = tmp_path / filename
    path.write_bytes(b"")

    with pytest.raises(EmptyDocumentError, match="empty"):
        validate(path, file_type, MAX_PAGES)


def test_validate_rejects_whitespace_only_text(tmp_path: Path) -> None:
    path = tmp_path / "blank.txt"
    path.write_bytes(b"   \n\n\t  \r\n")

    with pytest.raises(EmptyDocumentError):
        validate(path, FileType.TXT, MAX_PAGES)


# --- PDF extraction ----------------------------------------------------------


def test_extract_pdf_keeps_true_page_numbers_when_skipping_blank_pages(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "doc.pdf", ["First page text.", "", "Third page text."])

    pages = extract(pdf, FileType.PDF)

    assert [page.page_number for page in pages] == [1, 3]
    assert "First page text." in pages[0].text
    assert "Third page text." in pages[1].text


def test_extract_pdf_text_continuing_across_page_break(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "doc.pdf", ["Revenue grew strongly because", "of new customers."])

    pages = extract(pdf, FileType.PDF)

    assert [page.page_number for page in pages] == [1, 2]
    assert pages[0].text.strip().endswith("because")
    assert pages[1].text.strip().startswith("of new customers.")


def test_extract_rejects_pdf_without_text(tmp_path: Path) -> None:
    pdf = write_pdf(tmp_path / "scanned.pdf", [""])

    with pytest.raises(EmptyDocumentError):
        extract(pdf, FileType.PDF)


# --- Text decoding -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(b"Hello world", "Hello world", id="utf-8"),
        pytest.param("caf\u00e9".encode("utf-8-sig"), "caf\u00e9", id="utf-8-bom"),
        pytest.param("caf\u00e9".encode("cp1252"), "caf\u00e9", id="windows-1252"),
        pytest.param("caf\u00e9".encode("utf-16"), "caf\u00e9", id="utf-16-bom"),
        pytest.param(
            b"line one\r\nline two\rline three", "line one\nline two\nline three", id="crlf"
        ),
    ],
)
def test_text_files_are_decoded(tmp_path: Path, raw: bytes, expected: str) -> None:
    path = tmp_path / "doc.txt"
    path.write_bytes(raw)

    assert validate(path, FileType.TXT, MAX_PAGES).page_count is None
    assert extract(path, FileType.TXT) == [PageText(None, expected)]


def test_markdown_is_kept_as_written(tmp_path: Path) -> None:
    content = "# Title\n\nBody text.\n\n## Section\n\n- item"
    path = tmp_path / "doc.md"
    path.write_text(content, encoding="utf-8")

    assert extract(path, FileType.MD) == [PageText(None, content)]


def test_binary_file_named_txt_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "program.txt"
    path.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00binary")

    with pytest.raises(UnreadableDocumentError, match="text file"):
        validate(path, FileType.TXT, MAX_PAGES)
