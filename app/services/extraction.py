import codecs
from dataclasses import dataclass
from pathlib import Path

from pypdf import PageObject, PasswordType, PdfReader

from app.db.models import FileType
from app.errors import (
    EmptyDocumentError,
    TooManyPagesError,
    UnreadableDocumentError,
    UnsupportedFileTypeError,
)

SUPPORTED_EXTENSIONS: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".txt": FileType.TXT,
    ".md": FileType.MD,
}

_NO_TEXT_MESSAGE = (
    "The document contains no extractable text. Scanned (image-only) PDFs are not supported."
)
_BINARY_SNIFF_BYTES = 8192
_UTF16_BOMS = (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)


@dataclass(frozen=True)
class PageText:
    page_number: int | None
    text: str


@dataclass(frozen=True)
class ValidationResult:
    file_type: FileType
    page_count: int | None


def detect_file_type(filename: str) -> FileType:
    extension = Path(filename).suffix.lower()
    file_type = SUPPORTED_EXTENSIONS.get(extension)
    if file_type is None:
        supported = list(SUPPORTED_EXTENSIONS)
        shown = f"'{extension}'" if extension else "(no extension)"
        raise UnsupportedFileTypeError(
            f"Unsupported file type {shown}. Supported types: {', '.join(supported)}",
            {"supported_types": supported},
        )
    return file_type


def validate(path: Path, file_type: FileType, max_pages: int) -> ValidationResult:
    """Cheap checks for the upload request; full extraction happens later in the background."""
    if path.stat().st_size == 0:
        raise EmptyDocumentError("The file is empty.")

    if file_type is not FileType.PDF:
        if not _read_text_file(path).strip():
            raise EmptyDocumentError("The document contains only whitespace.")
        return ValidationResult(file_type, None)

    reader = _open_pdf(path)
    page_count = len(reader.pages)
    if page_count == 0:
        raise EmptyDocumentError("The PDF has no pages.")
    if page_count > max_pages:
        raise TooManyPagesError(
            f"PDF has {page_count} pages; the maximum is {max_pages}.",
            {"pages": page_count, "max_pages": max_pages},
        )
    # any() stops at the first page with text, so a normal PDF costs one page of extraction.
    has_text = any(
        _page_text(page, number).strip() for number, page in enumerate(reader.pages, start=1)
    )
    if not has_text:
        raise EmptyDocumentError(_NO_TEXT_MESSAGE)
    return ValidationResult(file_type, page_count)


def extract(path: Path, file_type: FileType) -> list[PageText]:
    """Full text extraction. PDF pages without text are skipped but keep their page numbers."""
    if file_type is FileType.PDF:
        reader = _open_pdf(path)
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = _page_text(page, number)
            if text.strip():
                pages.append(PageText(number, text))
    else:
        text = _read_text_file(path)
        pages = [PageText(None, text)] if text.strip() else []

    if not pages:
        raise EmptyDocumentError(_NO_TEXT_MESSAGE)
    return pages


def _open_pdf(path: Path) -> PdfReader:
    # pypdf raises many different exception types on malformed input; any failure to
    # parse an untrusted upload is reported as an unreadable document, not a server error.
    try:
        reader = PdfReader(path)
        if reader.is_encrypted and reader.decrypt("") == PasswordType.NOT_DECRYPTED:
            raise UnreadableDocumentError("The PDF is password-protected.")
        len(reader.pages)
    except UnreadableDocumentError:
        raise
    except Exception as exc:
        raise UnreadableDocumentError("The file could not be read as a PDF.") from exc
    return reader


def _page_text(page: PageObject, page_number: int) -> str:
    try:
        text = page.extract_text()
    except Exception as exc:
        raise UnreadableDocumentError(f"Could not extract text from page {page_number}.") from exc
    return _normalize(text or "")


def _read_text_file(path: Path) -> str:
    return _normalize(_decode_text(path.read_bytes()))


def _decode_text(raw: bytes) -> str:
    # UTF-16 is checked first because UTF-16 text legitimately contains NUL bytes.
    if raw.startswith(_UTF16_BOMS):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError as exc:
            raise UnreadableDocumentError("The file is not valid UTF-16 text.") from exc
    if b"\x00" in raw[:_BINARY_SNIFF_BYTES]:
        raise UnreadableDocumentError("The file does not look like a text file.")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
