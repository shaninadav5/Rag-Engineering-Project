from pathlib import Path

from fpdf import FPDF
from pypdf import PdfWriter


def write_pdf(path: Path, pages: list[str]) -> Path:
    """Write a PDF with one page per string; an empty string gives a page with no text."""
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for text in pages:
        pdf.add_page()
        if text:
            pdf.multi_cell(0, 10, text)
    pdf.output(str(path))
    return path


def write_encrypted_pdf(source: Path, target: Path, user_password: str) -> Path:
    """Encrypt a PDF. An empty user password means it opens without one (permissions only)."""
    writer = PdfWriter(clone_from=source)
    writer.encrypt(user_password=user_password, owner_password="owner-secret", algorithm="RC4-128")
    writer.write(target)
    return target


def write_pdf_without_pages(path: Path) -> Path:
    writer = PdfWriter()
    writer.write(path)
    return path
