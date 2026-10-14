"""Build the Stage 11 PDFs with planted facts. Output is gitignored under eval/."""

from pathlib import Path

from fpdf import FPDF

OUT = Path("eval")

FILLER = (
    "This section describes ordinary operating context and does not state a numeric result, "
    "a product name, an office opening, or a leave entitlement. Teams reviewed process notes, "
    "meeting cadence, and document formatting. No decision in this paragraph changes a figure "
    "reported elsewhere in the file. "
)


def page(fact: str | None) -> str:
    body = (FILLER * 8).strip()
    if fact is None:
        return body
    return f"{body}\n\n{fact}\n\n{body}"


def write(path: Path, pages: list[str]) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=11)
    for text in pages:
        pdf.add_page()
        pdf.multi_cell(0, 6, text)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    print(path, "pages", len(pages))


def main() -> None:
    write(
        OUT / "acme-q3-report.pdf",
        [
            page("Acme Corp Q3 2025 operating review. This page is an introduction only."),
            page("Q3 revenue was 48 million dollars, up 12 percent from Q2."),
            page(None),
            page("The Berlin office opened in March 2024."),
            page(None),
            page(None),
            page("The flagship product is called Northstar."),
            page(None),
            page("Employee headcount at quarter end was 640."),
            page("End of the Q3 2025 operating review."),
        ],
    )
    write(
        OUT / "acme-handbook.pdf",
        [
            page("Full-time staff receive 25 days of annual leave."),
            page(None),
            page("Parental leave is 16 weeks."),
            page(None),
            page("The office cafeteria closes at 3 pm."),
            page("End of the employee handbook."),
        ],
    )


if __name__ == "__main__":
    main()
