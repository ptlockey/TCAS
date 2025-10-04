"""Generate a PDF version of USER_GUIDE.md using ReportLab.

This script parses the Markdown guide and renders headings, paragraphs,
and fenced code blocks with appropriate styling in the output PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from reportlab.lib import pagesizes
from reportlab.lib.styles import ParagraphStyle, StyleSheet1, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate


PROJECT_ROOT = Path(__file__).parent
MARKDOWN_PATH = PROJECT_ROOT / "USER_GUIDE.md"
OUTPUT_PDF_PATH = PROJECT_ROOT / "USER_GUIDE.pdf"


def load_styles() -> StyleSheet1:
    """Return a stylesheet configured for headings, body text, and code blocks."""

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="CustomHeading1",
            parent=styles["Heading1"],
            fontSize=20,
            leading=24,
            spaceAfter=12,
            spaceBefore=18,
            alignment=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CustomHeading2",
            parent=styles["Heading2"],
            fontSize=16,
            leading=20,
            spaceAfter=10,
            spaceBefore=16,
            alignment=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CustomHeading3",
            parent=styles["Heading3"],
            fontSize=14,
            leading=18,
            spaceAfter=8,
            spaceBefore=14,
            alignment=0,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body", parent=styles["BodyText"], spaceAfter=8, leading=15
        )
    )
    styles.add(
        ParagraphStyle(
            name="Bullet",
            parent=styles["BodyText"],
            leftIndent=18,
            bulletIndent=10,
            spaceAfter=4,
            leading=15,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Code",
            fontName="Courier",
            fontSize=10,
            leading=12,
            leftIndent=12,
            rightIndent=12,
            spaceBefore=6,
            spaceAfter=12,
        )
    )

    return styles


def register_fonts() -> None:
    """Register fonts needed for rendering."""

    # Ensure Courier is available for the code blocks.
    try:
        pdfmetrics.registerFont(TTFont("Courier", "Courier.ttf"))
    except Exception:
        # Courier is a built-in PDF font; registration is optional if unavailable.
        pass


def parse_markdown(md_text: str, styles: StyleSheet1) -> List:
    """Convert markdown text to a list of ReportLab flowables."""

    flowables: List = []
    paragraphs: List[str] = []
    in_code_block = False
    code_lines: List[str] = []

    def flush_paragraphs() -> None:
        nonlocal paragraphs
        if paragraphs:
            text = " ".join(paragraphs).strip()
            if text:
                flowables.append(Paragraph(text, styles["Body"]))
            paragraphs = []

    def flush_code_block() -> None:
        nonlocal code_lines
        if code_lines:
            code_text = "\n".join(code_lines).rstrip()
            flowables.append(Preformatted(code_text, styles["Code"]))
            code_lines = []

    for raw_line in md_text.splitlines():
        line = raw_line.rstrip()

        if line.startswith("```"):
            if in_code_block:
                flush_code_block()
                in_code_block = False
            else:
                flush_paragraphs()
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        if not line:
            flush_paragraphs()
            continue

        if line.startswith("#"):
            flush_paragraphs()
            level = min(len(line) - len(line.lstrip("#")), 3)
            heading_text = line.lstrip("# ").strip()
            style_name = {1: "CustomHeading1", 2: "CustomHeading2", 3: "CustomHeading3"}[level]
            flowables.append(Paragraph(f"<b>{heading_text}</b>", styles[style_name]))
            continue

        stripped = line.lstrip()
        if stripped.startswith(("- ", "* ")):
            flush_paragraphs()
            bullet_text = stripped[2:].strip()
            flowables.append(
                Paragraph(
                    f"<bullet>&bull;</bullet> {bullet_text}", styles["Bullet"]
                )
            )
            continue

        paragraphs.append(line)

    flush_paragraphs()
    if in_code_block:
        flush_code_block()

    return flowables


def build_pdf(flowables: Iterable) -> None:
    """Create the PDF document from the flowables."""

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF_PATH),
        pagesize=pagesizes.LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    doc.build(list(flowables))


def main() -> None:
    register_fonts()
    styles = load_styles()
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    flowables = parse_markdown(markdown, styles)
    build_pdf(flowables)


if __name__ == "__main__":
    main()
