"""
Download Apple's latest 10-Q from SEC EDGAR and save as PDF to data/.
Uses sec-downloader for metadata, requests for the HTML, bs4 for text
extraction, and fpdf2 to produce a PDF suitable for RAG ingestion.
"""
import re
import unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from fpdf import FPDF
from sec_downloader import Downloader
from sec_downloader.types import RequestedFilings

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HEADERS = {"User-Agent": "FinancialRAGAssistant research@example.com"}


def _sanitize(text: str) -> str:
    """Replace non-latin-1 characters so fpdf2's core fonts can encode them."""
    text = unicodedata.normalize("NFKD", text)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def fetch_filing_html(primary_doc_url: str) -> str:
    r = requests.get(primary_doc_url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.text


def html_to_paragraphs(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements
    for tag in soup(["script", "style", "meta", "link", "ix:header", "ix:hidden"]):
        tag.decompose()

    # Walk the full document line by line, grouping by block-level breaks.
    # This captures XBRL-annotated text (ix:nonfraction, ix:nonnumeric, spans)
    # that tag-specific selectors miss.
    full_text = soup.get_text(separator="\n")
    raw_lines = full_text.splitlines()

    paragraphs = []
    buffer: list[str] = []
    for line in raw_lines:
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            if buffer:
                combined = " ".join(buffer).strip()
                if len(combined) > 15:
                    paragraphs.append(combined)
                buffer = []
        else:
            buffer.append(line)

    if buffer:
        combined = " ".join(buffer).strip()
        if len(combined) > 15:
            paragraphs.append(combined)

    return paragraphs


def write_pdf(paragraphs: list[str], output_path: Path, title: str) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 8, _sanitize(title))
    pdf.ln(4)
    pdf.set_font("Helvetica", size=9)

    for para in paragraphs:
        pdf.multi_cell(0, 5, _sanitize(para))
        pdf.ln(1)

    pdf.output(str(output_path))


def download_apple_10q():
    print("Fetching Apple 10-Q metadata from SEC EDGAR...")
    dl = Downloader("FinancialRAGAssistant", "research@example.com")
    metadatas = dl.get_filing_metadatas(
        RequestedFilings(ticker_or_cik="AAPL", form_type="10-Q", limit=1)
    )
    m = metadatas[0]
    print(f"  {m.company_name} | {m.form_type} | filed {m.filing_date} | period {m.report_date}")

    print("Downloading HTML filing...")
    html = fetch_filing_html(m.primary_doc_url)

    print("Extracting text...")
    paragraphs = html_to_paragraphs(html)
    print(f"  Extracted {len(paragraphs)} text blocks.")

    filename = f"AAPL_10Q_{m.report_date}.pdf"
    output_path = DATA_DIR / filename

    print(f"Writing PDF to {output_path}...")
    title = f"{m.company_name} — {m.form_type} (Period: {m.report_date}, Filed: {m.filing_date})"
    write_pdf(paragraphs, output_path, title)

    size_kb = output_path.stat().st_size // 1024
    print(f"Done. Saved {filename} ({size_kb} KB)")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    download_apple_10q()
