"""
Download SEC filings (10-Q or 10-K) for any ticker from EDGAR,
convert the HTML filing to a PDF, and save it to the data/ directory.
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
_dl = Downloader("FinancialRAGAssistant", "research@example.com")


def _sanitize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _html_to_paragraphs(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "meta", "link", "ix:header", "ix:hidden"]):
        tag.decompose()

    paragraphs: list[str] = []

    # Process tables row-by-row so label and value cells stay on the same line.
    # EDGAR financial tables put row labels in one <td> and dollar amounts in
    # adjacent <td>s — get_text() on the whole document splits them onto separate
    # lines, orphaning the numbers from their labels.
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = []
            for td in tr.find_all(["td", "th"]):
                cell_text = re.sub(r"\s+", " ", td.get_text(separator=" ")).strip()
                if cell_text:
                    cells.append(cell_text)
            if cells:
                row = " | ".join(cells)
                if len(row) > 5:
                    paragraphs.append(row)
        table.decompose()  # prevent double-counting in the prose pass below

    # Extract remaining prose (narrative, notes) line-by-line
    buffer: list[str] = []
    for line in soup.get_text(separator="\n").splitlines():
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


def _write_pdf(paragraphs: list[str], output_path: Path, title: str) -> None:
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


def download_filing(
    ticker: str,
    form_type: str = "10-Q",
    progress_cb=None,
) -> tuple[Path, dict]:
    """
    Download the latest filing for a ticker from SEC EDGAR.

    Returns (pdf_path, metadata) where metadata contains:
        ticker, company_name, form_type, filing_date, report_date
    """
    DATA_DIR.mkdir(exist_ok=True)

    def _step(msg: str):
        if progress_cb:
            progress_cb(msg)

    _step(f"Fetching {form_type} metadata for {ticker} from SEC EDGAR...")
    metadatas = _dl.get_filing_metadatas(
        RequestedFilings(ticker_or_cik=ticker, form_type=form_type, limit=1)
    )
    if not metadatas:
        raise ValueError(f"No {form_type} filing found for {ticker}")
    m = metadatas[0]

    pdf_path = DATA_DIR / f"{ticker}_{form_type.replace('-', '')}_{m.report_date}.pdf"

    if pdf_path.exists():
        _step(f"PDF already exists: {pdf_path.name}")
    else:
        _step(f"Downloading HTML filing ({m.filing_date})...")
        resp = requests.get(m.primary_doc_url, headers=HEADERS, timeout=60)
        resp.raise_for_status()

        _step("Extracting text from filing...")
        paragraphs = _html_to_paragraphs(resp.text)

        _step(f"Writing PDF ({len(paragraphs)} text blocks)...")
        title = (
            f"{m.company_name} — {m.form_type} "
            f"(Period: {m.report_date}, Filed: {m.filing_date})"
        )
        _write_pdf(paragraphs, pdf_path, title)

    metadata = {
        "ticker": ticker,
        "company_name": m.company_name,
        "form_type": m.form_type,
        "filing_date": m.filing_date,
        "report_date": m.report_date,
        "pdf_path": str(pdf_path),
    }
    return pdf_path, metadata
