"""PDF loading and text extraction from local files or DOIs.

Download pipeline: cached local → Unpaywall (open access) → Sci-Hub → error.
"""

from __future__ import annotations

import json
import os
import re
import ssl
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from pypdf import PdfReader


@dataclass
class PDFPage:
    """A single page extracted from a PDF."""
    page_number: int
    text: str


@dataclass
class PDFTableImage:
    """A table region rendered as an image from a PDF page."""
    page_number: int
    table_index: int
    image_bytes: bytes


@dataclass
class VisionNeedReport:
    """Report on which pages need vision-based extraction."""
    needs_vision: bool
    opaque_table_pages: List[int]  # 1-indexed page numbers with tables but no text
    text_pages: List[int]          # 1-indexed page numbers with extractable text
    total_pages: int


_DOI_PATTERN = re.compile(r"^10\.\d{4,}/")
_DEFAULT_PDF_DIR = "/tmp/evident_pdfs"
_UNPAYWALL_EMAIL = "evident.project@gmail.com"

# Lenient SSL context for Sci-Hub mirrors
_LENIENT_SSL = ssl.create_default_context()
_LENIENT_SSL.check_hostname = False
_LENIENT_SSL.verify_mode = ssl.CERT_NONE


def _resolve_source(source: str, pdf_dir: str = _DEFAULT_PDF_DIR) -> str:
    """Resolve a source (DOI or path) to a local file path."""
    if _DOI_PATTERN.match(source):
        path = _download_pdf_from_doi(source, dest_dir=pdf_dir)
    else:
        path = source
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF not found: {path}")
    return path


def load_pdf_pages(source: str, pdf_dir: str = _DEFAULT_PDF_DIR) -> list[PDFPage]:
    """Load pages from a PDF file (local path) or DOI string.

    For DOIs, tries: local cache → Unpaywall → Sci-Hub.

    Args:
        source: Local file path or a DOI (e.g. "10.7326/M19-3602").
        pdf_dir: Directory for caching downloaded PDFs.

    Returns:
        List of PDFPage with page number and extracted text.
    """
    path = _resolve_source(source, pdf_dir)
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(PDFPage(page_number=i + 1, text=text))
    return pages


def detect_opaque_tables(
    source: str,
    pdf_dir: str = _DEFAULT_PDF_DIR,
    min_cell_fill: float = 0.1,
) -> VisionNeedReport:
    """Detect PDF pages with table structures but no extractable cell text.

    Uses pdfplumber to find tables and checks if the table cells contain
    extractable text. Pages with tables where <min_cell_fill of cells have
    text are considered "opaque" and need vision-based extraction.

    Args:
        source: Local file path or DOI.
        pdf_dir: Directory for caching downloaded PDFs.
        min_cell_fill: Minimum fraction of non-empty cells for a table to be
            considered text-extractable (default 0.1 = 10%).

    Returns:
        VisionNeedReport indicating which pages need vision extraction.
    """
    import pdfplumber

    path = _resolve_source(source, pdf_dir)
    reader = PdfReader(path)
    pdf = pdfplumber.open(path)

    opaque_table_pages = []
    text_pages = []
    total_pages = len(reader.pages)

    for i in range(total_pages):
        page = pdf.pages[i]
        tables = page.find_tables()

        if not tables:
            # No tables — classify by text presence
            text = reader.pages[i].extract_text() or ""
            if text.strip():
                text_pages.append(i + 1)
            continue

        # Check if table cells have extractable text
        has_opaque_table = False
        for table in tables:
            extracted = table.extract()
            total_cells = sum(len(row) for row in extracted)
            if total_cells == 0:
                has_opaque_table = True
                break
            non_empty = sum(
                1 for row in extracted for cell in row
                if cell and cell.strip()
            )
            if non_empty / total_cells < min_cell_fill:
                has_opaque_table = True
                break

        if has_opaque_table:
            opaque_table_pages.append(i + 1)
        else:
            text_pages.append(i + 1)

    pdf.close()

    return VisionNeedReport(
        needs_vision=len(opaque_table_pages) > 0,
        opaque_table_pages=opaque_table_pages,
        text_pages=text_pages,
        total_pages=total_pages,
    )


def _doi_to_filename(doi: str) -> str:
    return doi.replace("/", "_").replace(".", "_") + ".pdf"


def _download_pdf_from_doi(doi: str, dest_dir: str = _DEFAULT_PDF_DIR) -> str:
    """Download a PDF for a DOI, trying multiple sources.

    Order: local cache → Unpaywall (open access) → Sci-Hub.
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, _doi_to_filename(doi))

    if os.path.isfile(dest_path):
        return dest_path

    # 1. Try Unpaywall (open access)
    pdf_url = _unpaywall_pdf_url(doi)
    if pdf_url:
        try:
            _download_url(pdf_url, dest_path)
            if _is_valid_pdf(dest_path):
                return dest_path
            os.remove(dest_path)
        except Exception:
            if os.path.isfile(dest_path):
                os.remove(dest_path)

    # 2. Try Sci-Hub
    try:
        _download_from_scihub(doi, dest_path)
        if _is_valid_pdf(dest_path):
            return dest_path
        os.remove(dest_path)
    except Exception:
        if os.path.isfile(dest_path):
            os.remove(dest_path)

    raise RuntimeError(
        f"Could not download PDF for DOI {doi}. "
        f"Please download manually and place at: {dest_path}"
    )


def _unpaywall_pdf_url(doi: str) -> str | None:
    """Query Unpaywall for an open-access PDF URL."""
    url = f"https://api.unpaywall.org/v2/{doi}?email={_UNPAYWALL_EMAIL}"
    req = Request(url, headers={"User-Agent": "evident/1.0"})
    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            oa = data.get("best_oa_location")
            if oa:
                return oa.get("url_for_pdf") or None
    except Exception:
        pass
    return None


def _download_from_scihub(doi: str, dest_path: str) -> None:
    """Download a PDF from Sci-Hub."""
    scihub_url = f"https://sci-hub.ru/{doi}"
    req = Request(scihub_url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0"
    })

    try:
        with urlopen(req, timeout=30, context=_LENIENT_SSL) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"Sci-Hub request failed: {e}") from e

    # Find the PDF embed URL in the page
    # Sci-Hub embeds PDFs in <iframe> or <embed> tags
    pdf_url = _extract_scihub_pdf_url(html)
    if not pdf_url:
        raise RuntimeError("Could not find PDF URL in Sci-Hub page")

    _download_url(pdf_url, dest_path, ssl_context=_LENIENT_SSL)


def _extract_scihub_pdf_url(html: str) -> str | None:
    """Extract PDF URL from Sci-Hub HTML page."""
    patterns = [
        # Direct PDF URL in storage path (current sci-hub.ru format)
        re.compile(r'(https?://sci-hub\.[a-z]+/storage/[^"\'>\s]+\.pdf)', re.IGNORECASE),
        # iframe or embed src
        re.compile(r'<iframe[^>]+src="([^"]+\.pdf[^"]*)"', re.IGNORECASE),
        re.compile(r'<embed[^>]+src="([^"]+\.pdf[^"]*)"', re.IGNORECASE),
        # Protocol-relative URLs
        re.compile(r'<iframe[^>]+src="(//[^"]+)"', re.IGNORECASE),
        re.compile(r'<embed[^>]+src="(//[^"]+)"', re.IGNORECASE),
        # onclick location.href
        re.compile(r"location\.href='([^']+\.pdf[^']*)'", re.IGNORECASE),
        # Any quoted URL ending in .pdf
        re.compile(r'["\']([^"\']*?/storage/[^"\']+\.pdf)', re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(html)
        if match:
            url = match.group(1)
            if url.startswith("//"):
                url = "https:" + url
            elif url.startswith("/"):
                url = "https://sci-hub.ru" + url
            return url
    return None


def _download_url(url: str, dest_path: str, ssl_context: ssl.SSLContext | None = None) -> None:
    """Download a URL to a local file."""
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    })
    kwargs = {"timeout": 60}
    if ssl_context:
        kwargs["context"] = ssl_context

    with urlopen(req, **kwargs) as resp:
        with open(dest_path, "wb") as f:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                f.write(chunk)


def _is_valid_pdf(path: str) -> bool:
    """Check if a file is a valid PDF by reading its header."""
    try:
        with open(path, "rb") as f:
            header = f.read(5)
            return header == b"%PDF-"
    except Exception:
        return False


def load_pdf_table_images(
    source: str,
    pdf_dir: str = _DEFAULT_PDF_DIR,
    resolution: int = 300,
    page_numbers: Optional[List[int]] = None,
) -> list[PDFTableImage]:
    """Render full pages that contain tables as images using pdfplumber.

    Detects pages with table structures and renders the full page as an image.
    Full-page rendering avoids issues with table bounding boxes that miss columns.

    Args:
        source: Local file path or a DOI string.
        pdf_dir: Directory for caching downloaded PDFs.
        resolution: DPI for rendering page images.
        page_numbers: If provided, render only these 1-indexed pages (skip table
            detection). Useful when detection was already done upstream.

    Returns:
        List of PDFTableImage, one per page that contains a table.
    """
    import io
    import pdfplumber

    path = _resolve_source(source, pdf_dir)

    table_images = []
    pdf = pdfplumber.open(path)

    if page_numbers is not None:
        # Render specific pages (already identified as needing vision)
        for page_num in page_numbers:
            idx = page_num - 1
            if 0 <= idx < len(pdf.pages):
                img = pdf.pages[idx].to_image(resolution=resolution)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                table_images.append(PDFTableImage(
                    page_number=page_num,
                    table_index=0,
                    image_bytes=buf.getvalue(),
                ))
    else:
        # Scan all pages for tables
        seen_pages = set()
        for i, page in enumerate(pdf.pages):
            if i in seen_pages:
                continue
            tables = page.find_tables()
            if tables:
                seen_pages.add(i)
                img = page.to_image(resolution=resolution)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                table_images.append(PDFTableImage(
                    page_number=i + 1,
                    table_index=0,
                    image_bytes=buf.getvalue(),
                ))

    pdf.close()
    return table_images


def download_all_dois(
    dois: list[str],
    dest_dir: str = _DEFAULT_PDF_DIR,
) -> dict[str, str]:
    """Download PDFs for a list of DOIs. Returns {doi: path_or_error}."""
    results = {}
    for doi in dois:
        try:
            path = _download_pdf_from_doi(doi, dest_dir=dest_dir)
            results[doi] = path
            print(f"  OK: {doi} → {path}")
        except Exception as e:
            results[doi] = f"ERROR: {e}"
            print(f"  FAIL: {doi} — {e}")
    return results
