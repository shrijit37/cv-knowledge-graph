"""Document loading (PDF -> text) with an OCR fallback for scanned files.

* Downloaded from a URL is supported via ``requests`` when available.
* PDFs are read with ``pypdf``; if the page yields no text it is treated as a
  scan and OCR'd with ``tesseract`` (via the ``pytesseract``/``PIL`` stack when
  installed). The OCR path is optional so the pipeline runs without it.
"""

from __future__ import annotations

import io
from pathlib import Path

from .build_graph import merge_graphs  # noqa: F401  (re-export for convenience)
from .build_graph import KG  # noqa: F401


def load_pdf(path: Path, ocr: bool = True, dpi: int = 200) -> str:
    """Return the text content of ``path`` (PDF or plain text)."""
    if path.suffix.lower() == ".pdf":
        return _extract_pdf(path, ocr=ocr, dpi=dpi)
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_pdf(path: Path, ocr: bool, dpi: int) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError("pypdf is required to read PDFs: pip install pypdf") from exc

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            parts.append(text)
        elif ocr:
            parts.append(_ocr_page(path, page, dpi))
    return "\n".join(parts)


def _ocr_page(path: Path, page: Any, dpi: int) -> str:  # pragma: no cover - env-dependent
    """OCR a scanned page. Set ``PYTESSERACT_CMD`` to override the tesseract path."""
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        return ""
    try:
        cmd = __import__("os").getenv("PYTESSERACT_CMD")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        pdf_bytes = _page_to_png_bytes(path, page)
        img = Image.open(io.BytesIO(pdf_bytes)).convert("L")
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def _page_to_png_bytes(path: Path, page: Any) -> bytes:  # pragma: no cover
    try:
        import fitz  # PyMuPDF, optional
    except ImportError:
        return b""
    doc = fitz.open(str(path))
    pix = doc[page.page_number - 1].get_pixmap(dpi=200)
    return pix.tobytes("png")
