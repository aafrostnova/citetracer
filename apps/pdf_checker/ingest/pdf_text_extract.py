from __future__ import annotations

from pathlib import Path



def extract_pdf_pages(pdf_path: str | Path) -> list[str]:
    path = Path(pdf_path)
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return pages
    except Exception:
        # Fallback keeps the pipeline operational in constrained environments.
        raw = path.read_bytes()
        text = raw.decode("latin-1", errors="ignore")
        return [text]
