from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_API_URL = "http://export.arxiv.org/api/query"


@dataclass
class ArxivEntry:
    paper_id: str
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    year: int | None
    pdf_url: str
    abstract_url: str



def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "paper"



def fetch_feed(query: str, max_results: int, start: int = 0) -> str:
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    target = f"{ARXIV_API_URL}?{urlencode(params)}"
    request = Request(target, headers={"User-Agent": "citation-checker/1.0"})
    with urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8")



def parse_feed(feed_xml: str) -> list[ArxivEntry]:
    root = ET.fromstring(feed_xml)
    entries: list[ArxivEntry] = []
    for node in root.findall("atom:entry", ATOM_NS):
        abstract_url = (node.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()
        raw_title = (node.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
        title = " ".join(raw_title.split())
        published = (node.findtext("atom:published", default="", namespaces=ATOM_NS) or "").strip()
        year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None

        authors: list[str] = []
        for author_node in node.findall("atom:author", ATOM_NS):
            name = (author_node.findtext("atom:name", default="", namespaces=ATOM_NS) or "").strip()
            if name:
                authors.append(name)

        arxiv_id = abstract_url.rsplit("/", maxsplit=1)[-1]

        pdf_url = ""
        for link in node.findall("atom:link", ATOM_NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break
        if not pdf_url and abstract_url:
            pdf_url = abstract_url.replace("/abs/", "/pdf/") + ".pdf"

        entries.append(
            ArxivEntry(
                paper_id=arxiv_id,
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                published=published,
                year=year,
                pdf_url=pdf_url,
                abstract_url=abstract_url,
            )
        )
    return entries



def download_pdf(pdf_url: str, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(pdf_url, headers={"User-Agent": "citation-checker/1.0"})
    try:
        with urlopen(request, timeout=30) as response:
            output_path.write_bytes(response.read())
        return True
    except Exception:
        return False



def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if content:
        content = f"{content}\n"
    path.write_text(content, encoding="utf-8")



def build_mirror_records(entries: list[ArxivEntry]) -> list[dict]:
    mirror = []
    for entry in entries:
        mirror.append(
            {
                "title": entry.title,
                "authors": entry.authors,
                "venue": "arXiv",
                "year": entry.year,
                "doi": "",
                "arxiv_id": entry.arxiv_id,
                "url": entry.abstract_url,
            }
        )
    return mirror



def build_manifest(entries: list[ArxivEntry], output_dir: Path) -> dict:
    items = []
    for entry in entries:
        local_pdf = output_dir / f"{_safe_filename(entry.paper_id)}.pdf"
        items.append(
            {
                "paper_id": entry.paper_id,
                "arxiv_id": entry.arxiv_id,
                "title": entry.title,
                "published": entry.published,
                "pdf_url": entry.pdf_url,
                "pdf_path": str(local_pdf),
            }
        )
    return {
        "generated_at_unix": int(time.time()),
        "source": "arxiv",
        "items": items,
    }



def run(
    query: str,
    max_results: int,
    output_dir: Path,
    metadata_out: Path,
    manifest_out: Path,
    mirror_out: Path,
    download_pdfs: bool,
) -> dict:
    feed_xml = fetch_feed(query=query, max_results=max_results)
    entries = parse_feed(feed_xml)

    if download_pdfs:
        for entry in entries:
            target = output_dir / f"{_safe_filename(entry.paper_id)}.pdf"
            download_pdf(entry.pdf_url, target)

    metadata_rows = [entry.__dict__ for entry in entries]
    write_jsonl(metadata_out, metadata_rows)
    write_jsonl(mirror_out, build_mirror_records(entries))

    manifest = build_manifest(entries, output_dir)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "query": query,
        "count": len(entries),
        "metadata_out": str(metadata_out),
        "manifest_out": str(manifest_out),
        "mirror_out": str(mirror_out),
        "download_pdfs": download_pdfs,
    }



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch initial arXiv seed data for citation-checker smoke tests.")
    parser.add_argument("--query", default="cat:cs.LG", help="arXiv API search query.")
    parser.add_argument("--max-results", type=int, default=3, help="Maximum number of papers to fetch.")
    parser.add_argument("--output-dir", default="data/seed/arxiv_pdfs", help="Directory for downloaded PDFs.")
    parser.add_argument("--metadata-out", default="data/real_world/arxiv_seed_metadata.jsonl", help="JSONL metadata output path.")
    parser.add_argument("--manifest-out", default="data/real_world/arxiv_seed_manifest.json", help="Manifest output path.")
    parser.add_argument("--mirror-out", default="data/real_world/arxiv_seed_mirror.jsonl", help="Offline mirror JSONL output path.")
    parser.add_argument("--download-pdfs", action="store_true", help="Download PDFs to output-dir.")
    return parser



def main() -> None:
    args = build_arg_parser().parse_args()
    result = run(
        query=args.query,
        max_results=args.max_results,
        output_dir=Path(args.output_dir),
        metadata_out=Path(args.metadata_out),
        manifest_out=Path(args.manifest_out),
        mirror_out=Path(args.mirror_out),
        download_pdfs=args.download_pdfs,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
