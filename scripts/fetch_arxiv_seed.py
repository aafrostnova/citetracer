from __future__ import annotations

import argparse
import json
import os
import re
import tarfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_API_URL = "https://export.arxiv.org/api/query"


@dataclass
class ArxivEntry:
    paper_id: str
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    year: int | None
    pdf_url: str
    source_url: str
    abstract_url: str



def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned or "paper"



def _download_binary(url: str, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "citation-checker/1.0"})
    try:
        with urlopen(request, timeout=45) as response:
            output_path.write_bytes(response.read())
        return True
    except Exception:
        return False



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
        abstract_url = abstract_url.replace("http://arxiv.org", "https://arxiv.org")
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
        pdf_url = pdf_url.replace("http://arxiv.org", "https://arxiv.org")
        source_url = abstract_url.replace("/abs/", "/e-print/") if abstract_url else ""

        entries.append(
            ArxivEntry(
                paper_id=arxiv_id,
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                published=published,
                year=year,
                pdf_url=pdf_url,
                source_url=source_url,
                abstract_url=abstract_url,
            )
        )
    return entries



def download_pdf(pdf_url: str, output_path: Path) -> bool:
    return _download_binary(pdf_url, output_path)



def download_source_archive(source_url: str, output_path: Path) -> bool:
    return _download_binary(source_url, output_path)



def _is_within_directory(directory: Path, target: Path) -> bool:
    directory_abs = str(directory.resolve())
    target_abs = str(target.resolve())
    return os.path.commonpath([directory_abs]) == os.path.commonpath([directory_abs, target_abs])



def _safe_extract_tar(archive_path: Path, output_dir: Path) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, mode="r:*") as tar:
            for member in tar.getmembers():
                candidate = output_dir / member.name
                if not _is_within_directory(output_dir, candidate):
                    return False
            tar.extractall(path=output_dir)
        return True
    except tarfile.TarError:
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



def build_manifest(
    entries: list[ArxivEntry],
    pdf_output_dir: Path,
    source_output_dir: Path,
    pdf_downloaded: dict[str, bool] | None = None,
    source_downloaded: dict[str, bool] | None = None,
    source_extracted: dict[str, bool] | None = None,
) -> dict:
    pdf_downloaded = pdf_downloaded or {}
    source_downloaded = source_downloaded or {}
    source_extracted = source_extracted or {}

    items = []
    for entry in entries:
        safe_name = _safe_filename(entry.paper_id)
        local_pdf = pdf_output_dir / f"{safe_name}.pdf"
        local_source_archive = source_output_dir / f"{safe_name}.tar"
        local_source_extract_dir = source_output_dir / safe_name
        items.append(
            {
                "paper_id": entry.paper_id,
                "arxiv_id": entry.arxiv_id,
                "title": entry.title,
                "published": entry.published,
                "pdf_url": entry.pdf_url,
                "source_url": entry.source_url,
                "pdf_path": str(local_pdf),
                "source_archive_path": str(local_source_archive),
                "source_extract_dir": str(local_source_extract_dir),
                "pdf_downloaded": bool(pdf_downloaded.get(entry.paper_id, False)),
                "source_downloaded": bool(source_downloaded.get(entry.paper_id, False)),
                "source_extracted": bool(source_extracted.get(entry.paper_id, False)),
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
    pdf_output_dir: Path,
    source_output_dir: Path,
    metadata_out: Path,
    manifest_out: Path,
    mirror_out: Path,
    download_pdfs: bool,
    download_sources: bool,
    extract_sources: bool,
) -> dict:
    feed_xml = fetch_feed(query=query, max_results=max_results)
    entries = parse_feed(feed_xml)

    pdf_downloaded: dict[str, bool] = {}
    source_downloaded: dict[str, bool] = {}
    source_extracted: dict[str, bool] = {}

    for entry in entries:
        safe_name = _safe_filename(entry.paper_id)

        if download_pdfs:
            pdf_target = pdf_output_dir / f"{safe_name}.pdf"
            pdf_downloaded[entry.paper_id] = download_pdf(entry.pdf_url, pdf_target)

        if download_sources:
            source_archive = source_output_dir / f"{safe_name}.tar"
            source_downloaded[entry.paper_id] = download_source_archive(entry.source_url, source_archive)
            if extract_sources and source_downloaded[entry.paper_id]:
                source_extract_dir = source_output_dir / safe_name
                source_extracted[entry.paper_id] = _safe_extract_tar(source_archive, source_extract_dir)

    metadata_rows = [entry.__dict__ for entry in entries]
    write_jsonl(metadata_out, metadata_rows)
    write_jsonl(mirror_out, build_mirror_records(entries))

    manifest = build_manifest(
        entries=entries,
        pdf_output_dir=pdf_output_dir,
        source_output_dir=source_output_dir,
        pdf_downloaded=pdf_downloaded,
        source_downloaded=source_downloaded,
        source_extracted=source_extracted,
    )
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "query": query,
        "count": len(entries),
        "metadata_out": str(metadata_out),
        "manifest_out": str(manifest_out),
        "mirror_out": str(mirror_out),
        "download_pdfs": download_pdfs,
        "download_sources": download_sources,
        "extract_sources": extract_sources,
        "pdf_downloaded_count": sum(1 for ok in pdf_downloaded.values() if ok),
        "source_downloaded_count": sum(1 for ok in source_downloaded.values() if ok),
        "source_extracted_count": sum(1 for ok in source_extracted.values() if ok),
    }



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch initial arXiv seed data for citation-checker smoke tests.")
    parser.add_argument("--query", default="cat:cs.LG", help="arXiv API search query.")
    parser.add_argument("--max-results", type=int, default=3, help="Maximum number of papers to fetch.")
    parser.add_argument("--pdf-output-dir", default="data/seed/arxiv_pdfs", help="Directory for downloaded PDFs.")
    parser.add_argument("--source-output-dir", default="data/seed/arxiv_sources", help="Directory for downloaded arXiv source archives and extractions.")
    parser.add_argument("--metadata-out", default="data/real_world/arxiv_seed_metadata.jsonl", help="JSONL metadata output path.")
    parser.add_argument("--manifest-out", default="data/real_world/arxiv_seed_manifest.json", help="Manifest output path.")
    parser.add_argument("--mirror-out", default="data/real_world/arxiv_seed_mirror.jsonl", help="Offline mirror JSONL output path.")
    parser.add_argument("--download-pdfs", action="store_true", help="Download PDFs to pdf-output-dir.")
    parser.add_argument("--download-sources", action="store_true", help="Download arXiv source archives to source-output-dir.")
    parser.add_argument("--extract-sources", action="store_true", help="Extract source archives under source-output-dir/<paper_id>/.")
    return parser



def main() -> None:
    args = build_arg_parser().parse_args()
    result = run(
        query=args.query,
        max_results=args.max_results,
        pdf_output_dir=Path(args.pdf_output_dir),
        source_output_dir=Path(args.source_output_dir),
        metadata_out=Path(args.metadata_out),
        manifest_out=Path(args.manifest_out),
        mirror_out=Path(args.mirror_out),
        download_pdfs=args.download_pdfs,
        download_sources=args.download_sources,
        extract_sources=args.extract_sources,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
