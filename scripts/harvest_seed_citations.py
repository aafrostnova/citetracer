"""Harvest seed citations from academic connectors by querying topics × venues.

Builds a topic/venue/year matrix, hits multiple academic APIs in parallel,
and writes the union (deduped, richest variant kept per paper) to a JSON file
usable as input to scripts/generate_mutations.py.

Sources:
- OpenAlex         — best metadata coverage (venue/pages/publisher).
- CrossRef         — DOI 100% coverage.
- DBLP             — gold standard for CS venue names (NeurIPS/ICML acronyms).
- Semantic Scholar — citation graph + author IDs (no key needed up to 100 RPS).
- arXiv            — guarantees arxiv_id for preprints (used for P3/P4 seeds).
- ACL Anthology    — NLP proceedings (ACL/EMNLP/NAACL/EACL/COLING) with full
                     anthology IDs, pages, and venue names.

Writes incrementally after each (topic, venue, year, source) query — if the
script is interrupted or a connector rate-limits, prior progress is preserved.

Usage:
    python scripts/harvest_seed_citations.py \
        --out data/bib/harvested_seeds.json \
        --per-query 5

Output schema (one entry per citation):
    {
        "title": ..., "authors": [...], "venue": ..., "year": ...,
        "doi": ..., "arxiv_id": ..., "url": ...,
        "volume": ..., "pages": ..., "publisher": ..., "location": "",
        "_source": "openalex" | "crossref",
        "_query_topic": ..., "_query_venue": ..., "_query_year": ...,
    }
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

# Topics covering ML/AI subfields. Extend this list to broaden the seed pool.
DEFAULT_TOPICS = [
    "large language models",
    "diffusion models",
    "transformer attention",
    "reinforcement learning",
    "graph neural networks",
    "contrastive learning",
    "in-context learning",
    "retrieval augmented generation",
    "prompt injection",
    "multi-agent systems",
    "vision language models",
    "speech recognition",
    "knowledge distillation",
    "neural machine translation",
    "federated learning",
]

# Target venues — chosen to cover both DBLP-rich (NeurIPS, ICML, AAAI) and
# CrossRef-rich (TPAMI, JMLR, Nature, Science) sources.
DEFAULT_VENUES = [
    "NeurIPS",
    "ICML",
    "ICLR",
    "ACL",
    "EMNLP",
    "AAAI",
    "CVPR",
    "Journal of Machine Learning Research",
    "IEEE Transactions on Pattern Analysis and Machine Intelligence",
]

DEFAULT_YEARS = [2020, 2021, 2022, 2023, 2024]


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

def query_openalex(topic: str, venue: str, year: int, per_query: int) -> list[dict[str, Any]]:
    """Query OpenAlex for papers matching topic + venue + year.

    OpenAlex doesn't support filtering by venue display_name with .search suffix
    in stable API; instead we put the venue into the search string and filter
    only by year, then post-filter on returned source name.
    """
    params = {
        "search": f"{topic} {venue}",
        "filter": f"publication_year:{year}",
        "per-page": per_query * 4,  # over-fetch to allow post-filtering
        "select": "id,doi,display_name,authorships,publication_year,primary_location,biblio,locations",
    }
    try:
        resp = requests.get("https://api.openalex.org/works", params=params, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        print(f"  openalex error ({topic}/{venue}/{year}): {exc}", flush=True)
        return []
    out = []
    venue_lower = venue.lower()
    for work in resp.json().get("results", []):
        rec = _normalize_openalex(work, topic, venue, year)
        if not rec.get("title"):
            continue
        # Post-filter: ensure the returned source actually matches our target venue
        rec_venue = rec.get("venue", "").lower()
        if venue_lower in rec_venue or rec_venue in venue_lower:
            out.append(rec)
        if len(out) >= per_query:
            break
    return out


def _normalize_openalex(work: dict, topic: str, venue: str, year: int) -> dict[str, Any]:
    authors = []
    for a in work.get("authorships", []):
        name = (a.get("author") or {}).get("display_name", "")
        if name:
            authors.append(name)

    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    biblio = work.get("biblio") or {}
    fp, lp = biblio.get("first_page", ""), biblio.get("last_page", "")
    pages = f"{fp}-{lp}" if fp and lp else (fp or lp or "")
    doi = str(work.get("doi") or "")
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    elif doi.startswith("http://doi.org/"):
        doi = doi[len("http://doi.org/"):]
    elif doi.startswith("doi:"):
        doi = doi[len("doi:"):]

    # arxiv_id from any location with arxiv host
    arxiv_id = ""
    for loc in work.get("locations", []) or [primary]:
        src = (loc or {}).get("source") or {}
        if "arxiv" in (src.get("display_name") or "").lower():
            landing = (loc or {}).get("landing_page_url", "") or ""
            if "arxiv.org/abs/" in landing:
                arxiv_id = landing.split("arxiv.org/abs/")[-1].rstrip("/")
                break

    return {
        "title": str(work.get("display_name") or ""),
        "authors": authors,
        "venue": str(source.get("display_name") or ""),
        "year": work.get("publication_year"),
        "doi": doi.lower(),
        "arxiv_id": arxiv_id,
        "url": str(work.get("id") or ""),
        "volume": str(biblio.get("volume", "") or ""),
        "pages": pages,
        "publisher": str(source.get("host_organization_name") or source.get("publisher", "") or ""),
        "location": "",
        "_source": "openalex",
        "_query_topic": topic,
        "_query_venue": venue,
        "_query_year": year,
    }


# ---------------------------------------------------------------------------
# CrossRef (for DOI-rich seeds)
# ---------------------------------------------------------------------------

def query_crossref(topic: str, venue: str, year: int, per_query: int) -> list[dict[str, Any]]:
    """Query CrossRef. Returns DOI-bearing entries (post-filtered by venue)."""
    params = {
        "query.bibliographic": f"{topic} {venue}",
        "filter": f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31",
        "rows": per_query * 4,
        "select": "DOI,title,author,container-title,issued,volume,page,publisher",
        "mailto": "research@example.com",
    }
    try:
        resp = requests.get("https://api.crossref.org/works", params=params, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        print(f"  crossref error ({topic}/{venue}/{year}): {exc}", flush=True)
        return []
    out = []
    venue_lower = venue.lower()
    for item in resp.json().get("message", {}).get("items", []):
        norm = _normalize_crossref(item, topic, venue, year)
        if not (norm.get("doi") and norm.get("title")):
            continue
        rec_venue = norm.get("venue", "").lower()
        if venue_lower in rec_venue or rec_venue in venue_lower or not rec_venue:
            out.append(norm)
        if len(out) >= per_query:
            break
    return out


def _normalize_crossref(item: dict, topic: str, venue: str, year: int) -> dict[str, Any]:
    titles = item.get("title") or []
    containers = item.get("container-title") or []
    authors = []
    for a in item.get("author", []) or []:
        given = a.get("given", "")
        family = a.get("family", "")
        full = f"{given} {family}".strip() or a.get("name", "")
        if full:
            authors.append(full)
    issued = (item.get("issued") or {}).get("date-parts") or [[None]]
    pub_year = issued[0][0] if issued and issued[0] else None
    return {
        "title": titles[0] if titles else "",
        "authors": authors,
        "venue": containers[0] if containers else "",
        "year": pub_year,
        "doi": str(item.get("DOI", "") or "").lower(),
        "arxiv_id": "",
        "url": f"https://doi.org/{item.get('DOI', '')}" if item.get("DOI") else "",
        "volume": str(item.get("volume", "") or ""),
        "pages": str(item.get("page", "") or ""),
        "publisher": str(item.get("publisher", "") or ""),
        "location": "",
        "_source": "crossref",
        "_query_topic": topic,
        "_query_venue": venue,
        "_query_year": year,
    }


# ---------------------------------------------------------------------------
# DBLP (CS venue gold standard)
# ---------------------------------------------------------------------------

def query_dblp(topic: str, venue: str, year: int, per_query: int) -> list[dict[str, Any]]:
    """Query DBLP. Strong on CS venue names."""
    params = {
        "q": f"{topic} {venue} {year}",
        "format": "json",
        "h": per_query * 4,
    }
    try:
        resp = requests.get("https://dblp.org/search/publ/api", params=params, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        print(f"  dblp error ({topic}/{venue}/{year}): {exc}", flush=True)
        return []
    hits = (((resp.json() or {}).get("result") or {}).get("hits") or {}).get("hit") or []
    out = []
    venue_lower = venue.lower()
    for hit in hits:
        info = (hit or {}).get("info") or {}
        if str(info.get("year", "")) != str(year):
            continue
        rec = _normalize_dblp(info, topic, venue, year)
        rec_venue = (rec.get("venue") or "").lower()
        if not rec.get("title"):
            continue
        if venue_lower in rec_venue or rec_venue in venue_lower:
            out.append(rec)
        if len(out) >= per_query:
            break
    return out


def _normalize_dblp(info: dict, topic: str, venue: str, year: int) -> dict[str, Any]:
    raw_authors = ((info.get("authors") or {}).get("author")) or []
    if isinstance(raw_authors, dict):
        raw_authors = [raw_authors]
    authors = []
    for a in raw_authors:
        if isinstance(a, dict):
            n = a.get("text", "")
        else:
            n = str(a)
        if n:
            # strip trailing DBLP disambiguation digits like "Boyuan Chen 0003"
            import re as _re
            n = _re.sub(r"\s+\d{4}$", "", n)
            authors.append(n)
    return {
        "title": str(info.get("title", "") or "").rstrip("."),
        "authors": authors,
        "venue": str(info.get("venue", "") or ""),
        "year": int(info["year"]) if str(info.get("year", "")).isdigit() else None,
        "doi": str(info.get("doi", "") or "").lower(),
        "arxiv_id": "",
        "url": str(info.get("url", "") or info.get("ee", "") or ""),
        "volume": str(info.get("volume", "") or ""),
        "pages": str(info.get("pages", "") or ""),
        "publisher": "",
        "location": "",
        "_source": "dblp",
        "_query_topic": topic,
        "_query_venue": venue,
        "_query_year": year,
    }


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

def query_semanticscholar(topic: str, venue: str, year: int, per_query: int) -> list[dict[str, Any]]:
    """Query Semantic Scholar. No API key needed for low volume."""
    params = {
        "query": f"{topic} {venue}",
        "year": year,
        "limit": per_query * 4,
        "fields": "title,authors,venue,year,externalIds,publicationVenue,journal",
    }
    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params, timeout=20,
        )
        resp.raise_for_status()
    except Exception as exc:
        print(f"  s2 error ({topic}/{venue}/{year}): {exc}", flush=True)
        return []
    out = []
    venue_lower = venue.lower()
    for item in (resp.json() or {}).get("data", []) or []:
        rec = _normalize_s2(item, topic, venue, year)
        if not rec.get("title"):
            continue
        rec_venue = (rec.get("venue") or "").lower()
        if venue_lower in rec_venue or rec_venue in venue_lower or not rec_venue:
            out.append(rec)
        if len(out) >= per_query:
            break
    return out


def _normalize_s2(item: dict, topic: str, venue: str, year: int) -> dict[str, Any]:
    authors = [a.get("name", "") for a in (item.get("authors") or []) if a.get("name")]
    ext = item.get("externalIds") or {}
    journal = item.get("journal") or {}
    return {
        "title": str(item.get("title", "") or ""),
        "authors": authors,
        "venue": str(item.get("venue", "") or (item.get("publicationVenue") or {}).get("name", "") or ""),
        "year": item.get("year"),
        "doi": str(ext.get("DOI", "") or "").lower(),
        "arxiv_id": str(ext.get("ArXiv", "") or "").lower(),
        "url": str(item.get("url", "") or ""),
        "volume": str(journal.get("volume", "") or ""),
        "pages": str(journal.get("pages", "") or ""),
        "publisher": "",
        "location": "",
        "_source": "semanticscholar",
        "_query_topic": topic,
        "_query_venue": venue,
        "_query_year": year,
    }


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------

def query_arxiv(topic: str, venue: str, year: int, per_query: int) -> list[dict[str, Any]]:
    """Query arXiv. Returns Atom XML; we parse it. Strong for arxiv_id seeds.
    arXiv API recommends >=3s between requests; caller should set --sleep accordingly."""
    params = {
        "search_query": f"all:{topic}",
        "start": 0,
        "max_results": per_query * 4,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": "citation-hallucination-detection-research/0.1 (research)"}
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(
                "https://export.arxiv.org/api/query",
                params=params, headers=headers, timeout=30,
            )
            if resp.status_code == 429:
                wait = 5 * (attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except Exception as exc:
            print(f"  arxiv error attempt {attempt+1} ({topic}/{venue}/{year}): {exc}", flush=True)
            time.sleep(3)
    if resp is None or resp.status_code != 200:
        return []

    import xml.etree.ElementTree as ET
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    out = []
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return []
    for entry in root.findall("atom:entry", ns):
        rec = _normalize_arxiv(entry, ns, topic, venue, year)
        if not rec.get("title"):
            continue
        # Filter by year
        if rec.get("year") != year:
            continue
        out.append(rec)
        if len(out) >= per_query:
            break
    return out


def _normalize_arxiv(entry, ns, topic: str, venue: str, year: int) -> dict[str, Any]:
    def text(tag: str) -> str:
        node = entry.find(f"atom:{tag}", ns)
        return (node.text or "").strip() if node is not None else ""

    arxiv_url = text("id")  # http://arxiv.org/abs/2503.09573v1
    arxiv_id = arxiv_url.split("arxiv.org/abs/")[-1] if "arxiv.org/abs/" in arxiv_url else ""
    published = text("published")  # 2025-03-12T...
    pub_year = int(published[:4]) if published[:4].isdigit() else None

    authors = []
    for a in entry.findall("atom:author", ns):
        n = a.find("atom:name", ns)
        if n is not None and n.text:
            authors.append(n.text.strip())

    return {
        "title": " ".join(text("title").split()),
        "authors": authors,
        "venue": "arXiv",
        "year": pub_year,
        "doi": "",
        "arxiv_id": arxiv_id.lower(),
        "url": arxiv_url,
        "volume": "",
        "pages": "",
        "publisher": "",
        "location": "",
        "_source": "arxiv",
        "_query_topic": topic,
        "_query_venue": venue,
        "_query_year": year,
    }


# ---------------------------------------------------------------------------
# ACL Anthology (NLP proceedings)
# ---------------------------------------------------------------------------

# Map our generic venue names to ACL Anthology volume suffixes.
_ACL_VOLUME_SUFFIXES = {
    "ACL":    ["acl-long", "acl-short", "acl-main", "findings-acl"],
    "EMNLP":  ["emnlp-main", "emnlp-long", "emnlp-short", "findings-emnlp"],
    "NAACL":  ["naacl-long", "naacl-short", "naacl-main", "findings-naacl"],
    "EACL":   ["eacl-long", "eacl-short", "eacl-main", "findings-eacl"],
    "COLING": ["coling-main", "coling"],
    "AACL":   ["aacl-main", "aacl-short", "findings-aacl"],
}


def query_acl_anthology(topic: str, venue: str, year: int, per_query: int) -> list[dict[str, Any]]:
    """Query ACL Anthology. Fetches per-year volume .bib files and filters by topic match in title.

    Each (venue, year) pair maps to 1-4 volume URLs like:
        https://aclanthology.org/volumes/2024.acl-long.bib
    """
    venue_upper = venue.upper()
    suffixes = _ACL_VOLUME_SUFFIXES.get(venue_upper)
    if not suffixes:
        return []

    topic_words = {w for w in topic.lower().split() if len(w) >= 4}
    out: list[dict[str, Any]] = []

    for suffix in suffixes:
        url = f"https://aclanthology.org/volumes/{year}.{suffix}.bib"
        try:
            resp = requests.get(url, timeout=20)
        except Exception as exc:
            print(f"  acl_anthology error ({venue}/{year}/{suffix}): {exc}", flush=True)
            continue
        if resp.status_code != 200:
            continue
        entries = _parse_bibtex_entries(resp.text)
        for entry in entries:
            title_lower = entry.get("title", "").lower()
            if topic_words and not any(w in title_lower for w in topic_words):
                continue
            rec = _normalize_acl(entry, topic, venue, year)
            if rec.get("title"):
                out.append(rec)
            if len(out) >= per_query:
                return out

    return out


def _parse_bibtex_entries(bibtex_text: str) -> list[dict[str, str]]:
    """Minimal BibTeX parser — handles `@type{key, field = {value}, ...}` entries."""
    import re as _re
    entries = []
    for m in _re.finditer(r"@(\w+)\s*\{\s*([^,]+),\s*(.*?)\n\}\n", bibtex_text, _re.DOTALL):
        entry_type = m.group(1).lower()
        if entry_type in ("comment", "string", "preamble"):
            continue
        key = m.group(2).strip()
        body = m.group(3)
        fields: dict[str, str] = {"entry_type": entry_type, "key": key}
        # Match: `fieldname = {value}`  or  `fieldname = "value"`
        for fm in _re.finditer(r'(\w+)\s*=\s*[\{"]((?:[^{}"]|\{[^{}]*\})*)[\}"]\s*,?', body):
            fname = fm.group(1).lower()
            fval = fm.group(2).strip()
            # Clean nested braces used for capitalization protection
            fval = _re.sub(r"[{}]", "", fval).strip()
            fields[fname] = fval
        entries.append(fields)
    return entries


def _normalize_acl(entry: dict, topic: str, venue: str, year: int) -> dict[str, Any]:
    import re as _re
    authors_raw = entry.get("author", "")
    # Collapse all whitespace (BibTeX author field is often multi-line)
    authors_raw = _re.sub(r"\s+", " ", authors_raw).strip()
    # Split on " and " (BibTeX standard separator)
    authors = [a.strip() for a in _re.split(r"\s+and\s+", authors_raw) if a.strip()]
    # Convert "Last, First" → "First Last"
    authors = [
        (a.split(",", 1)[1].strip() + " " + a.split(",", 1)[0].strip()) if "," in a else a
        for a in authors
    ]

    pages_str = entry.get("pages", "").replace("--", "-")
    return {
        "title": entry.get("title", ""),
        "authors": authors,
        "venue": entry.get("booktitle", "") or entry.get("journal", "") or venue,
        "year": int(entry["year"]) if entry.get("year", "").isdigit() else year,
        "doi": (entry.get("doi", "") or "").lower(),
        "arxiv_id": "",
        "url": entry.get("url", ""),
        "volume": entry.get("volume", ""),
        "pages": pages_str,
        "publisher": entry.get("publisher", "") or "Association for Computational Linguistics",
        "location": entry.get("address", ""),
        "_source": "acl_anthology",
        "_query_topic": topic,
        "_query_venue": venue,
        "_query_year": year,
    }


# ---------------------------------------------------------------------------
# Dedup: keep the variant with the most populated fields per unique paper
# ---------------------------------------------------------------------------

# Fields that count toward "richness" — empty / None / [] does not score.
_FIELD_KEYS = ("title", "authors", "venue", "year", "doi", "arxiv_id",
               "url", "volume", "pages", "publisher", "location")


def _richness_score(rec: dict[str, Any]) -> int:
    """Number of populated fields. Used to pick the best variant per paper."""
    score = 0
    for k in _FIELD_KEYS:
        v = rec.get(k)
        if isinstance(v, list):
            if v:
                score += 1
        elif v not in (None, "", 0):
            score += 1
    return score


def _identity_key(rec: dict[str, Any]) -> str:
    """DOI if present, else normalized title prefix."""
    doi = (rec.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"
    title = (rec.get("title") or "").strip().lower()[:100]
    return f"title:{title}"


def _dedup_keep_richest(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for r in records:
        key = _identity_key(r)
        if not key.replace("doi:", "").replace("title:", "").strip():
            continue
        existing = by_key.get(key)
        if existing is None or _richness_score(r) > _richness_score(existing):
            by_key[key] = r
    return list(by_key.values())


# ---------------------------------------------------------------------------
# Orchestrate
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--topics", nargs="*", default=None, help="Topics to query (defaults to DEFAULT_TOPICS)")
    parser.add_argument("--venues", nargs="*", default=None, help="Venues to query")
    parser.add_argument("--years", nargs="*", type=int, default=None, help="Years to query")
    parser.add_argument("--per-query", type=int, default=3, help="Max records per (topic, venue, year, source)")
    parser.add_argument(
        "--sources", nargs="+",
        choices=["openalex", "crossref", "dblp", "semanticscholar", "arxiv", "acl_anthology"],
        default=["openalex", "crossref", "dblp", "semanticscholar", "arxiv", "acl_anthology"],
    )
    parser.add_argument("--sleep", type=float, default=0.3, help="Seconds between requests (rate limit)")
    parser.add_argument("--dedupe-by-doi", action="store_true", default=True)
    args = parser.parse_args()

    topics = args.topics or DEFAULT_TOPICS
    venues = args.venues or DEFAULT_VENUES
    years = args.years or DEFAULT_YEARS

    n_queries = len(topics) * len(venues) * len(years) * len(args.sources)
    print(f"Will issue up to {n_queries} queries ({len(topics)} topics × {len(venues)} venues × {len(years)} years × {len(args.sources)} sources)", flush=True)

    fn_by_source = {
        "openalex": query_openalex,
        "crossref": query_crossref,
        "dblp": query_dblp,
        "semanticscholar": query_semanticscholar,
        "arxiv": query_arxiv,
        "acl_anthology": query_acl_anthology,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Running state: dedup-keyed dict, rewritten to disk after each query batch
    # so progress is preserved if the script is interrupted or rate-limited.
    by_key: dict[str, dict[str, Any]] = {}

    # Resume mode: if the output file already has content, load and continue
    if out.exists() and out.stat().st_size > 0:
        try:
            existing = json.loads(out.read_text())
            for r in existing:
                key = _identity_key(r)
                if key.replace("doi:", "").replace("title:", "").strip():
                    by_key[key] = r
            print(f"Resuming: loaded {len(by_key)} existing records from {out}", flush=True)
        except json.JSONDecodeError:
            pass

    def _flush() -> None:
        out.write_text(json.dumps(list(by_key.values()), indent=2, ensure_ascii=False))

    for topic in topics:
        for venue in venues:
            for year in years:
                for source in args.sources:
                    fn = fn_by_source[source]
                    recs = fn(topic, venue, year, args.per_query)
                    added = 0
                    for r in recs:
                        key = _identity_key(r)
                        if not key.replace("doi:", "").replace("title:", "").strip():
                            continue
                        existing = by_key.get(key)
                        if existing is None or _richness_score(r) > _richness_score(existing):
                            by_key[key] = r
                            added += 1
                    if added:
                        _flush()
                    print(f"  [{source}] {topic[:30]} / {venue} / {year} → +{len(recs)} raw, +{added} unique (total unique: {len(by_key)})", flush=True)
                    time.sleep(args.sleep)

    all_records = list(by_key.values())
    _flush()

    # Summary
    by_source = {}
    with_doi = sum(1 for r in all_records if r.get("doi"))
    with_pages = sum(1 for r in all_records if r.get("pages"))
    with_publisher = sum(1 for r in all_records if r.get("publisher"))
    for r in all_records:
        by_source[r["_source"]] = by_source.get(r["_source"], 0) + 1
    print(f"\nWrote {len(all_records)} unique citations to {out}")
    print(f"  by source: {by_source}")
    print(f"  with DOI: {with_doi}")
    print(f"  with pages: {with_pages}")
    print(f"  with publisher: {with_publisher}")


if __name__ == "__main__":
    main()
