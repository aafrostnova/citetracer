"""Scan a directory of reference-extract JSONs and collect non-academic citations.

A citation is considered "non-academic" if ANY of:
  - URL matches a known non-academic domain (github, twitter, huggingface, medium,
    substack, personal blogs, youtube, news sites, company/product pages, etc.)
  - Venue/publisher keyword is non-academic (blog, tweet, github, ...)
  - Venue is empty AND title contains non-academic markers
  - No DOI, no arxiv_id, no recognized academic venue, BUT has a URL

Output: JSON list of unique non-academic citations, grouped with source paper.

Usage:
    python scripts/collect_non_academic_citations.py \
        --input-dir artifacts/iclr_2026_reference_extracts_v4 \
        --out data/bib/non_academic_sources.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Domain / venue matchers
# ---------------------------------------------------------------------------

_NON_ACADEMIC_DOMAINS = {
    # Social media
    "twitter.com", "x.com", "mastodon.social", "threads.net", "bsky.app", "reddit.com",
    "facebook.com", "linkedin.com", "instagram.com", "youtube.com", "youtu.be",
    # Code hosting / ML model hubs
    "github.com", "gitlab.com", "bitbucket.org", "huggingface.co",
    # Blogging platforms
    "medium.com", "substack.com", "wordpress.com", "blogspot.com", "tumblr.com",
    "ghost.io", "dev.to", "hashnode.dev", "notion.site", "notion.so",
    # Q&A
    "stackoverflow.com", "stackexchange.com", "quora.com",
    # Company / product blogs
    "openai.com", "anthropic.com", "deepmind.com", "deepmind.google",
    "ai.google", "ai.meta.com", "ai.facebook.com", "research.google",
    "about.fb.com", "blog.google", "blogs.microsoft.com", "blogs.nvidia.com",
    "developer.nvidia.com", "techcrunch.com", "theverge.com", "arstechnica.com",
    "wired.com", "venturebeat.com", "thenextweb.com",
    # News
    "bbc.com", "nytimes.com", "washingtonpost.com", "theguardian.com", "reuters.com",
    "bloomberg.com", "forbes.com", "cnbc.com", "cnn.com", "wsj.com",
    # Package registries
    "pypi.org", "npmjs.com", "cran.r-project.org",
    # Others
    "wikipedia.org", "wikimedia.org",
    # Paperswithcode also tends to point to non-peer-reviewed
    "paperswithcode.com",
}

# Substrings that also indicate non-academic hosting
_NON_ACADEMIC_URL_SUBSTRINGS = [
    "/blog/", "/blogs/", "/news/", "/posts/", "/post/", "/notes/",
    ".substack.com", ".medium.com",
]

# Domains that are neither useful as academic NOR non-academic seeds — skip entirely.
_SKIP_DOMAINS = {
    "anonymous.4open.science",   # anonymized review links from paper submissions
}

# Recognized academic hosts (if ONLY these appear, citation is academic)
_ACADEMIC_DOMAINS = {
    "arxiv.org", "aclanthology.org", "openreview.net", "doi.org", "dx.doi.org",
    "dblp.org", "scholar.google.com", "semanticscholar.org", "openalex.org",
    "ieee.org", "ieeexplore.ieee.org", "acm.org", "dl.acm.org",
    "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov", "europepmc.org",
    "springer.com", "springerlink.com", "link.springer.com",
    "sciencedirect.com", "elsevier.com",
    "nature.com", "science.org", "cell.com", "biorxiv.org", "medrxiv.org",
    "wiley.com", "onlinelibrary.wiley.com", "tandfonline.com",
    "jstor.org", "sagepub.com", "mit.edu", "mlr.press",
    "proceedings.mlr.press", "proceedings.neurips.cc", "papers.nips.cc",
    "cvpr.thecvf.com", "openaccess.thecvf.com", "eccv.ecva.net",
    # ML conference sites
    "neurips.cc", "icml.cc", "iclr.cc", "aaai.org", "ojs.aaai.org",
    "ijcai.org", "colmweb.org",
    # Journal sites
    "jmlr.org", "tacl.cc",
    # arXiv mirrors
    "arxiv-vanity.com",
    # Preprint servers / open-access repositories
    "osf.io", "psyarxiv.com", "chemrxiv.org",
    "ssrn.com", "papers.ssrn.com", "zenodo.org",
    "hal.science", "hal.archives-ouvertes.fr",
    "researchsquare.com", "authorea.com",
    # CS/ML conferences (additional)
    "ecva.net", "openaccess.thecvf.com",
    # Security / systems conferences
    "usenix.org", "ndss-symposium.org", "internetsociety.org",
    # Other CS societies
    "computer.org", "theiet.org", "rsc.org",
    # Publishers
    "royalsocietypublishing.org", "journals.aps.org", "iopscience.iop.org",
    "bmj.com", "nejm.org", "pnas.org", "annualreviews.org",
    "karger.com", "mdpi.com", "frontiersin.org", "aps.org",
    "ams.org", "siam.org", "oup.com", "academic.oup.com",
    "cambridge.org", "cup.org",
    # Arxiv host also can be cited as export
    "export.arxiv.org",
}

# Strong academic venue keywords — if present in venue, the citation is academic
# even if the URL host isn't on our allowlist.
_ACADEMIC_VENUE_RE = re.compile(
    r"\b("
    # Acronyms
    r"neurips|nips|icml|iclr|aaai|ijcai|acl|emnlp|naacl|coling|eacl|aacl|"
    r"cvpr|eccv|iccv|bmvc|wacv|3dv|"
    r"sigkdd|kdd|sigir|sigmod|sigcomm|sigplan|uist|chi|cscw|"
    r"interspeech|icassp|"
    # Security / systems
    r"usenix|ndss|ccs|oakland|oopsla|osdi|sosp|nsdi|asplos|"
    r"security symposium|network and distributed system security|"
    # Phrases
    r"proceedings of|conference on|conference proceedings|workshop on|"
    r"annual meeting|international conference|international joint conference|"
    r"transactions on|journal of|"
    r"advances in neural|annals of|"
    # Spelled-out conference/society names
    r"association for computational linguistics|"
    r"empirical methods in natural language processing|"
    r"international conference on machine learning|"
    r"neural information processing systems|"
    r"north american chapter|"
    r"computer vision and pattern recognition|"
    r"european conference on computer vision|"
    # Journals (named)
    r"nature|science|cell|the lancet|plos|"
    r"jmlr|tacl|tpami|pami|"
    # Generic journal suffixes/prefixes
    r"review of|letters|quarterly|bulletin of|annals of|"
    r"research journal|journal of the|"
    r"studies in|advances in"
    r")\b",
    re.IGNORECASE,
)


# Generic academic venue shape: multi-word titles that include one of these
# strong journal/proceedings tokens, OR contain `&` joining two field names
# (e.g. "Philosophy & Technology", "Mathematics & Statistics").
_GENERIC_JOURNAL_TOKENS = re.compile(
    r"\b(journal|proceedings|review|letters|quarterly|transactions|bulletin|"
    r"annals|studies|research|symposium|workshop|computing|intelligence|"
    r"computation|informatics|learning|physics|chemistry|biology|medicine|"
    r"mathematics|statistics|mechanics|engineering|sciences?|technology|"
    r"artificial intelligence|machine learning)\b",
    re.IGNORECASE,
)


def _venue_looks_academic(venue: str) -> bool:
    """Loose check: venue that contains journal-like tokens OR an ampersand-joined
    field pair (common in academic journals)."""
    if not venue:
        return False
    if "&" in venue and _GENERIC_JOURNAL_TOKENS.search(venue):
        return True
    # Two or more journal-like tokens present → academic
    hits = _GENERIC_JOURNAL_TOKENS.findall(venue)
    if len(hits) >= 2:
        return True
    return False

_NON_ACADEMIC_VENUE_KEYWORDS = [
    "blog", "tweet", "twitter", "medium", "substack", "github", "huggingface",
    "personal website", "personal blog", "news", "press release", "tech report",
    "online", "website", "product page", "documentation", "docs",
    "stackoverflow", "stack exchange", "reddit", "wikipedia",
]


def _domain(url: str) -> str:
    try:
        host = urlparse(url.strip()).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _host_matches(host: str, domains: set[str]) -> bool:
    """True if `host` equals any domain in `domains` OR is a subdomain of one."""
    if not host:
        return False
    if host in domains:
        return True
    for d in domains:
        if host.endswith("." + d):
            return True
    return False


def _venue_is_academic(venue: str) -> bool:
    """True if the venue string strongly indicates an academic publication."""
    if not venue:
        return False
    return bool(_ACADEMIC_VENUE_RE.search(venue))


def _is_non_academic(citation: dict[str, Any]) -> tuple[bool, str]:
    """Return (is_non_academic, reason)."""
    url = (citation.get("url") or "").strip()
    venue = (citation.get("venue") or "").strip().lower()
    publisher = (citation.get("publisher") or "").strip().lower()
    doi = (citation.get("doi") or "").strip()
    arxiv_id = (citation.get("arxiv_id") or "").strip()

    # ------------------------------------------------------------------
    # HARD EXCLUSIONS: these signals mean SKIP (not collected as non-academic).
    # ------------------------------------------------------------------
    # 0-pre: Skip-list domains (e.g. anonymized submission links)
    if url:
        host = _domain(url)
        if _host_matches(host, _SKIP_DOMAINS):
            return False, ""

    # 0a. Has DOI or arxiv_id → academic
    if doi or arxiv_id:
        return False, ""

    # 0b. Venue clearly academic (NeurIPS / ICML / Journal of.../ Proceedings of..., etc.)
    if _venue_is_academic(venue):
        return False, ""
    # 0b2. Looser: venue looks like a journal ("Philosophy & Technology", "Artificial Intelligence Review", ...)
    if _venue_looks_academic(venue):
        return False, ""

    # 0c. URL domain is academic (exact or subdomain match)
    if url:
        host = _domain(url)
        if _host_matches(host, _ACADEMIC_DOMAINS):
            return False, ""
        if ".edu" in host or ".ac." in host:
            return False, ""

    # ------------------------------------------------------------------
    # POSITIVE SIGNALS: flag as non-academic
    # ------------------------------------------------------------------
    # 1. URL domain is in known non-academic list (exact or subdomain match)
    if url:
        host = _domain(url)
        if _host_matches(host, _NON_ACADEMIC_DOMAINS):
            return True, f"url_domain={host}"
        for sub in _NON_ACADEMIC_URL_SUBSTRINGS:
            if sub in url.lower():
                return True, f"url_substring={sub}"

    # 2. Venue / publisher has explicit non-academic keyword
    for kw in _NON_ACADEMIC_VENUE_KEYWORDS:
        if kw in venue:
            return True, f"venue_keyword={kw}"
        if kw in publisher:
            return True, f"publisher_keyword={kw}"

    # 3. URL exists but doesn't match any academic source or academic venue pattern
    if url:
        host = _domain(url)
        if host:
            return True, f"non_academic_url_without_academic_id={host}"

    return False, ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dedupe", action="store_true", default=True)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    files = sorted(in_dir.glob("*.json"))
    print(f"Scanning {len(files)} files under {in_dir}/\n", flush=True)

    collected: list[dict] = []
    seen: set[str] = set()
    reason_counts = Counter()
    scanned = 0
    per_paper_counts: dict[str, int] = {}

    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception as exc:
            print(f"  ! failed to parse {f.name}: {exc}")
            continue
        citations = data.get("citations") if isinstance(data, dict) else data
        if not isinstance(citations, list):
            continue
        paper_id = f.stem
        kept_here = 0
        for c in citations:
            scanned += 1
            is_na, reason = _is_non_academic(c)
            if not is_na:
                continue
            # Dedupe: URL (if present) or title+authors key
            key = (c.get("url") or "").strip().lower()
            if not key:
                title = (c.get("title") or "").strip().lower()[:100]
                auth = str(c.get("authors") or "")[:50]
                key = f"title:{title}|auth:{auth}"
            if args.dedupe and key in seen:
                continue
            seen.add(key)

            entry = {
                "title":     c.get("title", ""),
                "authors":   c.get("authors", []),
                "venue":     c.get("venue", ""),
                "year":      c.get("year"),
                "url":       c.get("url", ""),
                "doi":       c.get("doi", ""),
                "arxiv_id":  c.get("arxiv_id", ""),
                "raw_text":  (c.get("raw_text") or "")[:500],
                "_source_paper": paper_id,
                "_source_citation_id": c.get("citation_id", ""),
                "_non_academic_reason": reason,
            }
            collected.append(entry)
            reason_counts[reason.split("=")[0]] += 1
            kept_here += 1
        if kept_here:
            per_paper_counts[paper_id] = kept_here

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(collected, indent=2, ensure_ascii=False))

    # Report
    print(f"Scanned citations: {scanned}")
    print(f"Non-academic collected (deduped): {len(collected)}")
    print(f"Saved to: {out}")
    print(f"\nBy reason type:")
    for reason, count in reason_counts.most_common():
        print(f"  {reason:40s} {count}")
    print(f"\nTop 10 papers contributing non-academic citations:")
    for paper, count in sorted(per_paper_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:3d}  {paper}")


if __name__ == "__main__":
    main()
