"""Generate hallucinated citation test data by mutating real citations.

Taxonomy:
  REAL:                R1 (exact), R2 (format variant), R3 (et al.), R4 (non-academic verified)
  POTENTIAL:           P1 (version diff), P2 (author name variant), P3 (unstable source)
  HALLUCINATED:        H1 (fabricated), H2a-c (title), H3a-c (author), H4-H5e (meta), H6 (sibling), H7 (source)

Uses LLM as mutator for realistic hallucinations.
Input: directory of verified citation JSON files.
Output: labeled dataset with original, mutated, label, subtype, and explanation.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class MutationSample:
    original: dict[str, Any]
    mutated: dict[str, Any]
    label: str           # REAL, POTENTIAL_HALLUCINATED, HALLUCINATED
    subtype: str         # e.g. H2a, H3b, R1, P1
    category: str        # e.g. Title, Author, Meta, Full, Source, Format
    mutation_type: str   # e.g. word_substitution, author_reordering
    explanation: str     # What was changed and why
    changed_fields: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

_CLIENT = None
_MODEL_ID = None


def _init_llm(region: str, bearer_token: str, model_id: str) -> None:
    global _CLIENT, _MODEL_ID
    import boto3
    _CLIENT = boto3.Session().client(
        "bedrock-runtime",
        region_name=region,
        endpoint_url=f"https://bedrock-runtime.{region}.amazonaws.com",
        aws_access_key_id=bearer_token,
        aws_secret_access_key="ignored",
        aws_session_token="ignored",
    )
    _MODEL_ID = model_id


def _call_llm(prompt: str, max_tokens: int = 1024) -> str:
    response = _CLIENT.converse(
        modelId=_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"temperature": 0.7, "maxTokens": max_tokens},
    )
    return response["output"]["message"]["content"][0]["text"]


def _call_llm_json(prompt: str, max_tokens: int = 1024) -> dict:
    text = _call_llm(prompt, max_tokens)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"No JSON found in LLM response: {text[:200]}")


# ===========================================================================
# REAL generators
# ===========================================================================

def gen_R1_exact_match(citation: dict, **_) -> MutationSample | None:
    """R1: No mutation, original citation is valid."""
    return MutationSample(
        original=citation,
        mutated=deepcopy(citation),
        label="REAL",
        subtype="R1",
        category="None",
        mutation_type="none",
        explanation="No mutation. All fields match the original valid citation.",
        changed_fields=[],
    )


def gen_R2_format_variant(citation: dict, **_) -> MutationSample | None:
    """R2: Case, venue abbreviation, first-name initial abbreviation."""
    mutated = deepcopy(citation)
    changes = []

    # Title: lowercase
    if mutated.get("title"):
        mutated["title"] = mutated["title"].lower()
        changes.append("title lowercased")

    # Authors: full name → initial (Gao Hao → G. Hao)
    if mutated.get("authors"):
        new_authors = []
        for a in mutated["authors"]:
            parts = a.split()
            if len(parts) >= 2:
                new_authors.append(f"{parts[0][0]}. {' '.join(parts[1:])}")
                changes.append(f"'{a}' → '{new_authors[-1]}'")
            else:
                new_authors.append(a)
        mutated["authors"] = new_authors

    # Venue: abbreviate if long
    venue = mutated.get("venue", "")
    if "Proceedings" in venue:
        mutated["venue"] = venue.replace("Proceedings of the ", "Proc. ")
        changes.append("venue abbreviated")

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="REAL",
        subtype="R2",
        category="Format",
        mutation_type="format_variant",
        explanation=f"Format-only changes: {'; '.join(changes)}. Still the same valid citation.",
        changed_fields=[f for f in ["title", "authors", "venue"] if changes],
    )


def gen_R3_et_al(citation: dict, **_) -> MutationSample | None:
    """R3: Truncate author list with et al."""
    mutated = deepcopy(citation)
    authors = mutated.get("authors", [])
    if len(authors) < 3:
        return None

    n_keep = random.randint(1, min(3, len(authors) - 1))
    kept = authors[:n_keep]
    marker = random.choice(["et al.", "Others", "and others"])
    kept.append(marker)
    mutated["authors"] = kept

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="REAL",
        subtype="R3",
        category="Author",
        mutation_type="et_al_abbreviation",
        explanation=f"Author list truncated from {len(authors)} to {n_keep} + '{marker}'. All listed authors are correct.",
        changed_fields=["authors"],
    )


def gen_R4_non_academic(citation: dict, **_) -> MutationSample | None:
    """R4: Keep non-academic citation as-is (tweet, blog, etc.)."""
    url = citation.get("url", "")
    non_academic = any(d in url for d in [
        "twitter.com", "x.com", "github.com", "medium.com", "substack.com",
        "blog", "news", "huggingface.co",
    ])
    if not non_academic and not (citation.get("venue", "").lower() in ("tweet", "blog", "github")):
        return None

    return MutationSample(
        original=citation,
        mutated=deepcopy(citation),
        label="REAL",
        subtype="R4",
        category="Source",
        mutation_type="non_academic_verified",
        explanation=f"Non-academic source kept as-is. URL: {url}. Verified via web search.",
        changed_fields=[],
    )


# ===========================================================================
# POTENTIAL HALLUCINATED generators
# ===========================================================================

def gen_P1_version_difference(citation: dict, **_) -> MutationSample | None:
    """P1: Simulate arXiv version difference — change title slightly as if from older version."""
    mutated = deepcopy(citation)
    original_title = citation.get("title", "")
    if not original_title or not citation.get("arxiv_id"):
        return None

    prompt = (
        f"Simulate an arXiv version difference: slightly modify this title as if it were "
        f"an earlier draft version (e.g., minor wording change, removed subtitle).\n"
        f"Original (published version): \"{original_title}\"\n"
        f'Return JSON: {{"draft_title": "...", "change": "what changed"}}\n'
        f"Return JSON only."
    )
    try:
        result = _call_llm_json(prompt)
        mutated["title"] = result["draft_title"]
        change = result.get("change", "title slightly modified")
    except Exception:
        # Fallback: remove subtitle after colon
        if ":" in original_title:
            mutated["title"] = original_title.split(":")[0].strip()
            change = "subtitle removed (simulating draft version)"
        else:
            return None

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="POTENTIAL_HALLUCINATED",
        subtype="P1",
        category="Title",
        mutation_type="version_difference",
        explanation=f"Simulated arXiv version difference: {change}. Original: \"{original_title}\". Draft: \"{mutated['title']}\".",
        changed_fields=["title"],
    )


def gen_P2_author_name_variant(citation: dict, **_) -> MutationSample | None:
    """P2: Replace author first name with a plausible nickname/variant (Katherine→Kate)."""
    mutated = deepcopy(citation)
    authors = list(mutated.get("authors", []))
    if not authors:
        return None

    idx = random.randint(0, len(authors) - 1)
    original_name = authors[idx]

    prompt = (
        f"Replace this author's first name with a plausible nickname or shortened variant "
        f"that the same person might use (e.g., Katherine→Kate, Michael→Mike, Robert→Bob, "
        f"William→Will, Richard→Dick, Elizabeth→Liz, 张伟→Wei Zhang vs William Zhang).\n"
        f"Original: \"{original_name}\"\n"
        f'Return JSON: {{"variant_name": "...", "explanation": "what changed"}}\n'
        f"Return JSON only."
    )
    try:
        result = _call_llm_json(prompt)
        authors[idx] = result["variant_name"]
        change = result.get("explanation", "name variant applied")
    except Exception:
        # Fallback: simple nickname
        parts = original_name.split()
        if len(parts) >= 2 and len(parts[0]) > 4:
            authors[idx] = f"{parts[0][:3]}. {' '.join(parts[1:])}"
            change = f"first name shortened: {original_name} → {authors[idx]}"
        else:
            return None

    mutated["authors"] = authors
    return MutationSample(
        original=citation,
        mutated=mutated,
        label="POTENTIAL_HALLUCINATED",
        subtype="P2",
        category="Author",
        mutation_type="author_name_variant",
        explanation=f"Author name variant: \"{original_name}\" → \"{authors[idx]}\". {change}",
        changed_fields=["authors"],
    )


def gen_P3_unstable_source(citation: dict, **_) -> MutationSample | None:
    """P3: Simulate a source that has been deleted/moved — change URL to broken version."""
    mutated = deepcopy(citation)
    url = citation.get("url", "")
    if not url:
        return None

    # Simulate URL change (moved/deleted)
    if "twitter.com" in url or "x.com" in url:
        mutated["url"] = url.replace("x.com", "twitter.com") + "?deleted=true"
        explanation = "Tweet URL simulated as deleted/moved."
    elif "github.com" in url:
        mutated["url"] = url.rstrip("/") + "/tree/archived-2023"
        explanation = "GitHub repo simulated as archived."
    else:
        mutated["url"] = url.rstrip("/") + "/page-moved"
        explanation = "URL simulated as moved to different location."

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="POTENTIAL_HALLUCINATED",
        subtype="P3",
        category="Source",
        mutation_type="unstable_source",
        explanation=f"{explanation} Original URL: {url}. Indirect evidence (reposts, caches) may confirm existence.",
        changed_fields=["url"],
    )


# ===========================================================================
# HALLUCINATED generators — Title
# ===========================================================================

def gen_H2a_word_substitution(citation: dict, **_) -> MutationSample | None:
    """H2a: Replace 1-2 key words with synonyms."""
    mutated = deepcopy(citation)
    original_title = citation.get("title", "")
    if not original_title:
        return None

    prompt = (
        f"Replace 1-2 key words in this paper title with synonyms or related terms. "
        f"The result should look plausible but be detectably different.\n"
        f"Original: \"{original_title}\"\n"
        f'Return JSON: {{"mutated_title": "...", "changed_words": "word1→word2, ..."}}\n'
        f"Return JSON only."
    )
    try:
        result = _call_llm_json(prompt)
        mutated["title"] = result["mutated_title"]
        changed = result.get("changed_words", "")
    except Exception:
        words = original_title.split()
        if len(words) > 3:
            idx = random.randint(1, len(words) - 2)
            old_word = words[idx]
            words[idx] = "novel"
            mutated["title"] = " ".join(words)
            changed = f"{old_word}→novel"
        else:
            return None

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H2a",
        category="Title",
        mutation_type="word_substitution",
        explanation=f"Word substitution: {changed}. Original: \"{original_title}\".",
        changed_fields=["title"],
    )


def gen_H2b_title_paraphrase(citation: dict, **_) -> MutationSample | None:
    """H2b: Rephrase with different wording."""
    mutated = deepcopy(citation)
    original_title = citation.get("title", "")
    if not original_title:
        return None

    prompt = (
        f"Paraphrase this paper title using different words but keeping similar meaning. "
        f"It should be clearly a different title string.\n"
        f"Original: \"{original_title}\"\n"
        f'Return JSON: {{"mutated_title": "..."}}\n'
        f"Return JSON only."
    )
    try:
        result = _call_llm_json(prompt)
        mutated["title"] = result["mutated_title"]
    except Exception:
        return None

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H2b",
        category="Title",
        mutation_type="title_paraphrase",
        explanation=f"Title paraphrased. Original: \"{original_title}\". Mutated: \"{mutated['title']}\".",
        changed_fields=["title"],
    )


def gen_H2c_title_fabrication(citation: dict, **_) -> MutationSample | None:
    """H2c: Semantically different, loosely related title."""
    mutated = deepcopy(citation)
    original_title = citation.get("title", "")
    if not original_title:
        return None

    prompt = (
        f"Generate a completely different paper title that is loosely related to the same topic "
        f"but describes a different research contribution.\n"
        f"Original: \"{original_title}\"\n"
        f'Return JSON: {{"mutated_title": "..."}}\n'
        f"Return JSON only."
    )
    try:
        result = _call_llm_json(prompt)
        mutated["title"] = result["mutated_title"]
    except Exception:
        return None

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H2c",
        category="Title",
        mutation_type="title_fabrication",
        explanation=f"Title fabricated. Original: \"{original_title}\". Fabricated: \"{mutated['title']}\".",
        changed_fields=["title"],
    )


# ===========================================================================
# HALLUCINATED generators — Author
# ===========================================================================

def gen_H3a_author_addition_deletion(citation: dict, **_) -> MutationSample | None:
    """H3a: Add or remove authors (not explainable by version history)."""
    mutated = deepcopy(citation)
    authors = list(mutated.get("authors", []))
    if not authors:
        return None

    action = random.choice(["add", "delete"]) if len(authors) >= 2 else "add"

    if action == "add":
        prompt = (
            f"Generate a plausible but fake author name for a paper in this field.\n"
            f"Existing authors: {authors}\n"
            f'Return JSON: {{"fake_author": "First Last"}}\n'
            f"Return JSON only."
        )
        try:
            result = _call_llm_json(prompt)
            fake = result["fake_author"]
        except Exception:
            fake = "Alexander J. Thompson"
        pos = random.randint(0, len(authors))
        authors.insert(pos, fake)
        explanation = f"Added fake author '{fake}' at position {pos}. Original had {len(citation['authors'])} authors."
    else:
        removed_idx = random.randint(0, len(authors) - 1)
        removed = authors.pop(removed_idx)
        explanation = f"Removed real author '{removed}' from position {removed_idx}. Original had {len(citation['authors'])} authors."

    mutated["authors"] = authors
    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H3a",
        category="Author",
        mutation_type="author_addition_deletion",
        explanation=explanation,
        changed_fields=["authors"],
    )


def gen_H3b_author_reordering(citation: dict, **_) -> MutationSample | None:
    """H3b: Change author order from published version."""
    mutated = deepcopy(citation)
    authors = list(mutated.get("authors", []))
    if len(authors) < 2:
        return None

    original_order = list(authors)
    # Shuffle until different from original
    for _ in range(10):
        random.shuffle(authors)
        if authors != original_order:
            break
    else:
        # Simple swap of first two
        authors[0], authors[1] = authors[1], authors[0]

    mutated["authors"] = authors
    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H3b",
        category="Author",
        mutation_type="author_reordering",
        explanation=f"Author order changed. Original: {original_order}. Reordered: {authors}.",
        changed_fields=["authors"],
    )


def gen_H3c_author_fabrication(citation: dict, **_) -> MutationSample | None:
    """H3c: Replace entire author list with fake names."""
    mutated = deepcopy(citation)
    authors = mutated.get("authors", [])
    if not authors:
        return None

    n = len(authors)
    prompt = (
        f"Generate {n} completely fake but plausible-sounding author names for an academic paper. "
        f"Mix different ethnic backgrounds.\n"
        f'Return JSON: {{"authors": ["Name1", "Name2", ...]}}\n'
        f"Return JSON only."
    )
    try:
        result = _call_llm_json(prompt)
        mutated["authors"] = result["authors"][:n]
    except Exception:
        mutated["authors"] = [f"Author {chr(65 + i)}" for i in range(n)]

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H3c",
        category="Author",
        mutation_type="author_fabrication",
        explanation=f"All {n} authors replaced with fabricated names. Original: {authors}.",
        changed_fields=["authors"],
    )


# ===========================================================================
# HALLUCINATED generators — Full / Meta
# ===========================================================================

def gen_H1_completely_fabricated(citation: dict, **_) -> MutationSample | None:
    """H1: Fabricate everything."""
    prompt = (
        f"Generate a completely fake but plausible-looking academic citation in a field related to: "
        f"\"{citation.get('title', '')[:50]}\". All metadata should be invented.\n"
        f'Return JSON: {{"title": "...", "authors": ["..."], "venue": "...", "year": YYYY, "doi": "10.xxxx/..."}}\n'
        f"Return JSON only."
    )
    try:
        result = _call_llm_json(prompt)
        mutated = deepcopy(citation)
        mutated["title"] = result.get("title", "A Novel Approach")
        mutated["authors"] = result.get("authors", ["J. Doe"])
        mutated["venue"] = result.get("venue", "NeurIPS")
        mutated["year"] = result.get("year", 2023)
        mutated["doi"] = result.get("doi", "")
        mutated["arxiv_id"] = ""
        mutated["url"] = ""
        mutated["pages"] = ""
        mutated["volume"] = ""
        mutated["publisher"] = ""
        mutated["location"] = ""
    except Exception:
        return None

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H1",
        category="Full",
        mutation_type="completely_fabricated",
        explanation="All fields fabricated by LLM. No connection to any real paper.",
        changed_fields=["title", "authors", "venue", "year", "doi"],
    )


def gen_H4_venue_year_fabrication(citation: dict, **_) -> MutationSample | None:
    """H4: Change venue AND year to wrong values."""
    mutated = deepcopy(citation)
    original_venue = citation.get("venue", "")
    original_year = citation.get("year")

    venues = ["NeurIPS", "ICML", "ICLR", "CVPR", "ACL", "EMNLP", "AAAI", "Nature", "Science", "SIGIR", "KDD"]
    new_venue = random.choice([v for v in venues if v.lower() not in (original_venue or "").lower()] or venues)
    year_offset = random.choice([-2, -1, 1, 2])
    new_year = (original_year or 2023) + year_offset

    mutated["venue"] = new_venue
    mutated["year"] = new_year

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H4",
        category="Meta",
        mutation_type="venue_year_fabrication",
        explanation=f"Venue: \"{original_venue}\" → \"{new_venue}\". Year: {original_year} → {new_year}.",
        changed_fields=["venue", "year"],
    )


def gen_H5a_doi_fabrication(citation: dict, **_) -> MutationSample | None:
    """H5a: DOI resolves to a different paper."""
    mutated = deepcopy(citation)
    fake_doi = f"10.{random.randint(1000, 9999)}/{random.randint(100000, 999999)}"
    original_doi = citation.get("doi", "")
    mutated["doi"] = fake_doi

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H5a",
        category="Meta",
        mutation_type="doi_fabrication",
        explanation=f"DOI fabricated: \"{original_doi}\" → \"{fake_doi}\". Will resolve to wrong paper or 404.",
        changed_fields=["doi"],
    )


def gen_H5b_doi_nonexistent(citation: dict, **_) -> MutationSample | None:
    """H5b: DOI that doesn't exist at all."""
    mutated = deepcopy(citation)
    original_doi = citation.get("doi", "")
    mutated["doi"] = "10.99999/nonexistent.00000"

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H5b",
        category="Meta",
        mutation_type="doi_nonexistent",
        explanation=f"DOI replaced with nonexistent value. Original: \"{original_doi}\".",
        changed_fields=["doi"],
    )


def gen_H5c_date_error(citation: dict, **_) -> MutationSample | None:
    """H5c: Year is verifiably wrong."""
    mutated = deepcopy(citation)
    original_year = citation.get("year")
    if not original_year:
        return None

    offset = random.choice([-3, -2, 2, 3])
    mutated["year"] = original_year + offset

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H5c",
        category="Meta",
        mutation_type="date_error",
        explanation=f"Year: {original_year} → {mutated['year']}.",
        changed_fields=["year"],
    )


def gen_H5d_venue_mismatch(citation: dict, **_) -> MutationSample | None:
    """H5d: Paper exists but venue is wrong."""
    mutated = deepcopy(citation)
    original_venue = citation.get("venue", "")

    venues = ["NeurIPS", "ICML", "ICLR", "CVPR", "ACL", "EMNLP", "AAAI", "SIGIR", "KDD", "WWW"]
    new_venue = random.choice([v for v in venues if v.lower() not in (original_venue or "").lower()] or venues)
    mutated["venue"] = new_venue

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H5d",
        category="Meta",
        mutation_type="venue_mismatch",
        explanation=f"Venue: \"{original_venue}\" → \"{new_venue}\". Paper was never published at {new_venue}.",
        changed_fields=["venue"],
    )


def gen_H5e_pages_fabrication(citation: dict, **_) -> MutationSample | None:
    """H5e: Pages/volume is verifiably wrong."""
    mutated = deepcopy(citation)
    original_pages = citation.get("pages", "")
    original_volume = citation.get("volume", "")

    fake_start = random.randint(100, 9000)
    fake_pages = f"{fake_start}-{fake_start + random.randint(5, 20)}"
    fake_volume = str(random.randint(1, 200))

    mutated["pages"] = fake_pages
    mutated["volume"] = fake_volume

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H5e",
        category="Meta",
        mutation_type="pages_volume_fabrication",
        explanation=f"Pages: \"{original_pages}\" → \"{fake_pages}\". Volume: \"{original_volume}\" → \"{fake_volume}\".",
        changed_fields=["pages", "volume"],
    )


def gen_H6_sibling_confusion(citation: dict, all_citations: list[dict] | None = None, **_) -> MutationSample | None:
    """H6: Mix metadata from two different papers."""
    if not all_citations:
        return None
    mutated = deepcopy(citation)
    candidates = [
        c for c in all_citations
        if c.get("title", "") != citation.get("title", "")
        and c.get("title", "").strip()
        and c.get("venue", "").strip()
    ]
    if not candidates:
        return None

    sibling = random.choice(candidates)
    mutated["venue"] = sibling.get("venue", mutated.get("venue", ""))
    mutated["year"] = sibling.get("year", mutated.get("year"))

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H6",
        category="Meta",
        mutation_type="sibling_paper_confusion",
        explanation=(
            f"Venue/year taken from \"{sibling.get('title', '')[:50]}\". "
            f"Venue: \"{citation.get('venue', '')}\" → \"{mutated['venue']}\". "
            f"Year: {citation.get('year')} → {mutated['year']}."
        ),
        changed_fields=["venue", "year"],
    )


def gen_H7_nonexistent_source(citation: dict, **_) -> MutationSample | None:
    """H7: Non-academic URL that doesn't exist."""
    mutated = deepcopy(citation)
    fake_slug = re.sub(r"[^a-z0-9]+", "-", citation.get("title", "paper")[:30].lower()).strip("-")
    mutated["url"] = f"https://example.com/blog/{fake_slug}-{random.randint(1000, 9999)}"
    mutated["venue"] = "Personal Blog"
    mutated["doi"] = ""
    mutated["arxiv_id"] = ""

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H7",
        category="Source",
        mutation_type="nonexistent_source",
        explanation=f"URL fabricated: \"{mutated['url']}\". Original: \"{citation.get('url', '')}\".",
        changed_fields=["url", "venue", "doi", "arxiv_id"],
    )


# ===========================================================================
# Registry & Orchestrator
# ===========================================================================

ALL_GENERATORS = {
    # REAL
    "R1": gen_R1_exact_match,
    "R2": gen_R2_format_variant,
    "R3": gen_R3_et_al,
    "R4": gen_R4_non_academic,
    # POTENTIAL
    "P1": gen_P1_version_difference,
    "P2": gen_P2_author_name_variant,
    "P3": gen_P3_unstable_source,
    # HALLUCINATED — Full
    "H1": gen_H1_completely_fabricated,
    # HALLUCINATED — Title
    "H2a": gen_H2a_word_substitution,
    "H2b": gen_H2b_title_paraphrase,
    "H2c": gen_H2c_title_fabrication,
    # HALLUCINATED — Author
    "H3a": gen_H3a_author_addition_deletion,
    "H3b": gen_H3b_author_reordering,
    "H3c": gen_H3c_author_fabrication,
    # HALLUCINATED — Meta
    "H4": gen_H4_venue_year_fabrication,
    "H5a": gen_H5a_doi_fabrication,
    "H5b": gen_H5b_doi_nonexistent,
    "H5c": gen_H5c_date_error,
    "H5d": gen_H5d_venue_mismatch,
    "H5e": gen_H5e_pages_fabrication,
    # HALLUCINATED — Source
    "H7": gen_H7_nonexistent_source,
}

SIBLING_GENERATORS = {"H6": gen_H6_sibling_confusion}


def generate_mutations(
    citations: list[dict],
    subtypes: list[str] | None = None,
    samples_per_type: int = 1,
) -> list[dict]:
    subtypes = subtypes or list(ALL_GENERATORS.keys()) + list(SIBLING_GENERATORS.keys())
    results: list[dict] = []

    for subtype in subtypes:
        gen = ALL_GENERATORS.get(subtype) or SIBLING_GENERATORS.get(subtype)
        if not gen:
            print(f"  Unknown subtype: {subtype}", flush=True)
            continue

        count = 0
        attempts = 0
        while count < samples_per_type and attempts < samples_per_type * 3:
            attempts += 1
            citation = random.choice(citations)
            if not citation.get("title") or not citation.get("authors"):
                continue
            try:
                if subtype in SIBLING_GENERATORS:
                    sample = gen(citation, all_citations=citations)
                else:
                    sample = gen(citation)
            except Exception as e:
                print(f"  {subtype} error: {e}", flush=True)
                sample = None

            if sample is None:
                continue

            results.append({
                "source_paper": citation.get("_source_paper", ""),
                "original_citation_id": citation.get("citation_id", ""),
                "label": sample.label,
                "subtype": sample.subtype,
                "category": sample.category,
                "mutation_type": sample.mutation_type,
                "explanation": sample.explanation,
                "changed_fields": sample.changed_fields,
                "original": {k: v for k, v in sample.original.items() if k != "_source_paper"},
                "mutated": {k: v for k, v in sample.mutated.items() if k != "_source_paper"},
            })
            count += 1

        print(f"  {subtype:5s} ({gen.__doc__.split(':')[0].strip() if gen.__doc__ else subtype}): {count} samples", flush=True)

    return results


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate hallucinated citation test data.")
    parser.add_argument("--input-dir", required=True, help="Directory with verified citation JSONs.")
    parser.add_argument("--output", default="artifacts/hallucination_test_data.json", help="Output JSON file.")
    parser.add_argument("--samples-per-type", type=int, default=3, help="Samples per mutation subtype.")
    parser.add_argument("--max-papers", type=int, default=0, help="Max papers to use. 0 = all.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load config for Bedrock
    from apps.pdf_checker.config import load_pdf_checker_config
    cfg = load_pdf_checker_config()
    bcfg = cfg.verification_llm.bedrock
    bearer_token = bcfg.bearer_token or cfg.entry_extraction.bedrock.bearer_token
    _init_llm(region=bcfg.region, bearer_token=bearer_token, model_id=str(bcfg.model_id))
    print(f"LLM: model_id={bcfg.model_id} region={bcfg.region}", flush=True)

    # Load citations
    input_dir = Path(args.input_dir)
    all_citations: list[dict] = []
    files = sorted(input_dir.glob("*.json"))
    if args.max_papers > 0:
        files = files[:args.max_papers]

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        for c in data.get("citations", []):
            if c.get("title") and c.get("authors"):
                c["_source_paper"] = fpath.stem
                all_citations.append(c)

    print(f"Loaded {len(all_citations)} citations from {len(files)} papers\n", flush=True)

    results = generate_mutations(all_citations, samples_per_type=args.samples_per_type)

    # Summary
    from collections import Counter
    label_counts = Counter(r["label"] for r in results)
    subtype_counts = Counter(r["subtype"] for r in results)

    print(f"\n{'='*50}")
    print(f"Total: {len(results)} samples")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print()
    for subtype, count in sorted(subtype_counts.items()):
        r = next(r for r in results if r["subtype"] == subtype)
        print(f"  {subtype:5s} [{r['label']:25s}] {r['category']:8s} {count} samples")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
