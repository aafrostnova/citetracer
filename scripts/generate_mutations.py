"""Generate hallucinated citation test data by mutating real citations.

Taxonomy (13 codes — see docs/taxonomy.md):
  REAL:         R1 (exact match), R2 (format variant), R3 (et al. abbreviation)
  POTENTIAL:    P1 (author name variant), P2 (non-academic source),
                P3 (cross-candidate coverage), P4 (insufficient field evidence)
  HALLUCINATED: H1 (title error), H2 (author error), H3 (venue error),
                H4 (year error), H5 (DOI/identifier error),
                H6 (pages/volume/publisher/location error)

Uses LLM as mutator for realistic hallucinations.
Input: directory of verified citation JSON files, or a BibTeX file.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for split_mutations_by_code import


@dataclass
class MutationSample:
    original: dict[str, Any]
    mutated: dict[str, Any]
    label: str           # REAL, POTENTIAL_HALLUCINATED, HALLUCINATED
    subtype: str         # e.g. H2, H3, R1, P1
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
    import os
    import boto3
    # Bedrock API key auth: pass via AWS_BEARER_TOKEN_BEDROCK env var
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = bearer_token
    _CLIENT = boto3.client("bedrock-runtime", region_name=region)
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


# ===========================================================================
# POTENTIAL HALLUCINATED generators
# ===========================================================================

def gen_P1_author_name_variant(citation: dict, **_) -> MutationSample | None:
    """P1: Replace author first name with a plausible nickname/variant."""
    mutated = deepcopy(citation)
    authors = list(mutated.get("authors", []))
    if not authors:
        return None

    # Only pick authors that have both first and last name — P1 requires
    # a name STRUCTURE preserved (variant of first name; last name unchanged).
    eligible = [i for i, a in enumerate(authors) if len(a.split()) >= 2]
    if not eligible:
        return None
    idx = random.choice(eligible)
    original_name = authors[idx]
    original_parts = original_name.split()
    original_last = original_parts[-1]

    prompt = (
        f"Replace ONLY the FIRST NAME with a plausible nickname or shortened variant "
        f"that the same person might use. KEEP THE LAST NAME UNCHANGED.\n"
        f"Examples: Katherine Smith→Kate Smith, Michael Jones→Mike Jones, "
        f"Alexander Panchenko→Alex Panchenko, Ziming Li→Zim Li.\n"
        f"Original: \"{original_name}\" (first name: \"{original_parts[0]}\", "
        f"last name: \"{original_last}\")\n"
        f'Return JSON: {{"variant_name": "<new first name> {original_last}", "explanation": "..."}}\n'
        f"Return JSON only."
    )

    def _fallback() -> str | None:
        """Deterministic multi-letter truncation — preserves last name."""
        if len(original_parts) >= 2 and len(original_parts[0]) > 4:
            return f"{original_parts[0][:3]}. {' '.join(original_parts[1:])}"
        return None

    new_name: str | None = None
    change = ""
    try:
        result = _call_llm_json(prompt)
        candidate = str(result.get("variant_name", "")).strip()
        # Validate: must contain the original last name, have >=2 tokens,
        # AND differ from the original (no-op mutations are rejected)
        if (candidate
                and original_last.lower() in candidate.lower()
                and len(candidate.split()) >= 2
                and candidate.strip().lower() != original_name.strip().lower()):
            new_name = candidate
            change = result.get("explanation", "name variant applied")
        else:
            # LLM dropped last name / returned identical / malformed → fallback
            new_name = _fallback()
            change = "LLM output rejected, fell back to truncation"
    except Exception:
        new_name = _fallback()
        change = f"first name shortened: {original_name} → {new_name}" if new_name else ""

    if not new_name or new_name.strip().lower() == original_name.strip().lower():
        return None

    authors[idx] = new_name
    mutated["authors"] = authors
    return MutationSample(
        original=citation,
        mutated=mutated,
        label="POTENTIAL_HALLUCINATED",
        subtype="P1",
        category="Author",
        mutation_type="author_name_variant",
        explanation=f"Author name variant: \"{original_name}\" → \"{new_name}\". {change}",
        changed_fields=["authors"],
    )


def gen_P2_non_academic_source(citation: dict, **_) -> MutationSample | None:
    """P2: Simulate a non-academic source that cannot be fully verified."""
    mutated = deepcopy(citation)
    url = citation.get("url", "")

    if url:
        # Has URL: simulate it being deleted/moved
        if "twitter.com" in url or "x.com" in url:
            mutated["url"] = url.replace("x.com", "twitter.com") + "?deleted=true"
            explanation = "Tweet URL simulated as deleted/moved."
        elif "github.com" in url:
            mutated["url"] = url.rstrip("/") + "/tree/archived-2023"
            explanation = "GitHub repo simulated as archived."
        else:
            mutated["url"] = url.rstrip("/") + "/page-moved"
            explanation = "URL simulated as moved to different location."
    else:
        # No URL: convert to non-academic source with fabricated URL
        fake_slug = re.sub(r"[^a-z0-9]+", "-", citation.get("title", "paper")[:30].lower()).strip("-")
        mutated["url"] = f"https://example.com/blog/{fake_slug}-{random.randint(1000, 9999)}"
        mutated["venue"] = "Personal Blog"
        mutated["doi"] = ""
        mutated["arxiv_id"] = ""
        explanation = f"Converted to non-academic source with fabricated URL: {mutated['url']}."

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="POTENTIAL_HALLUCINATED",
        subtype="P2",
        category="Source",
        mutation_type="non_academic_source",
        explanation=f"{explanation} Original URL: {url or '(none)'}. Source cannot be fully verified via academic databases.",
        changed_fields=[f for f in ["url", "venue", "doi", "arxiv_id"] if mutated.get(f) != citation.get(f)],
    )


def gen_P3_cross_candidate_coverage(citation: dict, **_) -> MutationSample | None:
    """P3: Apply two minor format mutations simultaneously so no single connector
    matches all fields, but every field is covered by some connector."""
    mutated = deepcopy(citation)
    changes = []

    # Mutation A: author first names → single-letter initials (R2-style)
    authors = list(mutated.get("authors", []))
    if not authors:
        return None
    new_authors = []
    for a in authors:
        parts = a.split()
        if len(parts) >= 2:
            new_authors.append(f"{parts[0][0]}. {' '.join(parts[1:])}")
        else:
            new_authors.append(a)
    if new_authors == authors:
        return None
    mutated["authors"] = new_authors
    changes.append("authors → initials")

    # Mutation B: venue acronym ↔ full name
    venue = mutated.get("venue", "")
    venue_swap = {
        "NeurIPS": "Advances in Neural Information Processing Systems",
        "ICML": "International Conference on Machine Learning",
        "ICLR": "International Conference on Learning Representations",
        "ACL": "Association for Computational Linguistics",
        "EMNLP": "Empirical Methods in Natural Language Processing",
        "AAAI": "AAAI Conference on Artificial Intelligence",
        "CVPR": "IEEE Conference on Computer Vision and Pattern Recognition",
    }
    if venue in venue_swap:
        mutated["venue"] = venue_swap[venue]
        changes.append(f"venue '{venue}' → full name")
    elif venue:
        # Reverse direction: try contracting
        for short, full in venue_swap.items():
            if full.lower() in venue.lower():
                mutated["venue"] = short
                changes.append(f"venue full name → '{short}'")
                break
        else:
            return None
    else:
        return None

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="POTENTIAL_HALLUCINATED",
        subtype="P3",
        category="Meta",
        mutation_type="cross_candidate_coverage",
        explanation=(
            f"Distributed minor mutations: {'; '.join(changes)}. "
            f"No single connector will match all fields, but each field is recoverable across connectors."
        ),
        changed_fields=["authors", "venue"],
    )


def gen_P4_insufficient_field_evidence(citation: dict, **_) -> MutationSample | None:
    """P4: Add plausible but unverifiable volume/pages/publisher/location to a real
    conference paper. Core fields stay correct; optional fields cannot be verified
    because no connector supplies them for that conference."""
    mutated = deepcopy(citation)
    venue = (citation.get("venue") or "").lower()

    # Only apply to conferences where DBLP/CrossRef typically lack volume/pages
    target_venues = ("neurips", "icml", "iclr", "aaai", "naacl", "acl", "emnlp",
                     "advances in neural information processing systems",
                     "international conference on machine learning",
                     "international conference on learning representations")
    if not any(v in venue for v in target_venues):
        return None

    changes = []
    if not mutated.get("volume"):
        mutated["volume"] = str(random.randint(30, 40))
        changes.append("volume")
    if not mutated.get("pages"):
        start = random.randint(1000, 30000)
        mutated["pages"] = f"{start}-{start + random.randint(8, 20)}"
        changes.append("pages")
    if not mutated.get("publisher"):
        mutated["publisher"] = random.choice(["PMLR", "OpenReview", "Curran Associates"])
        changes.append("publisher")
    if not mutated.get("location"):
        mutated["location"] = random.choice([
            "New Orleans, LA, USA", "Vancouver, Canada", "Vienna, Austria",
            "Sydney, Australia", "Singapore",
        ])
        changes.append("location")

    if not changes:
        return None

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="POTENTIAL_HALLUCINATED",
        subtype="P4",
        category="Meta",
        mutation_type="insufficient_field_evidence",
        explanation=(
            f"Added plausible {', '.join(changes)} to a {citation.get('venue')} paper. "
            f"Core fields (title/authors/venue/year) remain correct, but the added fields "
            f"cannot be verified because DBLP/CrossRef do not supply them."
        ),
        changed_fields=changes,
    )


# ===========================================================================
# HALLUCINATED generators — Title (H1)
# ===========================================================================

def gen_H1_word_substitution(citation: dict, **_) -> MutationSample | None:
    """H1: Replace 1-2 key words with synonyms."""
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
        subtype="H1",
        category="Title",
        mutation_type="word_substitution",
        explanation=f"Word substitution: {changed}. Original: \"{original_title}\".",
        changed_fields=["title"],
    )


def gen_H1_title_paraphrase(citation: dict, **_) -> MutationSample | None:
    """H1: Rephrase with different wording."""
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
        subtype="H1",
        category="Title",
        mutation_type="title_paraphrase",
        explanation=f"Title paraphrased. Original: \"{original_title}\". Mutated: \"{mutated['title']}\".",
        changed_fields=["title"],
    )


def gen_H1_title_fabrication(citation: dict, **_) -> MutationSample | None:
    """H1: Semantically different, loosely related title."""
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
        subtype="H1",
        category="Title",
        mutation_type="title_fabrication",
        explanation=f"Title fabricated. Original: \"{original_title}\". Fabricated: \"{mutated['title']}\".",
        changed_fields=["title"],
    )


# ===========================================================================
# HALLUCINATED generators — Author (H2)
# ===========================================================================

def gen_H2_author_addition_deletion(citation: dict, **_) -> MutationSample | None:
    """H2: Add or remove authors."""
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
        subtype="H2",
        category="Author",
        mutation_type="author_addition_deletion",
        explanation=explanation,
        changed_fields=["authors"],
    )


def gen_H2_author_reordering(citation: dict, **_) -> MutationSample | None:
    """H2: Change author order from published version."""
    mutated = deepcopy(citation)
    authors = list(mutated.get("authors", []))
    if len(authors) < 2:
        return None

    original_order = list(authors)
    for _ in range(10):
        random.shuffle(authors)
        if authors != original_order:
            break
    else:
        authors[0], authors[1] = authors[1], authors[0]

    mutated["authors"] = authors
    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H2",
        category="Author",
        mutation_type="author_reordering",
        explanation=f"Author order changed. Original: {original_order}. Reordered: {authors}.",
        changed_fields=["authors"],
    )


def gen_H2_author_fabrication(citation: dict, **_) -> MutationSample | None:
    """H2: Replace entire author list with fake names."""
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
        subtype="H2",
        category="Author",
        mutation_type="author_fabrication",
        explanation=f"All {n} authors replaced with fabricated names. Original: {authors}.",
        changed_fields=["authors"],
    )


# ===========================================================================
# HALLUCINATED generators — Venue (H3)
# ===========================================================================

def gen_H3_venue_fabrication(citation: dict, **_) -> MutationSample | None:
    """H3: Change venue to wrong value (year unchanged)."""
    mutated = deepcopy(citation)
    original_venue = citation.get("venue", "")

    venues = ["NeurIPS", "ICML", "ICLR", "CVPR", "ACL", "EMNLP", "AAAI", "Nature", "Science", "SIGIR", "KDD", "WWW"]
    new_venue = random.choice([v for v in venues if v.lower() not in (original_venue or "").lower()] or venues)
    mutated["venue"] = new_venue

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H3",
        category="Meta",
        mutation_type="venue_fabrication",
        explanation=f"Venue: \"{original_venue}\" → \"{new_venue}\". Paper was never published at {new_venue}.",
        changed_fields=["venue"],
    )


def gen_H3_venue_and_year(citation: dict, **_) -> MutationSample | None:
    """H3: Change venue AND year to wrong values."""
    mutated = deepcopy(citation)
    original_venue = citation.get("venue", "")
    original_year = citation.get("year")

    venues = ["NeurIPS", "ICML", "ICLR", "CVPR", "ACL", "EMNLP", "AAAI", "Nature", "Science"]
    new_venue = random.choice([v for v in venues if v.lower() not in (original_venue or "").lower()] or venues)
    year_offset = random.choice([-2, -1, 1, 2])
    new_year = (original_year or 2023) + year_offset

    mutated["venue"] = new_venue
    mutated["year"] = new_year

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H3",
        category="Meta",
        mutation_type="venue_year_fabrication",
        explanation=f"Venue: \"{original_venue}\" → \"{new_venue}\". Year: {original_year} → {new_year}.",
        changed_fields=["venue", "year"],
    )


# ===========================================================================
# HALLUCINATED generators — Year (H4)
# ===========================================================================

def gen_H4_date_error(citation: dict, **_) -> MutationSample | None:
    """H4: Year is verifiably wrong."""
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
        subtype="H4",
        category="Meta",
        mutation_type="date_error",
        explanation=f"Year: {original_year} → {mutated['year']}.",
        changed_fields=["year"],
    )


# ===========================================================================
# HALLUCINATED generators — DOI/Identifier (H5)
# ===========================================================================

def gen_H5_doi_fabrication(citation: dict, **_) -> MutationSample | None:
    """H5: DOI resolves to a different paper."""
    mutated = deepcopy(citation)
    fake_doi = f"10.{random.randint(1000, 9999)}/{random.randint(100000, 999999)}"
    original_doi = citation.get("doi", "")
    mutated["doi"] = fake_doi

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H5",
        category="Meta",
        mutation_type="doi_fabrication",
        explanation=f"DOI fabricated: \"{original_doi}\" → \"{fake_doi}\". Will resolve to wrong paper or 404.",
        changed_fields=["doi"],
    )


def gen_H5_doi_nonexistent(citation: dict, **_) -> MutationSample | None:
    """H5: DOI that doesn't exist at all."""
    mutated = deepcopy(citation)
    original_doi = citation.get("doi", "")
    mutated["doi"] = "10.99999/nonexistent.00000"

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H5",
        category="Meta",
        mutation_type="doi_nonexistent",
        explanation=f"DOI replaced with nonexistent value. Original: \"{original_doi}\".",
        changed_fields=["doi"],
    )


# ===========================================================================
# HALLUCINATED generators — Pages/Volume/Publisher/Location (H6)
# ===========================================================================

def gen_H6_pages_fabrication(citation: dict, **_) -> MutationSample | None:
    """H6: Pages/volume verifiably wrong (candidate has different value)."""
    mutated = deepcopy(citation)
    original_pages = citation.get("pages", "")
    original_volume = citation.get("volume", "")
    if not original_pages and not original_volume:
        return None

    fake_start = random.randint(100, 9000)
    fake_pages = f"{fake_start}-{fake_start + random.randint(5, 20)}"
    fake_volume = str(random.randint(1, 200))

    mutated["pages"] = fake_pages
    mutated["volume"] = fake_volume

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H6",
        category="Meta",
        mutation_type="pages_volume_fabrication",
        explanation=f"Pages: \"{original_pages}\" → \"{fake_pages}\". Volume: \"{original_volume}\" → \"{fake_volume}\".",
        changed_fields=["pages", "volume"],
    )


def gen_H6_publisher_fabrication(citation: dict, **_) -> MutationSample | None:
    """H6: Publisher is verifiably wrong."""
    mutated = deepcopy(citation)
    original_publisher = citation.get("publisher", "")
    if not original_publisher:
        return None

    candidates = ["Elsevier", "Springer", "MIT Press", "Wiley", "ACM", "IEEE", "Oxford University Press"]
    new_pub = random.choice([p for p in candidates if p.lower() not in original_publisher.lower()])
    mutated["publisher"] = new_pub

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H6",
        category="Meta",
        mutation_type="publisher_fabrication",
        explanation=f"Publisher: \"{original_publisher}\" → \"{new_pub}\".",
        changed_fields=["publisher"],
    )


def gen_H6_location_fabrication(citation: dict, **_) -> MutationSample | None:
    """H6: Location is verifiably wrong."""
    mutated = deepcopy(citation)
    original_location = citation.get("location", "")
    if not original_location:
        return None

    candidates = [
        "New York, NY, USA", "London, UK", "Tokyo, Japan", "Berlin, Germany",
        "Sydney, Australia", "Toronto, Canada", "Singapore", "Paris, France",
    ]
    new_loc = random.choice([c for c in candidates if c.lower() not in original_location.lower()])
    mutated["location"] = new_loc

    return MutationSample(
        original=citation,
        mutated=mutated,
        label="HALLUCINATED",
        subtype="H6",
        category="Meta",
        mutation_type="location_fabrication",
        explanation=f"Location: \"{original_location}\" → \"{new_loc}\".",
        changed_fields=["location"],
    )


# ===========================================================================
# Registry & Orchestrator
# ===========================================================================

ALL_GENERATORS = {
    # REAL
    "R1": gen_R1_exact_match,
    "R2": gen_R2_format_variant,
    "R3": gen_R3_et_al,
    # POTENTIAL
    "P1": gen_P1_author_name_variant,
    "P2": gen_P2_non_academic_source,
    "P3": gen_P3_cross_candidate_coverage,
    "P4": gen_P4_insufficient_field_evidence,
    # HALLUCINATED — Title (H1: 3 sub-generators, all labeled H1)
    "H1-sub": gen_H1_word_substitution,
    "H1-para": gen_H1_title_paraphrase,
    "H1-fab": gen_H1_title_fabrication,
    # HALLUCINATED — Author (H2: 3 sub-generators, all labeled H2)
    "H2-add": gen_H2_author_addition_deletion,
    "H2-reorder": gen_H2_author_reordering,
    "H2-fab": gen_H2_author_fabrication,
    # HALLUCINATED — Venue (H3)
    "H3-venue": gen_H3_venue_fabrication,
    "H3-both": gen_H3_venue_and_year,
    # HALLUCINATED — Year (H4)
    "H4": gen_H4_date_error,
    # HALLUCINATED — DOI (H5)
    "H5-fab": gen_H5_doi_fabrication,
    "H5-nonexist": gen_H5_doi_nonexistent,
    # HALLUCINATED — Pages/Volume/Publisher/Location (H6)
    "H6-pages": gen_H6_pages_fabrication,
    "H6-publisher": gen_H6_publisher_fabrication,
    "H6-location": gen_H6_location_fabrication,
}


def _make_record(sample, citation: dict) -> dict:
    return {
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
    }


def generate_mutations(
    citations: list[dict],
    subtypes: list[str] | None = None,
    samples_per_type: int = 1,
    workers: int = 1,
) -> list[dict]:
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    from concurrent.futures import ThreadPoolExecutor, as_completed

    subtypes = subtypes or list(ALL_GENERATORS.keys())
    results: list[dict] = []

    def _attempt(gen):
        """Single attempt: pick a random seed, run generator, return (sample, seed) or None."""
        citation = random.choice(citations)
        if not citation.get("title") or not citation.get("authors"):
            return None
        try:
            sample = gen(citation, all_citations=citations)
        except Exception:
            return "error"
        if sample is None:
            return None
        return (sample, citation)

    for subtype in subtypes:
        gen = ALL_GENERATORS.get(subtype)
        if not gen:
            print(f"  Unknown subtype: {subtype}", flush=True)
            continue

        desc = gen.__doc__.split(":")[0].strip() if gen.__doc__ else subtype
        pbar = tqdm(total=samples_per_type,
                    desc=f"{subtype:12s} {desc[:30]:30s}",
                    unit="sample", dynamic_ncols=True) if tqdm else None

        count = 0
        errors = 0
        skipped = 0
        max_attempts = samples_per_type * 3

        if workers <= 1:
            # Sequential path
            attempts = 0
            while count < samples_per_type and attempts < max_attempts:
                attempts += 1
                res = _attempt(gen)
                if res == "error":
                    errors += 1
                elif res is None:
                    skipped += 1
                else:
                    sample, citation = res
                    results.append(_make_record(sample, citation))
                    count += 1
                    if pbar:
                        pbar.update(1)
                if pbar:
                    pbar.set_postfix({"err": errors, "skip": skipped})
        else:
            # Parallel path: submit tasks in waves of `workers` until target reached.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                submitted = 0
                while count < samples_per_type and submitted < max_attempts:
                    wave = min(workers, samples_per_type - count + errors + skipped, max_attempts - submitted)
                    futures = [pool.submit(_attempt, gen) for _ in range(wave)]
                    submitted += wave
                    for fut in as_completed(futures):
                        res = fut.result()
                        if res == "error":
                            errors += 1
                        elif res is None:
                            skipped += 1
                        else:
                            if count < samples_per_type:
                                sample, citation = res
                                results.append(_make_record(sample, citation))
                                count += 1
                                if pbar:
                                    pbar.update(1)
                        if pbar:
                            pbar.set_postfix({"err": errors, "skip": skipped})

        if pbar:
            pbar.close()
        else:
            print(f"  {subtype:12s} ({desc}): {count}/{samples_per_type} samples", flush=True)

    return results


# ===========================================================================
# CLI
# ===========================================================================

def _parse_bib_file(bib_path: str | Path) -> list[dict]:
    """Parse a BibTeX file into a list of citation dicts."""
    with open(bib_path) as f:
        content = f.read()

    entries = re.findall(r'@\w+\{[^@]+\}', content, re.DOTALL)
    citations = []
    for entry_text in entries:
        m = re.match(r'@(\w+)\{([^,]+),', entry_text)
        if not m:
            continue
        key = m.group(2).strip()

        fields: dict[str, str] = {}
        for fm in re.finditer(r'(\w+)\s*=\s*\{(.*?)\}', entry_text, re.DOTALL):
            fields[fm.group(1).lower()] = fm.group(2).strip()

        authors = []
        if 'author' in fields:
            for a in fields['author'].split(' and '):
                a = a.strip()
                if ',' in a:
                    parts = a.split(',', 1)
                    a = f"{parts[1].strip()} {parts[0].strip()}"
                if a:
                    authors.append(a)

        year = None
        if 'year' in fields:
            try:
                year = int(fields['year'])
            except ValueError:
                pass

        title = fields.get('title', '')
        if not title:
            continue

        # Extract arxiv_id from eprint, url, or journal fields
        arxiv_id = ''
        for src in [fields.get('eprint', ''), fields.get('url', ''), fields.get('journal', '')]:
            aid_m = re.search(r'(\d{4}\.\d{4,5})', src)
            if aid_m:
                arxiv_id = aid_m.group(1)
                break

        citations.append({
            "citation_id": key,
            "title": title,
            "authors": authors,
            "venue": fields.get('journal', '') or fields.get('booktitle', ''),
            "year": year,
            "volume": fields.get('volume', ''),
            "pages": fields.get('pages', ''),
            "publisher": fields.get('publisher', ''),
            "doi": fields.get('doi', ''),
            "url": fields.get('url', ''),
            "arxiv_id": arxiv_id,
            "_source_paper": "bib_input",
        })

    return citations


def main():
    parser = argparse.ArgumentParser(description="Generate hallucinated citation test data.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir", help="Directory with verified citation JSONs.")
    group.add_argument("--bib", help="BibTeX file with valid citations.")
    group.add_argument("--seeds", help="Flat JSON list of citation dicts (from harvest_seed_citations.py).")
    parser.add_argument("--output", default="artifacts/hallucination_test_data.json", help="Output JSON file.")
    parser.add_argument("--total", type=int, default=100, help="Target total number of samples.")
    parser.add_argument("--samples-per-type", type=int, default=None,
                        help="Override samples per generator key (e.g. H1-sub). See --samples-per-code for code-level target.")
    parser.add_argument("--samples-per-code", type=int, default=None,
                        help="Target samples per taxonomy code (e.g. H1 total). Split across sub-variants.")
    parser.add_argument("--subtypes", nargs="+", default=None,
                        help="Restrict to these subtypes (match ALL_GENERATORS keys, e.g. R1 H1-sub P1).")
    parser.add_argument("--exclude-subtypes", nargs="+", default=None,
                        help="Exclude these subtypes (match ALL_GENERATORS keys).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--split-outdir", default=None,
                        help="If set, after generation split samples into per-code files under this dir.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker threads per subtype (LLM calls are IO-bound, 4-8 recommended).")
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
    all_citations: list[dict] = []
    if args.bib:
        all_citations = _parse_bib_file(args.bib)
        print(f"Loaded {len(all_citations)} citations from BibTeX file\n", flush=True)
    elif args.seeds:
        with open(args.seeds) as f:
            raw = json.load(f)
        for c in raw:
            if c.get("title") and c.get("authors"):
                all_citations.append(c)
        print(f"Loaded {len(all_citations)} citations from seeds JSON\n", flush=True)
    else:
        input_dir = Path(args.input_dir)
        for fpath in sorted(input_dir.glob("*.json")):
            with open(fpath) as f:
                data = json.load(f)
            for c in data.get("citations", []):
                if c.get("title") and c.get("authors"):
                    c["_source_paper"] = fpath.stem
                    all_citations.append(c)
        print(f"Loaded {len(all_citations)} citations from JSON files\n", flush=True)

    if not all_citations:
        print("No citations found!")
        return

    # Filter subtypes if requested
    selected_subtypes = list(ALL_GENERATORS.keys())
    if args.subtypes:
        wanted = set(args.subtypes)
        selected_subtypes = [s for s in selected_subtypes if s in wanted]
        missing = wanted - set(ALL_GENERATORS.keys())
        if missing:
            print(f"WARNING: unknown subtypes ignored: {sorted(missing)}")
    if args.exclude_subtypes:
        excluded = set(args.exclude_subtypes)
        selected_subtypes = [s for s in selected_subtypes if s not in excluded]

    n_types = len(selected_subtypes)
    if n_types == 0:
        print("No subtypes selected after filtering!")
        return

    # Compute per-subtype sample counts
    per_subtype: dict[str, int] = {}
    if args.samples_per_code is not None:
        # Group sub-variants by taxonomy code (key before '-')
        from collections import defaultdict
        code_groups: dict[str, list[str]] = defaultdict(list)
        for s in selected_subtypes:
            code_groups[s.split("-")[0]].append(s)
        for code, subs in code_groups.items():
            base = args.samples_per_code // len(subs)
            rem = args.samples_per_code - base * len(subs)
            for i, s in enumerate(subs):
                per_subtype[s] = base + (1 if i < rem else 0)
    else:
        n = args.samples_per_type or max(1, args.total // n_types)
        per_subtype = {s: n for s in selected_subtypes}

    total_target = sum(per_subtype.values())
    print(f"Running {n_types} generator keys, total target = {total_target}")
    for s in selected_subtypes:
        print(f"  {s:15s} {per_subtype[s]}")
    print()

    # Run per-subtype so counts can differ
    results: list[dict] = []
    for s in selected_subtypes:
        batch = generate_mutations(all_citations, subtypes=[s],
                                    samples_per_type=per_subtype[s],
                                    workers=args.workers)
        results.extend(batch)

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

    if args.split_outdir:
        from split_mutations_by_code import split_mutations, print_split_report
        print(f"\nSplitting into per-code files under {args.split_outdir}/ ...")
        manifest = split_mutations(results, args.split_outdir)
        print_split_report(manifest, args.split_outdir)


if __name__ == "__main__":
    main()
