"""Parse desk-reject comments into structured citation records.

Uses parse_reference_entry from
/project/pi_shiqingma_umass_edu/mingzheli/Citation_Hallucination_Detection
to extract title/authors/venue/year/doi/arxiv_id/url/volume/pages from each
hallucinated citation string.

Input:  /project/pi_shiqingma_umass_edu/iclr2026_hallucinated/hallucinated_refs.json
Output: /project/pi_shiqingma_umass_edu/iclr2026_hallucinated/hallucinated_refs_structured.json
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

REPO = "/project/pi_shiqingma_umass_edu/mingzheli/Citation_Hallucination_Detection"
sys.path.insert(0, REPO)

from apps.pdf_checker.ingest.citation_parser import parse_reference_entry  # noqa: E402

INPUT = "/project/pi_shiqingma_umass_edu/iclr2026_hallucinated/hallucinated_refs.json"
OUTPUT = "/project/pi_shiqingma_umass_edu/iclr2026_hallucinated/hallucinated_refs_structured.json"

PREAMBLE_PATTERNS = [
    r"^\s*The following references in this submission do not refer to real documents.*?(?:bibliographic information|information)\s*:",
    r"^\s*Desk[-\s]?rejected because of (?:the following )?hallucinated (?:citations?|references?)\s*:",
    r"^\s*Desk[-\s]?rejected because of hallucinated citations\s*:",
    r"^\s*Hallucinated references? are against ICLR polic(?:y|ies)\s*:",
    r"^\s*Hallucinated (?:reference|citation)(?:s)?\s*(?:identified)?\s*:",
    r"^\s*Evidence for LLM usage includes hallucinated references\s*:",
    r"^\s*Desk[-\s]?rejected because of LLM usage.*?\n",
]
PREAMBLE_RE = re.compile("|".join(PREAMBLE_PATTERNS), re.IGNORECASE | re.MULTILINE)

ENTRY_MARKER_RE = re.compile(
    r"^\s*(?:\[\d{1,4}\]|\(\d{1,4}\)|\d{1,4}\.|[\-\*•])\s+"
)

AC_NOTE_PREFIX_RE = re.compile(
    r"^\s*(?:->|—>|->|→|\.\.\.|note:|ac note:|comment:)",
    re.IGNORECASE,
)

AC_NOTE_CONTENT_RE = re.compile(
    r"\b("
    r"this paper does(?:n't| not) (?:exist|contain)|"
    r"these papers? do(?:es)?n'?t exist|"
    r"the first author should be|"
    r"accepted paper list does not contain|"
    r"ai (?:may )?generated this fake reference|"
    r"it is not posted on arxiv|"
    r"the arxiv preprint number is hallucinated|"
    r"this paper does not have .*? as an author|"
    r"appear to be entirely hallucinated|"
    r"it should be|"
    r"paper by that title is not|"
    r"llm artifact|"
    r"please fill in final metadata"
    r")",
    re.IGNORECASE,
)

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")


def strip_preamble(comment: str) -> str:
    m = PREAMBLE_RE.search(comment)
    if m:
        return comment[m.end():].lstrip()
    return comment.lstrip()


def strip_wrapping_quotes(text: str) -> str:
    t = text.strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("“") and t.endswith("”")):
        return t[1:-1].strip()
    return t


def collapse_hard_wraps(text: str) -> str:
    """Collapse mid-sentence single line breaks into spaces while preserving blank lines."""
    paragraphs = re.split(r"\n\s*\n", text)
    joined = []
    for p in paragraphs:
        joined.append(re.sub(r"\s*\n\s*", " ", p).strip())
    return "\n\n".join(joined).strip()


def split_into_chunks(body: str) -> list[str]:
    """Split the body into citation-candidate chunks.

    Operates on the RAW body (newlines preserved) so numbered markers at
    line starts are detectable.
    """
    body = body.strip()
    if not body:
        return []

    # 1. Numbered markers at line starts — highest priority
    numbered = re.split(r"(?m)(?=^\s*(?:\[\d{1,4}\]|\(\d{1,4}\)|\d{1,4}\.)\s+)", body)
    numbered = [p.strip() for p in numbered if p.strip()]
    if len(numbered) >= 2:
        return [re.sub(r"\s*\n\s*", " ", p).strip() for p in numbered]

    # 2. Blank-line separated paragraphs
    paragraphs = [c.strip() for c in re.split(r"\n\s*\n", body) if c.strip()]
    if len(paragraphs) >= 2:
        return [re.sub(r"\s*\n\s*", " ", p).strip() for p in paragraphs]

    # 3. Bullet markers (-, *, •) at line starts
    bulleted = re.split(r"(?m)(?=^\s*[\-\*•]\s+)", body)
    bulleted = [p.strip() for p in bulleted if p.strip()]
    if len(bulleted) >= 2:
        return [re.sub(r"\s*\n\s*", " ", p).strip() for p in bulleted]

    # 4. Author-start heuristic on lines
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if len(lines) >= 2:
        entries: list[str] = []
        current: list[str] = []
        for ln in lines:
            starts_new = bool(
                re.match(r"^[A-Z][^.!?]{1,150}\(\d{4}[a-z]?\)", ln)
                or re.match(r"^[A-Z][A-Za-z'`.\-]+,\s+(?:[A-Z]\.\s*){1,4}", ln)
                or re.match(r"^[A-Z][A-Za-z\-']+(?:,\s+[A-Z][A-Za-z\-']+){1,}", ln)
            )
            if starts_new and current:
                entries.append(" ".join(current))
                current = [ln]
            else:
                current.append(ln)
        if current:
            entries.append(" ".join(current))
        if len(entries) >= 2:
            return entries

    # 5. Single chunk — collapse internal newlines
    return [re.sub(r"\s*\n\s*", " ", body).strip()]


META_COMMENTARY_RE = re.compile(
    r"\b(hallucinat|desk\s?reject|reviewers?|grounds for rejection|against iclr polic|"
    r"llm\s*usage|llm\s*policy|identified by|pointed out by|against policy)\b",
    re.IGNORECASE,
)


def strip_entry_marker(text: str) -> str:
    return ENTRY_MARKER_RE.sub("", text).strip()


def is_ac_note(text: str) -> bool:
    if AC_NOTE_PREFIX_RE.match(text):
        return True
    # Short chunk that mentions "doesn't exist" etc. without year
    if len(text) < 300 and AC_NOTE_CONTENT_RE.search(text) and not YEAR_RE.search(text):
        return True
    return False


def is_plausible_citation(text: str) -> bool:
    if not text or len(text) < 30:
        return False
    # Explicit signals
    if YEAR_RE.search(text):
        return True
    if re.search(r"arXiv:\d{4}\.\d{4,5}", text, re.IGNORECASE):
        return True
    if re.search(r"\b10\.\d{4,9}/\S+", text):
        return True
    # Truncated citations still count if they have an author-like prefix + a period
    if re.search(r"^[A-Z]\.?\s*[A-Za-z'`\-]+(?:,\s+|\s+and\s+)", text) and "." in text:
        return True
    # Fallback: any chunk that looks like a sentence with a comma-separated prefix
    if text.count(",") >= 2 and "." in text:
        return True
    # Any chunk with multiple periods (author. title. venue...) that isn't pure prose
    if text.count(".") >= 2 and len(text) >= 40:
        return True
    return False


def extract_citation_and_note(chunk: str) -> tuple[str, str]:
    """Split a chunk into (citation_text, ac_note). Either may be empty."""
    # Case: leading "->" marker means entire chunk is AC note
    if AC_NOTE_PREFIX_RE.match(chunk):
        return "", chunk.strip()

    # Find the first AC-note marker inside the chunk
    m = re.search(r"(?:\s+->|\s+—>|\s+→|\n->|\n—>)", chunk)
    if m:
        citation = chunk[:m.start()].strip()
        note = chunk[m.start():].lstrip().strip()
        return citation, note

    # Find AC-note content pattern inside chunk
    # This is tricky: we want to keep the citation intact but tag trailing commentary.
    # Heuristic: find the last sentence that matches AC_NOTE_CONTENT_RE; take from its
    # start as AC note.
    content_match = AC_NOTE_CONTENT_RE.search(chunk)
    if content_match:
        # Walk back to sentence boundary
        idx = content_match.start()
        # Find previous ". " before idx
        prev = chunk.rfind(". ", 0, idx)
        if prev > 0 and prev < idx:
            # Check that citation prefix has a year (still a valid citation)
            citation = chunk[:prev + 1].strip()
            note = chunk[prev + 2:].strip()
            if is_plausible_citation(citation):
                return citation, note

    return chunk.strip(), ""


def parse_comment(comment: str) -> list[dict]:
    body = strip_preamble(comment)
    body = strip_wrapping_quotes(body)
    # NOTE: do not collapse newlines before splitting — we rely on line starts
    # for numbered-marker detection.

    chunks = split_into_chunks(body)

    citations: list[dict] = []
    pending_note: list[str] = []

    for chunk in chunks:
        chunk = strip_entry_marker(chunk)
        if not chunk:
            continue

        if is_ac_note(chunk):
            pending_note.append(chunk)
            if citations:
                existing = citations[-1].get("ac_note", "")
                citations[-1]["ac_note"] = (existing + "\n" + chunk).strip() if existing else chunk
                pending_note = []
            continue

        citation_text, ac_note = extract_citation_and_note(chunk)

        if not citation_text:
            if ac_note and citations:
                existing = citations[-1].get("ac_note", "")
                citations[-1]["ac_note"] = (existing + "\n" + ac_note).strip() if existing else ac_note
            continue

        if not is_plausible_citation(citation_text):
            # Reject pure meta-commentary (e.g. "Hallucinated refs pointed out by reviewers...")
            if META_COMMENTARY_RE.search(citation_text) and not YEAR_RE.search(citation_text):
                if citations:
                    existing = citations[-1].get("ac_note", "")
                    citations[-1]["ac_note"] = (existing + "\n" + citation_text).strip() if existing else citation_text
                continue
            # If we have no citations yet, accept as title-only fallback
            if not citations and len(citation_text) >= 15:
                pass
            else:
                if citations:
                    existing = citations[-1].get("ac_note", "")
                    citations[-1]["ac_note"] = (existing + "\n" + citation_text).strip() if existing else citation_text
                continue

        cid = f"ref:{len(citations) + 1}"
        try:
            record = parse_reference_entry(citation_text, cid)
            record_dict = asdict(record)
        except Exception as exc:
            record_dict = {
                "citation_id": cid,
                "raw_text": citation_text,
                "parse_error": str(exc),
            }

        if ac_note:
            record_dict["ac_note"] = ac_note
        if pending_note:
            existing_note = record_dict.get("ac_note", "")
            attached = "\n".join(pending_note)
            record_dict["ac_note"] = (existing_note + "\n" + attached).strip() if existing_note else attached
            pending_note = []

        citations.append(record_dict)

    return citations


def main() -> None:
    with open(INPUT) as f:
        entries = json.load(f)

    out = []
    for i, entry in enumerate(entries, start=1):
        if i % 50 == 0:
            print(f"  {i}/{len(entries)}", flush=True)
        citations = parse_comment(entry["comment"])
        out.append({
            "paper_id": entry["paper_id"],
            "paper_title": entry["title"],
            "url": entry["site"],
            "desk_reject_comment_raw": entry["comment"],
            "hallucinated_citations": citations,
            "num_citations": len(citations),
        })

    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)

    total_citations = sum(o["num_citations"] for o in out)
    papers_with_zero = sum(1 for o in out if o["num_citations"] == 0)
    print(f"\nParsed {len(out)} papers, {total_citations} citations total")
    print(f"Papers with 0 extracted citations: {papers_with_zero}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
