"""Bedrock LLM implementations of the 3 cascading agents with in-context learning."""
from __future__ import annotations

import json
import re
from typing import Any

from .models import (
    CandidateMatch,
    CitationRecord,
    DiscrepancyEvidence,
    DownstreamAgentResult,
    FieldComparisonResult,
    ValidAgentResult,
    VerdictLabel,
)


def _call_bedrock(client, model_id: str, prompt: str, max_tokens: int = 1024) -> dict:
    response = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"temperature": 0, "maxTokens": max_tokens},
    )
    text = response["output"]["message"]["content"][0]["text"]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    return {}


def _field_summary(fs: dict) -> str:
    lines = []
    for fname, info in fs.items():
        if not isinstance(info, dict):
            continue
        status = info.get("status", "")
        if status in ("both_missing",):
            continue
        ref = str(info.get("reference", ""))[:60]
        cand = str(info.get("candidate", ""))[:60]
        lines.append(f"  {fname}: {status} (ref='{ref}' cand='{cand}')")
    return "\n".join(lines) or "  (no fields to compare)"


def _parse_taxonomy(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(",") if t.strip()]
    return []


# ===========================================================================
# ValidAgent
# ===========================================================================

VALID_AGENT_PROMPT = """You are a citation validity checker. Your job: determine if a citation matches a candidate record from a database or web search.

## Taxonomy

### Base rule (applies to ALL R-types):
To be REAL (valid=true), ALL of the following MUST hold:
  - title: match (after case normalization)
  - year: match
  - venue: match (after normalization/abbreviation/proceedings-prefix) OR both_missing OR candidate_missing

    USE YOUR WORLD KNOWLEDGE to recognize ALL of these as equivalent (match, not mismatch):

    1. Acronym ≡ full name — recognize the acronym regardless of whether it's in your
       training data. Rule: if the short form's letters match the initial letters of the
       full form's content words (skipping "of", "on", "the", "and", "for", "in"), treat as match.
       Examples: "JMLR" ≡ "Journal of Machine Learning Research",
                 "SICOMP" ≡ "SIAM Journal on Computing",
                 "COLT" ≡ "Conference on Learning Theory",
                 "AISTATS" ≡ "Artificial Intelligence and Statistics",
                 "STOC" ≡ "Symposium on Theory of Computing",
                 "UAI" ≡ "Uncertainty in Artificial Intelligence"

    2. Dot-truncated word abbreviations ≡ full name — when each dot-separated token is a
       prefix of the corresponding full word.
       Examples: "J. Mach. Learn. Res." ≡ "Journal of Machine Learning Research",
                 "Artif. Intell." ≡ "Artificial Intelligence",
                 "SIAM J. Comput." ≡ "SIAM Journal on Computing",
                 "IEEE Trans. Pattern Anal. Mach. Intell." ≡ "IEEE Transactions on Pattern Analysis and Machine Intelligence"

    3. Conference vs proceedings prefix:
       "NeurIPS" ≡ "Advances in Neural Information Processing Systems" ≡ "Proceedings of NeurIPS"
       "ICML" ≡ "International Conference on Machine Learning" ≡ "Proceedings of ICML"
       "AISTATS" ≡ "Proceedings of The 24th International Conference on Artificial Intelligence and Statistics"

    4. Conference-proceedings book titles (Springer/LNCS style):
       "Learning Theory and Kernel Machines" ≡ "COLT" (LNCS book title for COLT 2003 proceedings)

    GENERAL PRINCIPLE: If you recognize both names as the same conference/journal (even if
    the automated field_status says "mismatch"), OVERRIDE it and treat as match. You have
    world knowledge the automated comparator doesn't.

    Only treat as MISMATCH when venues are genuinely different publishers/venues
    (e.g. "NeurIPS" vs "ICML", "Personal Blog" vs "NeurIPS", "ICLR" vs "arXiv").
  - authors: match (details vary by R-type below)

HARD RULES — if ANY of these is true, return valid=false immediately:
  - venue is MISMATCH → valid=false (e.g. "Personal Blog" vs "NeurIPS", "ICLR" vs "arXiv")
  - year is MISMATCH → valid=false
  - title is MISMATCH → valid=false

NOTE on authors=reordered_match:
  The automated comparison may report "reordered_match" due to name-format differences
  (e.g. "Wang, S" vs "Sicheng Wang" confusing the token matcher). Do NOT blindly trust it.
  Instead, compare the actual author lists yourself:
  - If the authors are the SAME people in the SAME order (just different name formats like
    "LastName, Initial" vs "FirstName LastName") → treat as match, valid=true
  - If the authors are genuinely reordered (different people in different positions) → valid=false

### REAL (valid=true) — base rules must ALL pass, PLUS:
- R1. Exact match: All fields match exactly (no normalization needed).
- R2. Format variant: Base rules pass after these DETERMINISTIC transformations ONLY:
  * Single-letter initial: "Gao Hao" → "G. Hao" (EXACTLY one letter) ✓ R2
  * Case change: "attention is all you need" vs "Attention Is All You Need" ✓ R2
  * Venue abbreviation: "Artif. Intell." vs "Artificial Intelligence" ✓ R2
  * NOTHING ELSE counts as R2 for author names.
- R3. Et al. abbreviation: Base rules pass, PLUS author list uses "et al."/"Others",
  all listed authors are correct AND in correct order.
  R3 ONLY relaxes author count — venue and year must STILL match.
- R4. Non-academic source verified: Tweet, blog, GitHub confirmed via web search
  with matching title+author+year.

### NOT VALID (valid=false) — give hint
- hint="likely_potential": Discrepancy requires WORLD KNOWLEDGE to explain:
  * Nickname: "Mike" vs "Michael", "Bob" vs "Robert", "Kate" vs "Katherine" → P2
  * Multi-letter truncation: "Edw." vs "Edward", "Séb." vs "Sébastien" → P2 (more than 1 letter)
  * Transliteration: "Wei Zhang" vs "William Zhang" → P2
- hint="likely_hallucinated": Discrepancies are real errors:
  * Title word changed, wrong author count, wrong year by >1, fabricated DOI, venue mismatch.

KEY DISTINCTION for author names:
  R2 = ONLY single-letter initial (exactly 1 character):
    "G. Hao" vs "Gao Hao" → R2 ✓ (G is one letter)
    "S. Bubeck" vs "Sébastien Bubeck" → R2 ✓ (S is one letter)
    "D. D. Johnson" vs "Daniel D. Johnson" → R2 ✓ (D is one letter)
  P2 = EVERYTHING ELSE (multi-letter truncation, nickname, transliteration):
    "Edw. Hu" vs "Edward Hu" → P2 ✗ (Edw is 3 letters, not single-letter initial)
    "Mike Chen" vs "Michael Chen" → P2 ✗ (nickname, not abbreviation)
    "Bob Smith" vs "Robert Smith" → P2 ✗ (nickname)
    "Aar. Gokaslan" vs "Aaron Gokaslan" → P2 ✗ (3-letter truncation)

## Examples

Example 1 — R1 exact match:
Citation: Title="Attention Is All You Need" Authors=["Ashish Vaswani","Noam Shazeer"] Year=2017
Candidate: Title="Attention is All you Need" Authors=["Ashish Vaswani","Noam Shazeer"] Year=2017
Field: title=match, authors=match, year=match
→ {{"valid": true, "taxonomy": ["R1"], "hint": "", "issues": [], "reason": "All fields match."}}

Example 2 — R2 format variant (initial abbreviation):
Citation: Title="attention is all you need" Authors=["A. Vaswani","N. Shazeer"] Year=2017
Candidate: Title="Attention is All you Need" Authors=["Ashish Vaswani","Noam Shazeer"] Year=2017
Field: title=match, authors=partial_overlap (initials vs full names), year=match
→ {{"valid": true, "taxonomy": ["R2"], "hint": "", "issues": [], "reason": "A.=Ashish (first letter), N.=Noam (first letter). Rule-derivable abbreviation."}}

Example 3 — likely_potential (multi-letter truncation → NOT R2):
Citation: Title="Paper X" Authors=["Edw. Hu","Yel. Shen"] Year=2024
Candidate: Title="Paper X" Authors=["Edward Hu","Yelong Shen"] Year=2024
Field: title=match, authors=partial_overlap, year=match
→ {{"valid": false, "taxonomy": [], "hint": "likely_potential", "issues": ["author_name_variant"], "reason": "Edw=3 letters (not single-letter initial), Yel=3 letters. Multi-letter truncation → P2, not R2."}}

Example 4 — R3 et al. (venue must still match!):
Citation: Title="Attention Is All You Need" Authors=["Ashish Vaswani","Others"] Venue="NeurIPS" Year=2017
Candidate: Title="Attention is All you Need" Authors=["Ashish Vaswani","Noam Shazeer","Niki Parmar",...] Venue="NeurIPS" Year=2017
Field: title=match, authors=match (et al detected, listed author correct), venue=match, year=match
→ {{"valid": true, "taxonomy": ["R3"], "hint": "", "issues": [], "reason": "Et al. with correct listed author. Venue and year both match."}}

Example 4b — R3 et al. with NeurIPS proceedings-prefix venue (still MATCH):
Citation: Title="Language models are few-shot learners" Authors=["Tom Brown",...,"et al."] Venue="Advances in neural information processing systems" Year=2020
Candidate: Title="Language Models are Few-Shot Learners" Authors=["Tom B. Brown",...,"Dario Amodei"] Venue="Neural Information Processing Systems" Year=2020
Field: title=match, authors=match (et al), venue=match (NeurIPS proceedings ≡ NeurIPS), year=match
→ {{"valid": true, "taxonomy": ["R3"], "hint": "", "issues": [], "reason": "'Advances in NeurIPS' and 'NeurIPS' are the same venue (proceedings vs conference name)."}}

Example 5 — likely_potential (nickname, NOT rule-derivable):
Citation: Title="Large Language Models" Authors=["Mike Chen","Bob Smith"] Year=2024
Candidate: Title="Large Language Models" Authors=["Michael Chen","Robert Smith"] Year=2024
Field: title=match, authors=partial_overlap, year=match
→ {{"valid": false, "taxonomy": [], "hint": "likely_potential", "issues": ["author_name_variant"], "reason": "Mike≠initial of Michael (M≠Mi), Bob≠initial of Robert (B≠Bo). These are nicknames requiring world knowledge → P2."}}

Example 6 — likely_hallucinated (title word substitution):
Citation: Title="Training novel language models to reason" Authors=["Charlie Chen"] Year=2024
Candidate: Title="Training large language models to reason" Authors=["Charlie Chen"] Year=2024
Field: title=mismatch, authors=match, year=match
→ {{"valid": false, "taxonomy": [], "hint": "likely_hallucinated", "issues": ["title_mismatch"], "reason": "Title word 'novel' differs from 'large' — word substitution."}}

Example 7 — NOT valid (candidate missing volume that reference provides):
Citation: Title="Paper X" Authors=["Alice"] Year=2020 Volume='29' Pages='100-200' Publisher='PMLR'
Candidate: Title="Paper X" Authors=["Alice"] Year=2020 Volume='' Pages='100-200' Publisher='PMLR'
Field: title=match, authors=match, year=match, volume=candidate_missing, pages=match, publisher=match
→ {{"valid": false, "taxonomy": [], "hint": "likely_potential", "issues": ["volume_missing"], "reason": "Reference has volume='29' but candidate does not provide it. All fields the reference provides must be verified by the candidate."}}

Example 8 — likely_hallucinated (author reordered):
Citation: Title="Paper X" Authors=["Bob","Alice","Carol"] Year=2024
Candidate: Title="Paper X" Authors=["Alice","Bob","Carol"] Year=2024
Field: title=match, authors=reordered_match, year=match
→ {{"valid": false, "taxonomy": [], "hint": "likely_hallucinated", "issues": ["author_reordered"], "reason": "Author order differs from published version."}}

Example 8 — likely_hallucinated (year error):
Citation: Title="Paper X" Authors=["Alice"] Year=2027
Candidate: Title="Paper X" Authors=["Alice"] Year=2025
Field: title=match, authors=match, year=mismatch
→ {{"valid": false, "taxonomy": [], "hint": "likely_hallucinated", "issues": ["year_mismatch"], "reason": "Year 2027 vs actual 2025, differs by >1."}}

Example 9 — likely_hallucinated (venue mismatch, even with et al.):
Citation: Title="Paper X" Authors=["Alice","Others"] Venue="Personal Blog" Year=2024
Candidate: Title="Paper X" Authors=["Alice","Bob","Carol"] Venue="NeurIPS" Year=2024
Field: title=match, authors=match (et al), venue=mismatch, year=match
→ {{"valid": false, "taxonomy": [], "hint": "likely_hallucinated", "issues": ["venue_mismatch"], "reason": "Venue 'Personal Blog' vs 'NeurIPS' is a mismatch. Venue mismatch blocks VALID even with et al."}}

## Now evaluate this citation:

Citation: Title='{citation_title}' Authors={citation_authors} Venue='{citation_venue}' Year={citation_year} Volume='{citation_volume}' Pages='{citation_pages}' Publisher='{citation_publisher}'
Candidate (from {connector}): Title='{candidate_title}' Authors={candidate_authors} Venue='{candidate_venue}' Year={candidate_year} Volume='{candidate_volume}' Pages='{candidate_pages}' Publisher='{candidate_publisher}'

Field comparison:
{field_summary}
{web_evidence}
NOTE on volume/pages/publisher:
  - If both sides have a value and they DIFFER → issues should include "volume_mismatch" / "pages_mismatch" / "publisher_mismatch" and valid=false (H7).
  - If the REFERENCE has a value but the CANDIDATE does not (candidate_missing) → valid=false. The candidate must verify all fields the reference provides.
  - If the REFERENCE is missing a value (reference_missing or both_missing) → OK, no verification needed for that field.
  - Page ranges with different separators ("1877-1901" vs "1877--1901" vs "pp. 1877–1901") are the SAME, treat as match.
Return JSON only:
{{"valid": true/false, "taxonomy": [...], "hint": "likely_potential"/"likely_hallucinated"/"", "issues": [...], "reason": "..."}}"""


class BedrockValidAgent:
    def __init__(self, client, model_id: str) -> None:
        self.client = client
        self.model_id = model_id

    def evaluate(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        comparison: FieldComparisonResult,
    ) -> ValidAgentResult:
        fs = comparison.field_status if isinstance(comparison, FieldComparisonResult) else {}
        raw = candidate.raw_record if isinstance(candidate.raw_record, dict) else {}
        snippet = str(raw.get("search_result_text", "") or raw.get("raw_content", "") or "")[:2000]

        web_evidence = ""
        if snippet and candidate.connector in ("google_search", "web_search"):
            web_evidence = f"\nWeb search evidence:\n{snippet[:1500]}\n"

        prompt = VALID_AGENT_PROMPT.format(
            citation_title=citation.title,
            citation_authors=citation.authors,
            citation_venue=citation.venue or "",
            citation_year=citation.year,
            citation_volume=citation.volume or "",
            citation_pages=citation.pages or "",
            citation_publisher=citation.publisher or "",
            connector=candidate.connector,
            candidate_title=candidate.title,
            candidate_authors=candidate.authors,
            candidate_venue=candidate.venue or "",
            candidate_year=candidate.year,
            candidate_volume=getattr(candidate, "volume", "") or "",
            candidate_pages=getattr(candidate, "pages", "") or "",
            candidate_publisher=getattr(candidate, "publisher", "") or "",
            field_summary=_field_summary(fs),
            web_evidence=web_evidence,
        )

        try:
            parsed = _call_bedrock(self.client, self.model_id, prompt)
        except Exception:
            return ValidAgentResult(
                is_valid=False, hint="likely_hallucinated",
                issues=["llm_error"], reason="LLM call failed",
                field_result=comparison, candidate=candidate,
            )

        is_valid = bool(parsed.get("valid", False))
        taxonomy = _parse_taxonomy(parsed.get("taxonomy", []))
        hint = str(parsed.get("hint", "likely_hallucinated")).strip()
        issues = parsed.get("issues", []) or []
        if isinstance(issues, str):
            issues = [issues]
        reason = str(parsed.get("reason", "") or "").strip()

        return ValidAgentResult(
            is_valid=is_valid,
            taxonomy=taxonomy,
            hint=hint if not is_valid else "",
            issues=issues if not is_valid else [],
            reason=reason,
            field_result=comparison,
            candidate=candidate,
            raw_llm_response=parsed,
        )


# ===========================================================================
# PotentialAgent
# ===========================================================================

POTENTIAL_AGENT_PROMPT = """You are a citation discrepancy analyst. The citation was NOT a direct match. Your job: determine if the discrepancies are explainable.

## Taxonomy

### POTENTIAL (discrepancies are explainable)
- P1. Version difference: Metadata matches an older arXiv version (confirmed by version history).
- P2. Author name variant: Name is a plausible nickname/transliteration. E.g., "Katherine" vs "Kate", "Mike" vs "Michael", "Wei Zhang" vs "William Zhang".
- P3. Non-academic source unverifiable: Citation references a non-academic source (blog, tweet, GitHub, personal website) whose existence cannot be fully verified through structured databases. The source may have been deleted/moved, or may still exist but is not indexed by academic databases. Indirect evidence (reposts, caches, web search snippets) may or may not confirm it.

### HALLUCINATED (unexplained errors → escalate)
- If ANY discrepancy cannot be explained by P1/P2/P3, return HALLUCINATED.

## Examples

Example 1 — P2 (author name variant):
Issues: ["author_name_variant"]
Evidence: authors: [EXPLAINED] 'Edw.' is a prefix of 'Edward' — plausible abbreviation.
→ {{"label": "POTENTIAL", "taxonomy": ["P2"], "reason": "Author 'Edw.' is a plausible abbreviation of 'Edward'."}}

Example 2 — P1 (version difference):
Issues: ["year_off_by_one", "venue_preprint_gap"]
Evidence: venue: [EXPLAINED] Paper has both arXiv and DOI, confirming preprint-to-publication.
→ {{"label": "POTENTIAL", "taxonomy": ["P1"], "reason": "Year and venue difference explained by preprint→publication transition."}}

Example 3 — Escalate to HALLUCINATED:
Issues: ["author_name_variant", "venue_mismatch"]
Evidence: authors: [EXPLAINED] prefix match. venue: [UNEXPLAINED] 'Personal Blog' vs 'NeurIPS'.
→ {{"label": "HALLUCINATED", "taxonomy": ["H4"], "reason": "Venue 'Personal Blog' cannot be explained as a variant of 'NeurIPS'."}}

## Now evaluate:

Citation: Title='{citation_title}' Authors={citation_authors} Venue='{citation_venue}' Year={citation_year}
Candidate: Title='{candidate_title}' Authors={candidate_authors} Venue='{candidate_venue}' Year={candidate_year}

Issues from ValidAgent: {issues}
ValidAgent reason: {valid_reason}

Secondary evidence:
{evidence_lines}

Return JSON only:
{{"label": "POTENTIAL"/"HALLUCINATED", "taxonomy": ["P2"], "reason": "..."}}"""


class BedrockPotentialAgent:
    def __init__(self, client, model_id: str) -> None:
        self.client = client
        self.model_id = model_id

    def evaluate(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        comparison: FieldComparisonResult,
        valid_hint: ValidAgentResult,
        evidence: list[DiscrepancyEvidence],
    ) -> DownstreamAgentResult:
        evidence_lines = []
        for ev in evidence:
            tag = "EXPLAINED" if ev.evidence_found else "UNEXPLAINED"
            evidence_lines.append(f"  {ev.field}: [{tag}] {ev.explanation[:120]}")

        prompt = POTENTIAL_AGENT_PROMPT.format(
            citation_title=citation.title,
            citation_authors=citation.authors,
            citation_venue=citation.venue or "",
            citation_year=citation.year,
            candidate_title=candidate.title,
            candidate_authors=candidate.authors,
            candidate_venue=candidate.venue or "",
            candidate_year=candidate.year,
            issues=valid_hint.issues,
            valid_reason=valid_hint.reason,
            evidence_lines="\n".join(evidence_lines) or "  (none gathered)",
        )

        try:
            parsed = _call_bedrock(self.client, self.model_id, prompt)
        except Exception:
            return DownstreamAgentResult(
                label=VerdictLabel.FAKE_REFERENCE,
                reason="PotentialAgent LLM failed",
            )

        label_str = str(parsed.get("label", "HALLUCINATED")).upper()
        if "POTENTIAL" in label_str:
            label = VerdictLabel.POTENTIAL_REFERENCE
        else:
            label = VerdictLabel.FAKE_REFERENCE

        return DownstreamAgentResult(
            label=label,
            taxonomy=_parse_taxonomy(parsed.get("taxonomy", [])),
            reason=str(parsed.get("reason", "") or "").strip(),
            escalate_to="hallucinated" if label == VerdictLabel.FAKE_REFERENCE else "",
            raw_llm_response=parsed,
        )


# ===========================================================================
# HallucinatedAgent
# ===========================================================================

HALLUCINATED_AGENT_PROMPT = """You are a citation error classifier. The citation has confirmed errors. Your job: list ALL errors found.

## Taxonomy — list ALL that apply

- H1: Completely fabricated — no matching paper found anywhere.
- H2: Title error — title differs from real paper (word substitution, paraphrase, or fabrication).
  Example: "Attention Is What You Need" vs real "Attention Is All You Need"
- H3: Author error — wrong authors: addition, deletion, reordering, or fabrication.
  Example: Real "Alice, Bob" → "Alice, Bob, Carol" (addition) or "Bob, Alice" (reorder)
- H4: Venue error — paper exists but at a different venue.
  Example: "In EACL, 2024" but paper only on arXiv.
- H5: Year error — year is verifiably wrong.
  Example: Citation says 2027 but paper published 2025.
- H6: DOI/identifier error — DOI resolves to a different paper or doesn't exist.
- H7: Pages/volume error — page numbers or volume are wrong.

## Examples

Example 1 — single error (title):
Citation: Title="Training novel language models" Authors=["Alice"] Year=2024
Candidate: Title="Training large language models" Authors=["Alice"] Year=2024
Field: title=mismatch, authors=match, year=match
→ {{"label": "HALLUCINATED", "taxonomy": ["H2"], "reason": "Title word 'novel' substituted for 'large'."}}

Example 2 — multiple errors:
Citation: Title="Paper X" Authors=["Alice","Bob","Carol"] Venue="ICML" Year=2027
Candidate: Title="Paper X" Authors=["Alice","Bob"] Venue="NeurIPS" Year=2025
Field: title=match, authors=count_mismatch, venue=mismatch, year=mismatch
→ {{"label": "HALLUCINATED", "taxonomy": ["H3","H4","H5"], "reason": "Author 'Carol' added; venue ICML vs NeurIPS; year 2027 vs 2025."}}

Example 3 — author reorder:
Citation: Title="Paper X" Authors=["Bob","Alice"] Year=2024
Candidate: Title="Paper X" Authors=["Alice","Bob"] Year=2024
Field: title=match, authors=reordered_match, year=match
→ {{"label": "HALLUCINATED", "taxonomy": ["H3"], "reason": "Author order changed: Bob,Alice vs published Alice,Bob."}}

Example 4 — rare downgrade to POTENTIAL:
Citation: Title="Paper X" Authors=["Mike Chen"] Year=2024
Candidate: Title="Paper X" Authors=["Michael Chen"] Year=2024
Field: title=match, authors=partial_overlap, year=match
→ {{"label": "POTENTIAL", "taxonomy": ["P2"], "reason": "Mike is a common nickname for Michael — plausible variant."}}

## Now classify:

Citation: Title='{citation_title}' Authors={citation_authors} Venue='{citation_venue}' Year={citation_year} DOI='{citation_doi}' Pages='{citation_pages}'
Candidate: Title='{candidate_title}' Authors={candidate_authors} Venue='{candidate_venue}' Year={candidate_year}

Field comparison:
{field_summary}

Issues from ValidAgent: {issues}
ValidAgent reason: {valid_reason}

Return JSON only. List ALL errors:
{{"label": "HALLUCINATED"/"POTENTIAL", "taxonomy": ["H2", "H5"], "reason": "..."}}"""


class BedrockHallucinatedAgent:
    def __init__(self, client, model_id: str) -> None:
        self.client = client
        self.model_id = model_id

    def evaluate(
        self,
        citation: CitationRecord,
        candidate: CandidateMatch,
        comparison: FieldComparisonResult,
        valid_hint: ValidAgentResult,
    ) -> DownstreamAgentResult:
        fs = comparison.field_status if isinstance(comparison, FieldComparisonResult) else {}

        prompt = HALLUCINATED_AGENT_PROMPT.format(
            citation_title=citation.title,
            citation_authors=citation.authors,
            citation_venue=citation.venue or "",
            citation_year=citation.year,
            citation_doi=citation.doi or "",
            citation_pages=citation.pages or "",
            candidate_title=candidate.title,
            candidate_authors=candidate.authors,
            candidate_venue=candidate.venue or "",
            candidate_year=candidate.year,
            field_summary=_field_summary(fs),
            issues=valid_hint.issues,
            valid_reason=valid_hint.reason,
        )

        try:
            parsed = _call_bedrock(self.client, self.model_id, prompt)
        except Exception:
            return DownstreamAgentResult(
                label=VerdictLabel.FAKE_REFERENCE,
                taxonomy=["H1"],
                reason="HallucinatedAgent LLM failed",
            )

        label_str = str(parsed.get("label", "HALLUCINATED")).upper()
        if "POTENTIAL" in label_str:
            label = VerdictLabel.POTENTIAL_REFERENCE
        else:
            label = VerdictLabel.FAKE_REFERENCE

        return DownstreamAgentResult(
            label=label,
            taxonomy=_parse_taxonomy(parsed.get("taxonomy", [])),
            reason=str(parsed.get("reason", "") or "").strip(),
            raw_llm_response=parsed,
        )
