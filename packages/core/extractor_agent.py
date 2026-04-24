"""Extractor Agent: Extract structured citation fields from web search raw content.

Given raw_content/snippet from web search, uses LLM to extract:
  title, authors, venue, year, doi, pages, volume
Then fills these into the CandidateMatch so downstream agents can do
normal field-level comparison.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .models import CandidateMatch


EXTRACTOR_PROMPT = """You are a metadata extractor. Given web search result content about an academic paper, extract structured citation fields.

## Positive examples

Example 1 — arXiv page:
Content: "Attention is All you Need... Ashish Vaswani, Noam Shazeer, Niki Parmar... Published as a conference paper at NeurIPS 2017"
→ {{"title": "Attention is All you Need", "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"], "venue": "NeurIPS", "year": 2017, "doi": "", "pages": "", "volume": "", "publisher": ""}}

Example 2 — GitHub repo:
Content: "[ICLR 2025 Oral] Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models... by kuleshov-group"
→ {{"title": "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models", "authors": [], "venue": "ICLR", "year": 2025, "doi": "", "pages": "", "volume": "", "publisher": ""}}

Example 3 — PubMed page (real journal volume present):
Content: "Gate-variants of GRU neural networks... Ravanelli M, Brakel P, Oord A... 2018 IEEE Signal Processing Letters, vol 25, pp 1597-1601. doi:10.1109/LSP.2018.2868880"
→ {{"title": "Gate-variants of GRU neural networks", "authors": ["Ravanelli M", "Brakel P", "Oord A"], "venue": "IEEE Signal Processing Letters", "year": 2018, "doi": "10.1109/LSP.2018.2868880", "pages": "1597-1601", "volume": "25", "publisher": ""}}

Example 4 — blog/news:
Content: "Sam Altman: 'I expect AI to be capable of superhuman persuasion...' Tweet, October 2023"
→ {{"title": "I expect AI to be capable of superhuman persuasion", "authors": ["Sam Altman"], "venue": "Tweet", "year": 2023, "doi": "", "pages": "", "volume": "", "publisher": ""}}

## Negative examples (common mistakes — DO NOT do these)

Example N1 — ACL conference paper, NO volume:
Content: "Language Models as an Alternative Evaluator... Tatsuki Kuribayashi, Takumi Ito... Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 488-504, Online, July 2020"
WRONG:  {{"volume": "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics", ...}}
WRONG:  {{"volume": "58", ...}}  (the "58th" qualifies the meeting, not a journal volume)
CORRECT: {{"title": "Language Models as an Alternative Evaluator", "authors": ["Tatsuki Kuribayashi", "Takumi Ito"], "venue": "ACL", "year": 2020, "pages": "488-504", "volume": "", "publisher": "Association for Computational Linguistics", "doi": ""}}

Example N2 — ACL Anthology slug is NOT volume:
Content: "[2024.findings-acl.123] Attention Heads... arXiv:2404.01234..."
WRONG:  {{"volume": "2024.findings-acl", ...}}
WRONG:  {{"volume": "abs/2404.01234", ...}}
CORRECT: {{"volume": "", ...}}

Example N3 — Conference proceedings with numeric-looking fragments:
Content: "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 869-893, December 2023"
WRONG:  {{"volume": "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing", ...}}
WRONG:  {{"volume": "2023", ...}}
CORRECT: {{"venue": "EMNLP", "year": 2023, "pages": "869-893", "volume": "", ...}}

Example N4 — Real journal volume with explicit notation:
Content: "Published in Nature, Vol. 612, pp. 123-130 (2022)"
CORRECT: {{"venue": "Nature", "year": 2022, "pages": "123-130", "volume": "612", ...}}

## Rules

1. Extract ONLY verbatim values from the content. PREFER EMPTY STRING over a guess. If you are less than 95% sure a value is correct, leave it empty ("").
2. If a field is not found in the content, output empty string or null. Do NOT invent, infer, or repurpose other text to fill a field.
3. For authors, extract ALL authors found. Keep original name format.
4. For venue, extract the conference/journal name (e.g., "NeurIPS", "Nature", "arXiv").
5. For year, extract the publication year as integer.
6. For DOI, extract if present (format: 10.xxxx/...).
7. For pages, extract page range ONLY in forms like "1597-1601", "435--444", "e123-e128", or a single page number. Never put a venue title, proceedings title, or free text in "pages".
8. For volume: a SHORT journal/periodical volume identifier, usually 1-4 characters, numeric or alphanumeric (e.g., "25", "612", "A23"). ONLY extract volume if content contains explicit "Vol. N" / "Volume N" / "vol. N" notation near the venue.

   DO NOT put in "volume":
   - Conference proceedings titles (e.g. "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics")
   - Conference/workshop/venue names ("EMNLP", "CVPR 2024", "Findings of ACL")
   - Paper slugs / IDs ("2024.findings-acl", "2024.findings-emnlp.45", "abs/2110.07679", "arXiv:2110.07679")
   - Year numbers (those go in "year")
   - Page numbers (those go in "pages")
   - DOI fragments or URL paths

   Most conference papers have NO volume — leave empty ("").
9. For publisher, extract publisher name if present (e.g., "PMLR", "Springer", "ACM", "IEEE", "Association for Computational Linguistics").

## Now extract from this content:

{content}

Return JSON only. When in doubt, leave fields empty:
{{"title": "...", "authors": [...], "venue": "...", "year": null, "doi": "...", "pages": "...", "volume": "...", "publisher": "..."}}"""


def _call_bedrock(client, model_id: str, prompt: str, max_tokens: int = 512) -> dict:
    import sys as _sys
    try:
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0, "maxTokens": max_tokens},
        )
    except Exception as exc:
        msg = str(exc)
        if ("Throttling" in msg or "TooManyRequests" in msg
                or "rate" in msg.lower() or "quota" in msg.lower()):
            print(
                f"[rate-limit] bedrock 429/Throttling on model={model_id} (extractor): {msg[:200]}",
                file=_sys.stderr, flush=True,
            )
        raise
    text = response["output"]["message"]["content"][0]["text"]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    return {}


class BedrockExtractorAgent:
    """Extract structured fields from web search raw content using LLM."""

    def __init__(self, client, model_id: str) -> None:
        self.client = client
        self.model_id = model_id

    def extract(self, candidate: CandidateMatch) -> CandidateMatch:
        """Extract fields from raw_content/snippet and fill into candidate."""
        raw = candidate.raw_record if isinstance(candidate.raw_record, dict) else {}
        raw_content = str(raw.get("raw_content", "") or "")
        snippet = str(raw.get("snippet", "") or raw.get("search_result_text", "") or "")
        content = raw_content[:3000] if raw_content else snippet[:2000]

        if not content.strip():
            return candidate

        prompt = EXTRACTOR_PROMPT.format(content=content)
        try:
            parsed = _call_bedrock(self.client, self.model_id, prompt)
        except Exception:
            return candidate

        if not parsed:
            return candidate

        # Fill extracted fields into candidate (only if currently empty)
        from dataclasses import replace as _replace

        new_title = str(parsed.get("title", "") or "").strip()
        new_authors = parsed.get("authors") or []
        if isinstance(new_authors, str):
            new_authors = [a.strip() for a in new_authors.split(",") if a.strip()]
        new_venue = str(parsed.get("venue", "") or "").strip()
        raw_year = parsed.get("year")
        new_year = int(raw_year) if raw_year is not None else None
        new_doi = str(parsed.get("doi", "") or "").strip()
        new_pages = str(parsed.get("pages", "") or "").strip()
        new_volume = str(parsed.get("volume", "") or "").strip()
        new_publisher = str(parsed.get("publisher", "") or "").strip()

        # Update raw_record with extraction info
        updated_raw = dict(raw)
        updated_raw["extractor_agent_result"] = parsed

        return _replace(
            candidate,
            title=new_title or candidate.title,
            authors=new_authors or candidate.authors,
            venue=new_venue or candidate.venue,
            year=new_year if new_year is not None else candidate.year,
            doi=new_doi or candidate.doi,
            pages=new_pages or candidate.pages,
            volume=new_volume or candidate.volume,
            publisher=new_publisher or candidate.publisher,
            raw_record=updated_raw,
        )
