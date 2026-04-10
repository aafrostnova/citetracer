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

## Examples

Example 1 — arXiv page:
Content: "Attention is All you Need... Ashish Vaswani, Noam Shazeer, Niki Parmar... Published as a conference paper at NeurIPS 2017"
→ {{"title": "Attention is All you Need", "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"], "venue": "NeurIPS", "year": 2017, "doi": "", "pages": "", "volume": ""}}

Example 2 — GitHub repo:
Content: "[ICLR 2025 Oral] Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models... by kuleshov-group"
→ {{"title": "Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models", "authors": [], "venue": "ICLR", "year": 2025, "doi": "", "pages": "", "volume": ""}}

Example 3 — PubMed page:
Content: "Gate-variants of GRU neural networks... Ravanelli M, Brakel P, Oord A... 2018 IEEE Signal Processing Letters, vol 25, pp 1597-1601. doi:10.1109/LSP.2018.2868880"
→ {{"title": "Gate-variants of GRU neural networks", "authors": ["Ravanelli M", "Brakel P", "Oord A"], "venue": "IEEE Signal Processing Letters", "year": 2018, "doi": "10.1109/LSP.2018.2868880", "pages": "1597-1601", "volume": "25"}}

Example 4 — blog/news:
Content: "Sam Altman: 'I expect AI to be capable of superhuman persuasion...' Tweet, October 2023"
→ {{"title": "I expect AI to be capable of superhuman persuasion", "authors": ["Sam Altman"], "venue": "Tweet", "year": 2023, "doi": "", "pages": "", "volume": ""}}

## Rules
1. Extract ONLY information explicitly present in the content. Do NOT guess or infer.
2. If a field is not found in the content, leave it as empty string or null.
3. For authors, extract ALL authors found. Keep original name format.
4. For venue, extract the conference/journal name (e.g., "NeurIPS", "Nature", "arXiv").
5. For year, extract the publication year as integer.
6. For DOI, extract if present (format: 10.xxxx/...).
7. For pages, extract page range if present (e.g., "1597-1601", "435--444").
8. For volume, extract volume number if present (e.g., "51", "33").
9. For publisher, extract publisher name if present (e.g., "PMLR", "Springer", "ACM", "IEEE").

## Now extract from this content:

{content}

Return JSON only:
{{"title": "...", "authors": [...], "venue": "...", "year": null, "doi": "...", "pages": "...", "volume": "...", "publisher": "..."}}"""


def _call_bedrock(client, model_id: str, prompt: str, max_tokens: int = 512) -> dict:
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
