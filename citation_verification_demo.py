from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from apps.pdf_checker.config import load_pdf_checker_config
from packages.connectors import ConnectorOrchestrator, default_orchestrator
from packages.core import CitationVerifier
from packages.core.adjudicate import LLMResolver
from packages.core.models import CandidateMatch, VerdictLabel
from packages.core.models import CitationRecord, ExtractionQuality, citation_verdict_to_dict
from packages.core.normalize import normalize_author, normalize_title, normalize_venue
from packages.core.report import build_summary, compute_risk_score


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run verification-only pipeline on existing parsed citation artifact JSON files."
    )
    parser.add_argument(
        "--input-dir",
        default="artifacts/icml_oral_2026_reference_extracts_20260301_215912_llm_reparse_test_20260302_214651",
        help="Directory containing parsed citation JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Default: artifacts/icml_oral_2026_verification_only_<UTC timestamp>",
    )
    parser.add_argument(
        "--cache-path",
        default="",
        help="Connector cache sqlite path. Empty means read from config.",
    )
    parser.add_argument(
        "--dblp-mirror-path",
        default="",
        help="Offline DBLP mirror JSONL path. Empty means read from config.",
    )
    parser.add_argument(
        "--dblp-sqlite-path",
        default="",
        help="DBLP sqlite path (used when set). Empty means read from config.",
    )
    parser.add_argument(
        "--semantic-scholar-api-key",
        default="",
        help="Semantic Scholar API key (optional).",
    )
    parser.add_argument(
        "--config-path",
        default="config.json",
        help="Config path used by apps.pdf_checker.config (default: config.json).",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Set CITATION_CHECKER_OFFLINE_ONLY=1 for this run.",
    )
    parser.add_argument(
        "--skip-healthcheck",
        action="store_true",
        help="Skip startup connector accessibility checks.",
    )
    parser.add_argument(
        "--enable-llm-source-router",
        action="store_true",
        help="Use a Bedrock LLM-based router to rank sources and cap the number of sources queried per citation.",
    )
    parser.add_argument(
        "--source-router-max-sources",
        type=int,
        default=0,
        help="Optional hard cap for the LLM source router. 0 means trust the router's suggested budget.",
    )
    return parser.parse_args()


def _to_citation_record(payload: dict[str, Any]) -> CitationRecord:
    year_value = payload.get("year")
    try:
        year = int(year_value) if year_value is not None else None
    except (TypeError, ValueError):
        year = None
    return CitationRecord(
        citation_id=str(payload.get("citation_id", "")),
        raw_text=str(payload.get("raw_text", "")),
        title=str(payload.get("title", "")),
        authors=[str(item) for item in (payload.get("authors") or [])],
        venue=str(payload.get("venue", "")),
        year=year,
        doi=str(payload.get("doi", "")),
        arxiv_id=str(payload.get("arxiv_id", "")),
        url=str(payload.get("url", "")),
        source_span=str(payload.get("source_span", "")),
        provenance=payload.get("provenance") or {},
        parsed_fields=payload.get("parsed_fields") or {},
    )


def _build_probe_citation(files: list[Path]) -> CitationRecord:
    for src in files:
        try:
            obj = json.loads(src.read_text(encoding="utf-8"))
        except Exception:
            continue
        for citation in obj.get("citations", []):
            if not isinstance(citation, dict):
                continue
            record = _to_citation_record(citation)
            if record.title or record.raw_text or record.doi or record.arxiv_id:
                if not record.citation_id:
                    record.citation_id = "probe:1"
                return record
    return CitationRecord(
        citation_id="probe:default",
        raw_text="Attention Is All You Need",
        title="Attention Is All You Need",
        authors=["Ashish Vaswani"],
        venue="NeurIPS",
        year=2017,
    )


def _print_connector_health(orchestrator: Any, probe: CitationRecord) -> None:
    print("[healthcheck] Connector accessibility check", flush=True)
    print(
        f"[healthcheck] probe_title={probe.title!r} doi={probe.doi!r} arxiv_id={probe.arxiv_id!r}",
        flush=True,
    )
    for connector in orchestrator.connectors:
        started = time.perf_counter()
        status = "OK"
        error = ""
        records_count = 0
        cache_status = "-"
        try:
            # Run raw connector.search here to check source availability directly.
            records = connector.search(probe, orchestrator.policy)
            records_count = len(records or [])

            # Add connector-specific sanity hints.
            if connector.name == "dblp_offline":
                mirror_path = getattr(connector, "mirror_path", None)
                loaded = len(getattr(connector, "_records", []) or [])
                cache_status = f"mirror_exists={bool(mirror_path and Path(mirror_path).exists())} loaded={loaded}"
            elif connector.name == "dblp_sqlite":
                sqlite_path = getattr(connector, "sqlite_path", None)
                cache_status = f"sqlite_exists={bool(sqlite_path and Path(sqlite_path).exists())}"
        except Exception as exc:  # noqa: BLE001
            status = "FAIL"
            error = str(exc)
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(
            f"[healthcheck] source={connector.name:<16} status={status:<4} "
            f"latency_ms={elapsed_ms:8.1f} records={records_count:<3} detail={cache_status} "
            f"{('error=' + error) if error else ''}",
            flush=True,
        )


def _build_bedrock_client(region: str, bearer_token: str | None = None):
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is not installed. Install boto3 to enable verification_llm.") from exc

    if bearer_token:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = str(bearer_token)
    return boto3.client(service_name="bedrock-runtime", region_name=region)


def _extract_text_from_converse_response(response: dict[str, Any]) -> str:
    content = response.get("output", {}).get("message", {}).get("content", [])
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and "text" in item:
            parts.append(str(item["text"]))
    return "\n".join(parts).strip()


def _parse_json_from_text(text: str) -> dict[str, Any]:
    candidates: list[str] = [text.strip()]
    if "```" in text:
        import re

        candidates.extend(re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL))
        match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if match:
            candidates.append(match.group(1).strip())
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise RuntimeError("Failed to parse JSON from verification_llm response.")


def _slim_report(report_dict: dict[str, Any]) -> dict[str, Any]:
    slim_citations: list[dict[str, Any]] = []
    for citation in report_dict.get("citations", []) or []:
        if not isinstance(citation, dict):
            continue
        candidate_evaluations = citation.get("candidate_evaluations", []) or []
        candidate_sources = sorted(
            {
                str(item.get("connector", "")).strip()
                for item in candidate_evaluations
                if isinstance(item, dict) and str(item.get("connector", "")).strip()
            }
        )
        final_reason = str(citation.get("adjudication_reason", "")).strip()
        llm_reason = str(citation.get("llm_recheck_reason", "")).strip()
        slim_citations.append(
            {
                "citation_id": citation.get("citation_id", ""),
                "title": citation.get("reference_snapshot", {}).get("title", ""),
                "validation_result": citation.get("verdict", ""),
                "evidence_sources": citation.get("evidence_sources", []),
                "candidate_sources": candidate_sources,
                "error_reason": final_reason,
                "summary": {
                    "title": citation.get("reference_snapshot", {}).get("title", ""),
                    "final_validation_result": citation.get("verdict", ""),
                    "candidate_sources": candidate_sources,
                    "llm_rechecked": bool(llm_reason),
                    "final_adjudication_reason": final_reason,
                },
                "all_candidates_compared": candidate_evaluations,
                "llm_recheck_adjudication_reason": llm_reason,
            }
        )

    return {
        "report_version": report_dict.get("report_version", "1.0"),
        "paper_id": report_dict.get("paper_id", ""),
        "pipeline_type": report_dict.get("pipeline_type", ""),
        "citations": slim_citations,
        "summary": report_dict.get("summary", {}),
        "requires_human_review": report_dict.get("requires_human_review", False),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _build_incremental_report(
    paper_id: str,
    pipeline_type: str,
    verdicts: list[Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    raw_report = {
        "report_version": "1.0",
        "paper_id": paper_id,
        "pipeline_type": pipeline_type,
        "citations": [citation_verdict_to_dict(v) for v in verdicts],
        "summary": build_summary(verdicts),
        "risk_score": compute_risk_score(verdicts),
        "requires_human_review": any(v.needs_human_review for v in verdicts),
        "metadata": metadata,
    }
    return _slim_report(raw_report)


def _is_et_al_token(name: str) -> bool:
    normalized = normalize_author(name).replace(" ", "")
    return normalized in {"etal", "etal."}


def _normalize_author_sequence(authors: list[str]) -> tuple[list[str], bool]:
    normalized: list[str] = []
    has_et_al = False
    for author in authors:
        text = str(author or "").strip()
        if not text:
            continue
        if _is_et_al_token(text):
            has_et_al = True
            continue
        normalized_text = normalize_author(text)
        if normalized_text:
            normalized.append(normalized_text)
    return normalized, has_et_al


def _strict_author_match(citation_authors: list[str], candidate_authors: list[str]) -> tuple[bool, str]:
    citation_norm, citation_has_et_al = _normalize_author_sequence(citation_authors)
    candidate_norm, candidate_has_et_al = _normalize_author_sequence(candidate_authors)

    if not citation_norm and not candidate_norm:
        return True, "Both citation and candidate omit author lists."
    if not citation_norm or not candidate_norm:
        return False, "One side is missing authors."

    if citation_has_et_al:
        if len(candidate_norm) < len(citation_norm):
            return False, "Candidate has fewer explicit authors than the citation prefix before et al."
        if candidate_norm[: len(citation_norm)] != citation_norm:
            return False, "Candidate author order does not match the citation author prefix before et al."
        return True, "Citation uses et al and the explicit author prefix matches in order."

    if len(citation_norm) != len(candidate_norm):
        return False, "Citation and candidate have different author counts."
    if citation_norm != candidate_norm:
        return False, "Citation and candidate authors do not match one-to-one in order."
    return True, "Citation and candidate authors match one-to-one in order."


def _author_count_eligible(citation_authors: list[str], candidate_authors: list[str]) -> tuple[bool, str]:
    citation_norm, citation_has_et_al = _normalize_author_sequence(citation_authors)
    candidate_norm, _ = _normalize_author_sequence(candidate_authors)

    if not citation_norm and not candidate_norm:
        return True, "Both citation and candidate omit author lists."
    if not citation_norm or not candidate_norm:
        return False, "One side is missing authors."

    if citation_has_et_al:
        if len(candidate_norm) < len(citation_norm):
            return False, "Candidate has fewer explicit authors than the citation prefix before et al."
        return True, "Candidate has enough authors to evaluate an et al citation."

    if len(citation_norm) != len(candidate_norm):
        return False, "Citation and candidate have different author counts."
    return True, "Citation and candidate author counts match."


def _edit_distance_at_most(left: str, right: str, max_distance: int) -> bool:
    if max_distance < 0:
        return False
    if left == right:
        return True
    if abs(len(left) - len(right)) > max_distance:
        return False

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        row_min = current[0]
        for j, right_char in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (0 if left_char == right_char else 1)
            value = min(insertion, deletion, substitution)
            current.append(value)
            if value < row_min:
                row_min = value
        if row_min > max_distance:
            return False
        previous = current
    return previous[-1] <= max_distance


def _strict_title_match(citation_title: str, candidate_title: str) -> bool:
    left = normalize_title(citation_title)
    right = normalize_title(candidate_title)
    return bool(left) and bool(right) and _edit_distance_at_most(left, right, max_distance=5)


def _strict_venue_ok(citation_venue: str, candidate_venue: str) -> bool:
    left = normalize_venue(citation_venue)
    right = normalize_venue(candidate_venue)
    if not left or not right:
        return True
    if left == "arxiv" or right == "arxiv":
        return True
    return left == right


class BedrockSourceSuitabilityRouter:
    _SOURCE_GUIDE = {
        "dblp_sqlite": "Very fast local CS bibliography source. Strong for computer science conference and journal papers.",
        "dblp_offline": "Fast local DBLP mirror. Useful for CS papers when sqlite is unavailable.",
        "url_direct": "Directly fetches the citation URL and extracts page metadata. Best when the citation already has a trustworthy URL.",
        "dblp_online": "Online DBLP search. Good for CS venues but slower than local DBLP.",
        "crossref": "Strong DOI and publisher metadata source. Good when DOI or formal publication metadata is likely.",
        "arxiv": "Strong for arXiv preprints and papers with arXiv identifiers or arXiv URLs.",
        "acl_anthology": "Useful for NLP papers that may appear in ACL Anthology.",
        "europepmc": "Useful for biomedical and life-science papers.",
        "pubmed": "Useful for biomedical and medical citations.",
        "openalex": "Broad structured scholarly metadata source. Good generic fallback.",
        "semantic_scholar": "Broad scholarly metadata source with identifiers and venue coverage.",
        "govinfo": "Only useful for government and legal documents.",
        "google_scholar": "Expensive but broad scholarly search. Strong fallback when structured sources may miss a paper.",
        "google_search": "Broad web search fallback. Useful when the work may only be mentioned on the web. High latency and noisier.",
        "searxng_search": "General web search fallback through SearxNG. Broad but noisy.",
    }

    def __init__(
        self,
        model_id: str,
        region: str,
        bearer_token: str | None = None,
        max_sources_cap: int = 0,
    ) -> None:
        self.model_id = model_id
        self.region = region
        self.max_sources_cap = max(0, max_sources_cap)
        self.client = _build_bedrock_client(region=region, bearer_token=bearer_token)

    def route(self, citation: CitationRecord, available_sources: list[str]) -> dict[str, Any]:
        prompt = self._build_prompt(citation, available_sources)
        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"temperature": 0},
            )
            text = _extract_text_from_converse_response(response)
            parsed = _parse_json_from_text(text)
        except Exception as exc:  # noqa: BLE001
            return {
                "ordered_sources": list(available_sources),
                "max_sources": self._fallback_budget(citation, len(available_sources)),
                "note": f"LLM source router failed: {exc}",
            }

        proposed = [str(item).strip() for item in (parsed.get("ordered_sources") or []) if str(item).strip()]
        ordered = [name for name in proposed if name in available_sources]
        for source in available_sources:
            if source not in ordered:
                ordered.append(source)

        max_sources = parsed.get("max_sources")
        try:
            max_sources = int(max_sources) if max_sources is not None else None
        except (TypeError, ValueError):
            max_sources = None
        if max_sources is None or max_sources <= 0:
            max_sources = self._fallback_budget(citation, len(available_sources))
        if self.max_sources_cap > 0:
            max_sources = min(max_sources, self.max_sources_cap)
        max_sources = max(1, min(max_sources, len(available_sources)))

        return {
            "ordered_sources": ordered,
            "max_sources": max_sources,
            "note": str(parsed.get("reason", "")).strip(),
        }

    def _build_prompt(self, citation: CitationRecord, available_sources: list[str]) -> str:
        source_docs = []
        for name in available_sources:
            source_docs.append(f"- {name}: {self._SOURCE_GUIDE.get(name, 'General source.')}")
        source_block = "\n".join(source_docs)
        return f"""You are a source suitability router for citation verification.
            Choose which sources should be queried first for the current citation.
            Optimize for: (1) higher probability of finding a VALID match, (2) fewer source queries, (3) lower latency, and (4) lower rate-limit pressure.
            Use citation metadata and source characteristics only. Be pragmatic and conservative.
            Citation: title={citation.title!r} authors={citation.authors!r} venue={citation.venue!r} year={citation.year!r} doi={citation.doi!r} arxiv_id={citation.arxiv_id!r} url={citation.url!r}
            Available sources:
            {source_block}
            Return JSON only:
            {{
            "ordered_sources": ["source1", "source2"],
            "max_sources": <positive integer>,
            "reason": "brief routing rationale"
            }}
            ROUTING RULES:
            1. Prefer sources that fit the citation metadata best.
            2. Prefer low-latency, high-yield sources earlier.
            3. If DOI is present, prioritize structured metadata sources.
            4. If arXiv ID or arXiv URL is present, prioritize arXiv and direct URL retrieval.
            5. For CS/ML venues, prioritize DBLP-like sources and Google Scholar before generic web search.
            6. For biomedical venues, prioritize PubMed and EuropePMC.
            7. For government or legal documents, prioritize GovInfo.
            8. Use Google Search and SearxNG later unless the citation is sparse and structured sources are unlikely to work.
            9. Return only source names from the available source list.
            10. Suggest a small but sufficient max_sources budget.
            """

    @staticmethod
    def _fallback_budget(citation: CitationRecord, total: int) -> int:
        if citation.doi or citation.arxiv_id:
            return min(4, total)
        if citation.url and citation.title:
            return min(5, total)
        if citation.title and citation.authors:
            return min(6, total)
        return min(8, total)


class RoutedConnectorOrchestrator:
    def __init__(self, base_orchestrator: Any, router: BedrockSourceSuitabilityRouter):
        self._base = base_orchestrator
        self._router = router
        self.connectors = getattr(base_orchestrator, "connectors", [])

    def query(self, citation: CitationRecord, max_connectors: int | None = None) -> list[Any]:
        connectors = list(getattr(self._base, "connectors", []) or [])
        if not connectors:
            return self._base.query(citation, max_connectors=max_connectors)
        available_sources = [str(getattr(connector, "name", connector.__class__.__name__)).strip() for connector in connectors]
        plan = self._router.route(citation, available_sources)
        by_name = {str(getattr(connector, "name", connector.__class__.__name__)).strip(): connector for connector in connectors}
        ordered = [by_name[name] for name in plan["ordered_sources"] if name in by_name]
        seen = {str(getattr(connector, "name", connector.__class__.__name__)).strip() for connector in ordered}
        ordered.extend(connector for connector in connectors if str(getattr(connector, "name", connector.__class__.__name__)).strip() not in seen)
        delegated = ConnectorOrchestrator(
            connectors=ordered,
            cache=self._base.cache,
            policy=self._base.policy,
            source_health=self._base.source_health,
        )
        limit = plan.get("max_sources")
        if max_connectors is not None and limit is not None:
            limit = min(max_connectors, limit)
        elif max_connectors is not None:
            limit = max_connectors
        return delegated.query(citation, max_connectors=limit)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


class BedrockStrictMatchResolver(LLMResolver):
    def __init__(
        self,
        model_id: str,
        region: str,
        bearer_token: str | None = None,
        max_candidates: int = 5,
    ) -> None:
        self.model_id = model_id
        self.region = region
        self.max_candidates = max(1, max_candidates)
        self.client = _build_bedrock_client(region=region, bearer_token=bearer_token)

    @staticmethod
    def _build_documents(candidates: list[CandidateMatch], max_candidates: int) -> str:
        top = candidates[:max_candidates]
        docs: list[str] = []
        for i, cand in enumerate(top, start=1):
            payload = cand.raw_record if isinstance(cand.raw_record, dict) else {}
            if cand.connector == "google_search":
                raw_text = str(payload.get("search_result_text", "") or "").strip()
                raw_json = str(payload.get("search_result_json", "") or "").strip()
                docs.append(
                    "\n".join(
                        [
                            f"[{i}] connector={cand.connector} score={cand.score:.4f}",
                            "raw_search_result:",
                            raw_text or "(missing)",
                            f"raw_search_result_json: {raw_json}" if raw_json else "",
                        ]
                    ).strip()
                )
                continue
            extra_lines: list[str] = []
            if payload.get("search_query"):
                extra_lines.append(f"search_query: {payload.get('search_query')}")
            if payload.get("snippet"):
                extra_lines.append(f"snippet: {payload.get('snippet')}")
            if payload.get("search_source"):
                extra_lines.append(f"search_source: {payload.get('search_source')}")
            if payload.get("search_author"):
                extra_lines.append(f"search_author: {payload.get('search_author')}")
            if payload.get("search_date"):
                extra_lines.append(f"search_date: {payload.get('search_date')}")
            docs.append(
                "\n".join(
                    [
                        f"[{i}] connector={cand.connector} score={cand.score:.4f}",
                        f"title: {payload.get('title', cand.title)}",
                        f"authors: {payload.get('authors', cand.authors)}",
                        f"venue: {payload.get('venue', cand.venue)}",
                        f"year: {payload.get('year', cand.year)}",
                        f"doi: {payload.get('doi', cand.doi)}",
                        f"arxiv_id: {payload.get('arxiv_id', cand.arxiv_id)}",
                        f"url: {payload.get('url', cand.url)}",
                        *extra_lines,
                    ]
                )
            )
        return "\n\n".join(docs) if docs else "(no candidates)"

    @staticmethod
    def _build_strict_match_prompt(citation: CitationRecord, documents: str) -> str:
        return (
            "Compare the citation with the downloaded content from search results and determine if any matches.\n"
            f"Citation to verify: Title: {citation.title} Authors: {citation.authors} Venue: {citation.venue} Year: {citation.year}\n"
            f"Downloaded content from the current candidate result: {documents}\n"
            "Return JSON:\n"
            "{\n"
            '  "match": true/false,\n'
            '  "matched_result": <number or null>,\n'
            '  "note": "brief reason"\n'
            "}\n"
            "MATCHING CRITERIA:\n"
            "1. Title must match COMPLETELY.\n"
            "- All significant words must be present.\n"
            '- Ignore only case, punctuation, and articles ("a", "an", "the").\n'
            "2. Authors must match STRICTLY in order.\n"
            "- Treat 'et al' as an omission marker, NOT as an author name.\n"
            "- If the citation does NOT contain 'et al', the citation and candidate must have the SAME number of authors.\n"
            "- If the citation contains 'et al', compare only the explicit authors before 'et al'.\n"
            "- Those explicit authors must match the candidate's leading authors one-by-one and in the SAME order.\n"
            "- If any author is inserted, missing, or shifted before the 'et al' boundary, match must be false.\n"
            "3. Venue matching rules:\n"
            '- If citation venue is "arXiv" OR downloaded content is from arXiv -> ALLOW any venue difference.\n'
            "- If citation venue is a conference (e.g., NeurIPS, ICML, CVPR) AND downloaded content is from a DIFFERENT conference -> REJECT.\n"
            "- If citation venue is conference and downloaded content is journal (or vice versa) -> ALLOW.\n"
            "- Year differences are acceptable if the work is the same (e.g., arXiv 2020, conference 2021).\n"
            "IMPORTANT:\n"
            "- match = true only if title matches AND the strict author rule passes AND venue rules are satisfied.\n"
            "- match = false if title mismatch OR strict author rule fails OR venue rule is violated.\n"
            "- Do NOT guess. Do NOT use partial matches. Be strict and conservative.\n"
            "Return JSON only."
        )

    @staticmethod
    def _build_google_existence_prompt(citation: CitationRecord, documents: str) -> str:
        citation_author_count = len([author for author in citation.authors if str(author or "").strip()])
        return (
            "Determine whether the citation appears to correspond to a real cited work based on Google search results.\n"
            f"Citation to verify: Title: {citation.title} Authors: {citation.authors} "
            f"AuthorCount: {citation_author_count} Venue: {citation.venue} Year: {citation.year}\n"
            f"Google search evidence: {documents}\n"
            "Return JSON:\n"
            "{\n"
            '  "match": true/false,\n'
            '  "matched_result": <number or null>,\n'
            '  "note": "brief reason"\n'
            "}\n"
            "EXISTENCE CHECK RULES:\n"
            "1. Use ONLY the raw Google search result evidence provided below. Do not rely on any extracted metadata fields or external knowledge.\n"
            "2. The result does NOT need to be a formal paper page. It may be a dataset page, GitHub repository, Hugging Face page, Google Scholar profile, bibliography page, personal site, tool page, book page, commercial site, or another page that explicitly mentions the cited work.\n"
            "3. Judge existence by metadata alignment only, not by page type.\n"
            "4. Title must match strongly and refer to the same work.\n"
            "5. Authors must align in content and order. Treat 'et al' as an omission marker, not an author name.\n"
            "6. If the citation does NOT contain 'et al', ALL listed authors must be explicitly supported by the raw search result in the correct order, and the supported author count must equal AuthorCount. Partial author matches are NOT sufficient.\n"
            "7. If the citation contains 'et al', compare only the explicit authors before 'et al'; those explicit authors must match the leading authors in the result one-by-one and in the same order.\n"
            "8. Venue must align when the citation provides a venue. If the citation omits venue, lack of venue evidence must NOT by itself cause rejection.\n"
            "9. Year must align when the citation provides a year. If the citation provides a year and the raw search result does not explicitly support that year, then match=false.\n"
            "10. A direct paper landing page is NOT required. The result does NOT need to prove the work is a standalone publication entity.\n"
            "11. If title, authors, and year align strongly in the raw search result and there is no conflicting information, that is sufficient evidence of existence.\n"
            "12. If the result is generic, off-topic, weakly related, or plausibly about a different work, match=false.\n"
            "13. Be strict and conservative. Do not infer missing authors or venue from likely coauthorship, background knowledge, or thematic similarity.\n"
            "IMPORTANT:\n"
            "- Do not reject solely because the page is not a formal paper page.\n"
            "- Do not reject solely because venue is absent when the citation itself omits venue.\n"
            "- When the citation does not use 'et al', ANY missing author forces match=false.\n"
            "- First count how many citation authors are explicitly supported by the raw search result. If that supported author count is not EXACTLY equal to AuthorCount and the citation does not use 'et al', then you MUST return match=false.\n"
            "- If the citation provides a year, the result MUST explicitly support that same year. Missing or implied year evidence forces match=false.\n"
            "- Do not say that the year is unnecessary when the citation includes a year.\n"
            "- Do not describe incomplete author support as a minor issue; incomplete authors are a decisive reason for match=false.\n"
            "Return JSON only."
        )

    def review(
        self,
        citation: CitationRecord,
        candidates: list[CandidateMatch],
        conflicts: list[str],
        proposed_verdict: VerdictLabel,
    ) -> dict[str, Any] | None:
        if not candidates:
            return {
                "label_override": str(proposed_verdict.value),
                "note": "Bedrock strict match=false matched_result=None. No candidate supplied for LLM recheck.",
            }
        candidate = candidates[0]
        documents = self._build_documents(candidates, self.max_candidates)
        if candidate.connector == "google_search":
            prompt = self._build_google_existence_prompt(citation, documents)
        else:
            prompt = self._build_strict_match_prompt(citation, documents)

        response = self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0},
        )
        text = _extract_text_from_converse_response(response)
        parsed = _parse_json_from_text(text)
        match = bool(parsed.get("match", False))
        note = str(parsed.get("note", "")).strip()
        matched_result = parsed.get("matched_result")

        if match:
            return {
                "label_override": "VALID",
                "confidence_override": 0.9,
                "note": f"Bedrock strict match=true matched_result={matched_result}. {note}".strip(),
            }
        return {
            "label_override": str(proposed_verdict.value),
            "note": f"Bedrock strict match=false matched_result={matched_result}. {note}".strip(),
        }


def main() -> None:
    args = _parse_args()

    if args.config_path:
        os.environ["CITATION_CHECKER_CONFIG_PATH"] = args.config_path

    cfg = load_pdf_checker_config()

    if args.offline_only:
        os.environ["CITATION_CHECKER_OFFLINE_ONLY"] = "1"

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise RuntimeError(f"Input dir not found: {input_dir}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(f"artifacts/icml_oral_2026_verification_only_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(p for p in input_dir.glob("*.json") if not p.name.startswith("_"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {input_dir}")

    print("[setup] Verification-only pipeline start", flush=True)
    print(f"[setup] input_dir={input_dir}", flush=True)
    print(f"[setup] output_dir={output_dir}", flush=True)
    print(f"[setup] files={len(json_files)}", flush=True)
    print(
        f"[setup] offline_only={os.getenv('CITATION_CHECKER_OFFLINE_ONLY', '0')} "
        f"dblp_sqlite_path={args.dblp_sqlite_path!r} "
        f"config_path={os.getenv('CITATION_CHECKER_CONFIG_PATH', 'config.json')}",
        flush=True,
    )

    dblp_sqlite_cli = str(args.dblp_sqlite_path or "").strip()
    dblp_sqlite_cfg = str(cfg.connectors.dblp_sqlite_path or "").strip()
    final_dblp_sqlite = dblp_sqlite_cli or dblp_sqlite_cfg

    semscholar_cli = str(args.semantic_scholar_api_key or "").strip()
    semscholar_cfg = str(cfg.connectors.semantic_scholar_api_key or "").strip()
    final_semscholar_key = semscholar_cli or semscholar_cfg or None

    cache_cli = str(args.cache_path or "").strip()
    mirror_cli = str(args.dblp_mirror_path or "").strip()
    final_cache_path = Path(cache_cli or cfg.connectors.cache_path)
    final_mirror_path = Path(mirror_cli or cfg.connectors.dblp_mirror_path)

    print(
        "[setup] connector_config "
        f"cache={final_cache_path} mirror={final_mirror_path} "
        f"dblp_sqlite={final_dblp_sqlite or '<none>'} "
        f"semantic_scholar_api_key={'<set>' if final_semscholar_key else '<empty>'} "
        f"govinfo_api_key={'<set>' if cfg.connectors.govinfo_api_key else '<empty>'} "
        f"searxng_base_url={cfg.connectors.searxng_base_url or '<empty>'} "
        f"ncbi_api_key={'<set>' if cfg.connectors.ncbi_api_key else '<empty>'} "
        f"google_api_key={'<set>' if cfg.connectors.google_api_key else '<empty>'} "
        f"google_cse_id={'<set>' if cfg.connectors.google_cse_id else '<empty>'} "
        f"serpapi_key={'<set>' if cfg.connectors.serpapi_key else '<empty>'} "
        f"enabled_sources={list(cfg.connectors.enabled_sources) if cfg.connectors.enabled_sources else '<all>'}",
        flush=True,
    )

    orchestrator = default_orchestrator(
        cache_path=final_cache_path,
        dblp_mirror_path=final_mirror_path,
        dblp_sqlite_path=Path(final_dblp_sqlite) if final_dblp_sqlite else None,
        enabled_sources=cfg.connectors.enabled_sources,
        semantic_scholar_api_key=final_semscholar_key,
        govinfo_api_key=cfg.connectors.govinfo_api_key,
        searxng_base_url=cfg.connectors.searxng_base_url,
        ncbi_api_key=cfg.connectors.ncbi_api_key,
        ncbi_email=cfg.connectors.ncbi_email,
        google_api_key=cfg.connectors.google_api_key,
        google_cse_id=cfg.connectors.google_cse_id,
        serpapi_key=cfg.connectors.serpapi_key,
    )

    if not args.skip_healthcheck:
        probe = _build_probe_citation(json_files)
        _print_connector_health(orchestrator, probe)

    llm_resolver = None
    if cfg.verification_llm.enabled and cfg.verification_llm.provider == "bedrock":
        llm_resolver = BedrockStrictMatchResolver(
            model_id=str(cfg.verification_llm.bedrock.model_id),
            region=cfg.verification_llm.bedrock.region,
            bearer_token=cfg.verification_llm.bedrock.bearer_token,
            max_candidates=cfg.verification_llm.max_candidates,
        )
        print(
            "[setup] verification_llm enabled "
            f"provider=bedrock model_id={cfg.verification_llm.bedrock.model_id} "
            f"region={cfg.verification_llm.bedrock.region} max_candidates={cfg.verification_llm.max_candidates}",
            flush=True,
        )
    else:
        print("[setup] verification_llm disabled; using rule-based adjudication only", flush=True)

    if args.enable_llm_source_router:
        if cfg.verification_llm.enabled and cfg.verification_llm.provider == "bedrock":
            source_router = BedrockSourceSuitabilityRouter(
                model_id=str(cfg.verification_llm.bedrock.model_id),
                region=cfg.verification_llm.bedrock.region,
                bearer_token=cfg.verification_llm.bedrock.bearer_token,
                max_sources_cap=args.source_router_max_sources,
            )
            orchestrator = RoutedConnectorOrchestrator(orchestrator, source_router)
            print(
                "[setup] llm_source_router enabled "
                f"model_id={cfg.verification_llm.bedrock.model_id} "
                f"region={cfg.verification_llm.bedrock.region} "
                f"max_sources_cap={args.source_router_max_sources or '<router>'}",
                flush=True,
            )
        else:
            print("[setup] llm_source_router requested but verification_llm bedrock is unavailable; router disabled", flush=True)
    else:
        print("[setup] llm_source_router disabled", flush=True)

    verifier = CitationVerifier(orchestrator=orchestrator, llm_resolver=llm_resolver)

    aggregate = Counter()
    per_file_summaries: list[dict[str, Any]] = []
    total = len(json_files)

    for idx, src in enumerate(json_files, start=1):
        started = time.perf_counter()
        obj = json.loads(src.read_text(encoding="utf-8"))
        raw_citations = obj.get("citations", [])
        citations = [_to_citation_record(c) for c in raw_citations if isinstance(c, dict)]

        print(
            f"[run] [{idx}/{total}] paper={src.stem} citations={len(citations)}",
            flush=True,
        )

        quality_map = {record.citation_id: ExtractionQuality.UNKNOWN for record in citations}
        report_metadata = {"source_artifact": str(src)}
        out_path = output_dir / src.name
        partial_verdicts = []
        total_citations = len(citations)
        for citation_idx, citation in enumerate(citations, start=1):
            print(
                f"[run] [{idx}/{total}] paper={src.stem} citation={citation_idx}/{total_citations} "
                f"id={citation.citation_id}",
                flush=True,
            )
            verdict = verifier.verify_citation(
                citation,
                extraction_quality=quality_map.get(citation.citation_id, ExtractionQuality.UNKNOWN),
            )
            partial_verdicts.append(verdict)
            report_dict = _build_incremental_report(
                paper_id=src.stem,
                pipeline_type="PDF_PARSED",
                verdicts=partial_verdicts,
                metadata=report_metadata,
            )
            _write_json_atomic(out_path, report_dict)
            print(
                f"[write] [{idx}/{total}] paper={src.stem} citation={citation_idx}/{total_citations} "
                f"verdict={verdict.verdict.value} out={out_path}",
                flush=True,
            )

        report_dict = _build_incremental_report(
            paper_id=src.stem,
            pipeline_type="PDF_PARSED",
            verdicts=partial_verdicts,
            metadata=report_metadata,
        )
        summary = report_dict.get("summary", {})
        aggregate.update(summary)
        per_file_summaries.append({"file": src.name, "summary": summary})

        elapsed_s = time.perf_counter() - started
        print(
            f"[done] [{idx}/{total}] paper={src.stem} elapsed_s={elapsed_s:.2f} "
            f"summary={json.dumps(summary, ensure_ascii=False)}",
            flush=True,
        )

    final_summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": len(json_files),
        "aggregate_summary": dict(aggregate),
        "per_file": per_file_summaries,
        "connector_setup": [connector.name for connector in orchestrator.connectors],
        "offline_only": os.getenv("CITATION_CHECKER_OFFLINE_ONLY", "0"),
    }
    summary_path = output_dir / "_verification_summary.json"
    summary_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[final] verification finished", flush=True)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2), flush=True)
    print(f"summary_file={summary_path}", flush=True)


if __name__ == "__main__":
    main()
