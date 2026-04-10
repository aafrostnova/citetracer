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
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Process at most N input files. 0 means all files.",
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
            connector_probe = probe
            if connector.name == "arxiv":
                # arXiv id lookup is more reliable than broad title search against the legacy API.
                connector_probe = CitationRecord(citation_id="probe:arxiv", arxiv_id="1706.03762v5")
            # Run raw connector.search here to check source availability directly.
            records = connector.search(connector_probe, orchestrator.policy)
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


def _slim_candidate(cand_eval: dict[str, Any]) -> dict[str, Any]:
    """Remove empty fields and trim large content from a candidate evaluation."""
    candidate = cand_eval.get("candidate", {})
    if not isinstance(candidate, dict):
        return cand_eval

    # Slim the candidate: remove empty fields
    slim_cand: dict[str, Any] = {}
    for k, v in candidate.items():
        if k == "raw_record":
            continue  # Handle separately
        if v is None or v == "" or v == [] or v == {}:
            continue
        slim_cand[k] = v

    # Slim raw_record: keep only non-empty fields, truncate large text
    raw = candidate.get("raw_record", {})
    if isinstance(raw, dict):
        slim_raw: dict[str, Any] = {}
        for k, v in raw.items():
            if v is None or v == "" or v == [] or v == {}:
                continue
            if k == "raw_content" and isinstance(v, str) and len(v) > 2000:
                slim_raw[k] = v[:2000] + "...[truncated]"
            elif k == "search_result_json":
                continue  # Redundant with other fields
            elif isinstance(v, str) and len(v) > 1000:
                slim_raw[k] = v[:1000] + "...[truncated]"
            else:
                slim_raw[k] = v
        if slim_raw:
            slim_cand["raw_record"] = slim_raw

    # Slim comparison: remove empty fields from field_status
    comparison = cand_eval.get("comparison", {})
    if isinstance(comparison, dict):
        field_status = comparison.get("field_status", {})
        slim_fs = {}
        for fname, info in field_status.items():
            status = info.get("status", "")
            if status == "both_missing":
                continue  # Skip entirely empty fields
            slim_fs[fname] = info
        comparison = {k: v for k, v in comparison.items() if k != "field_status"}
        comparison["field_status"] = slim_fs
    else:
        comparison = {}

    result = {
        "connector": cand_eval.get("connector", ""),
        "candidate": slim_cand,
        "verdict": cand_eval.get("verdict", ""),
        "reason": cand_eval.get("reason", ""),
    }
    llm_reason = cand_eval.get("llm_recheck_reason", "")
    if llm_reason:
        result["llm_recheck_reason"] = llm_reason
    if comparison:
        result["comparison"] = comparison
    sec = cand_eval.get("secondary_evidence", [])
    if sec:
        result["secondary_evidence"] = sec
    return result


def _slim_report(report_dict: dict[str, Any]) -> dict[str, Any]:
    slim_citations: list[dict[str, Any]] = []
    for citation in report_dict.get("citations", []) or []:
        if not isinstance(citation, dict):
            continue
        candidate_evaluations = citation.get("candidate_evaluations", []) or []
        slim_evals = [_slim_candidate(ev) for ev in candidate_evaluations if isinstance(ev, dict)]
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
                "taxonomy_subtype": citation.get("taxonomy_subtype", []),
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
                "all_candidates_compared": slim_evals,
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
        "web_search": "Broad web search fallback. Provider is configurable (for example SerpAPI or Tavily). High latency and noisier.",
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


class BedrockWebSearchEnricher:
    """Uses Bedrock LLM to extract structured fields from web search snippets."""

    def __init__(
        self,
        model_id: str,
        region: str,
        bearer_token: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.client = _build_bedrock_client(region=region, bearer_token=bearer_token)

    def enrich(self, citation: CitationRecord, candidate: CandidateMatch) -> CandidateMatch:
        raw = candidate.raw_record if isinstance(candidate.raw_record, dict) else {}
        snippet = str(raw.get("search_result_text", "") or "").strip()
        raw_json = str(raw.get("search_result_json", "") or "").strip()
        evidence = snippet or raw_json
        if not evidence:
            return candidate

        prompt = (
            "Extract structured metadata from the following web search result.\n"
            "Return ONLY valid JSON with these exact keys:\n"
            '{"title": "", "authors": [], "venue": "", "year": null}\n'
            "Rules:\n"
            "- Extract ONLY information that is EXPLICITLY present in the search result text.\n"
            "- title: the paper/work title as shown in the result. Remove prefixes like '[PDF]', '[Re]'.\n"
            "- authors: list of author names found in the result. Empty list if none found.\n"
            "- venue: journal or conference name if mentioned. Empty string if not found.\n"
            "- year: publication year as integer if found, null if not.\n"
            "- Do NOT guess or infer missing information.\n"
            "- Do NOT copy information from the query — only from the result content.\n\n"
            f"Web search result:\n{evidence[:2000]}"
        )

        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"temperature": 0, "maxTokens": 512},
            )
            text = _extract_text_from_converse_response(response)
            parsed = _parse_json_from_text(text)
        except Exception:
            return candidate

        if not parsed:
            return candidate

        new_title = str(parsed.get("title", "") or "").strip()
        new_authors = [str(a).strip() for a in parsed.get("authors", []) if str(a).strip()]
        new_venue = str(parsed.get("venue", "") or "").strip()
        new_year = parsed.get("year")
        if isinstance(new_year, str):
            try:
                new_year = int(new_year)
            except ValueError:
                new_year = None

        # Update candidate fields (keep raw_record intact for auditability)
        from dataclasses import replace as _replace
        enriched = _replace(
            candidate,
            title=new_title or candidate.title,
            authors=new_authors or candidate.authors,
            venue=new_venue or candidate.venue,
            year=new_year if new_year is not None else candidate.year,
        )
        return enriched


class BedrockStructuredJudge:
    """Agent 3: LLM judge for structured source candidates. Returns POTENTIAL or FAKE only."""

    def __init__(self, model_id: str, region: str, bearer_token: str | None = None) -> None:
        self.model_id = model_id
        self.client = _build_bedrock_client(region=region, bearer_token=bearer_token)

    def judge(self, citation, candidate, comparison, evidence) -> Any:
        from packages.core.models import LLMJudgment, VerdictLabel, FieldComparisonResult

        # Build field status summary
        fs = comparison.field_status if isinstance(comparison, FieldComparisonResult) else comparison.get("field_status", {})
        field_lines = []
        for fname, info in (fs.items() if isinstance(fs, dict) else []):
            status = info.get("status", "")
            if status not in ("both_missing",):
                ref = str(info.get("reference", ""))[:60]
                cand = str(info.get("candidate", ""))[:60]
                field_lines.append(f"  {fname}: {status} (ref='{ref}' cand='{cand}')")

        evidence_lines = []
        for ev in evidence:
            status = "EXPLAINED" if ev.evidence_found else "UNEXPLAINED"
            evidence_lines.append(f"  {ev.field}: [{status}] {ev.explanation[:100]}")

        prompt = (
            "You are a citation field comparison specialist.\n"
            f"Citation: Title='{citation.title}' Authors={citation.authors} "
            f"Venue='{citation.venue}' Year={citation.year}\n"
            f"Candidate (from {candidate.connector}): Title='{candidate.title}' "
            f"Authors={candidate.authors} Venue='{candidate.venue}' Year={candidate.year}\n\n"
            f"Field comparison:\n" + "\n".join(field_lines) + "\n\n"
            f"Secondary evidence:\n" + ("\n".join(evidence_lines) if evidence_lines else "  (none)") + "\n\n"
            "TASK: Determine label and list ALL detected issues.\n"
            "- POTENTIAL: discrepancies exist but are explained (name variant, version diff)\n"
            "- HALLUCINATED: ANY unexplained error → HALLUCINATED. List ALL errors found.\n"
            "You CANNOT return REAL/VALID. Only the rule-based system declares REAL.\n"
            "IMPORTANT: Venue 'Personal Blog', 'example.com', or any non-academic generic venue is HALLUCINATED (H9).\n"
            "A preprint-publication difference (arXiv vs conference) is NOT an excuse for completely different venues.\n\n"
            "TAXONOMY — list ALL that apply:\n"
            "  POTENTIAL: P1 (version diff), P2 (author name variant)\n"
            "  HALLUCINATED:\n"
            "    H1: completely fabricated  H2: title error  H3: author error\n"
            "    H4: venue error  H5: year error  H6: DOI/identifier error\n"
            "    H7: pages/volume error  H8: sibling paper confusion  H9: non-existent source\n\n"
            'Return JSON:\n'
            '{"label": "POTENTIAL"/"HALLUCINATED", "taxonomy": ["H2", "H3"], "reason": "..."}\n'
            "If HALLUCINATED, taxonomy must list ALL detected errors (e.g. [\"H2\", \"H5\"]).\n"
            "If POTENTIAL, taxonomy lists potential issues (e.g. [\"P2\"]).\n"
            "Return JSON only."
        )

        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"temperature": 0, "maxTokens": 512},
            )
            text = _extract_text_from_converse_response(response)
            parsed = _parse_json_from_text(text)
        except Exception as exc:
            return LLMJudgment(verdict=VerdictLabel.FAKE_REFERENCE, note=f"LLM error: {exc}")

        label_str = str(parsed.get("label", parsed.get("verdict", "HALLUCINATED"))).strip().upper()
        if "POTENTIAL" in label_str:
            verdict = VerdictLabel.POTENTIAL_REFERENCE
        else:
            verdict = VerdictLabel.FAKE_REFERENCE

        raw_tax = parsed.get("taxonomy", [])
        if isinstance(raw_tax, str):
            taxonomy = [t.strip() for t in raw_tax.split(",") if t.strip()]
        elif isinstance(raw_tax, list):
            taxonomy = [str(t).strip() for t in raw_tax if str(t).strip()]
        else:
            taxonomy = []

        return LLMJudgment(
            verdict=verdict,
            taxonomy=taxonomy,
            confidence=float(parsed.get("confidence", 0.5)),
            note=str(parsed.get("reason", parsed.get("note", "")) or "").strip(),
            raw_response=parsed,
        )


class BedrockExistenceJudge:
    """Agent 4: LLM judge for web search candidates. Returns VALID/POTENTIAL/FAKE."""

    def __init__(self, model_id: str, region: str, bearer_token: str | None = None) -> None:
        self.model_id = model_id
        self.client = _build_bedrock_client(region=region, bearer_token=bearer_token)
        self.source_paper_title: str = ""

    def judge(self, citation, candidate, source_paper_title: str = "") -> Any:
        from packages.core.models import LLMJudgment, VerdictLabel

        raw = candidate.raw_record if isinstance(candidate.raw_record, dict) else {}
        search_text = str(raw.get("search_result_text", "") or "").strip()
        raw_content = str(raw.get("raw_content", "") or "")
        evidence_text = raw_content[:3000] if raw_content else search_text[:2000]

        source_title = source_paper_title or self.source_paper_title
        source_line = ""
        if source_title:
            source_line = (
                f"IMPORTANT: The CITING paper is titled '{source_title}'. "
                f"If a search result IS this paper or its reference list, reject it.\n"
            )

        citation_author_count = len([a for a in citation.authors if str(a or "").strip()])
        prompt = (
            "You are a citation existence checker.\n"
            f"{source_line}"
            f"Citation: Title='{citation.title}' Authors={citation.authors} "
            f"AuthorCount={citation_author_count} Venue='{citation.venue}' Year={citation.year}\n\n"
            f"Search evidence:\n{evidence_text}\n\n"
            "TASK: Does this search result confirm the cited work exists? List ALL issues found.\n"
            "- REAL: source IS the cited work with ALL metadata matching\n"
            "- POTENTIAL: partial evidence or source deleted/moved\n"
            "- HALLUCINATED: any error found or no evidence the work exists\n\n"
            "RULES:\n"
            "1. CIRCULAR REFERENCE: if result is a DIFFERENT paper citing this work → HALLUCINATED\n"
            "2. Non-academic sources (tweets, blogs, GitHub) are legitimate\n"
            "3. Title must match. Different title = different work\n"
            "4. Without 'et al': ALL authors must be present\n"
            "5. Year must match when provided\n\n"
            "TAXONOMY — list ALL that apply:\n"
            "  REAL: R1 (exact match), R4 (non-academic verified)\n"
            "  POTENTIAL: P1 (version diff), P2 (author name variant), P3 (unstable source)\n"
            "  HALLUCINATED: H1 (fabricated), H2 (title error), H3 (author error),\n"
            "    H4 (venue error), H5 (year error), H6 (DOI error), H9 (non-existent source)\n\n"
            'Return JSON:\n'
            '{"label": "REAL"/"POTENTIAL"/"HALLUCINATED", "taxonomy": ["R4"], "reason": "..."}\n'
            "If HALLUCINATED, taxonomy lists ALL errors (e.g. [\"H2\", \"H3\"]).\n"
            "Return JSON only."
        )

        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"temperature": 0, "maxTokens": 512},
            )
            text = _extract_text_from_converse_response(response)
            parsed = _parse_json_from_text(text)
        except Exception as exc:
            return LLMJudgment(verdict=VerdictLabel.FAKE_REFERENCE, note=f"LLM error: {exc}")

        label_str = str(parsed.get("label", parsed.get("verdict", "HALLUCINATED"))).strip().upper()
        if "REAL" in label_str or "VALID" in label_str:
            verdict = VerdictLabel.VALID
        elif "POTENTIAL" in label_str:
            verdict = VerdictLabel.POTENTIAL_REFERENCE
        else:
            verdict = VerdictLabel.FAKE_REFERENCE

        raw_tax = parsed.get("taxonomy", [])
        if isinstance(raw_tax, str):
            taxonomy = [t.strip() for t in raw_tax.split(",") if t.strip()]
        elif isinstance(raw_tax, list):
            taxonomy = [str(t).strip() for t in raw_tax if str(t).strip()]
        else:
            taxonomy = []

        return LLMJudgment(
            verdict=verdict,
            taxonomy=taxonomy,
            confidence=float(parsed.get("confidence", 0.5)),
            note=str(parsed.get("reason", parsed.get("note", "")) or "").strip(),
            raw_response=parsed,
        )


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
        self.source_paper_title: str = ""  # Set per-paper before verification

    @staticmethod
    def _build_documents(candidates: list[CandidateMatch], max_candidates: int) -> str:
        top = candidates[:max_candidates]
        docs: list[str] = []
        for i, cand in enumerate(top, start=1):
            payload = cand.raw_record if isinstance(cand.raw_record, dict) else {}
            if cand.connector in {"google_search", "web_search"}:
                raw_text = str(payload.get("search_result_text", "") or "").strip()
                raw_json = str(payload.get("search_result_json", "") or "").strip()
                docs.append(
                    "\n".join(
                        [
                            f"[{i}] connector={cand.connector}",
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
                        f"[{i}] connector={cand.connector}",
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
            '  "taxonomy": "R1/R2/R3/P1/P2/H2a/H3a/H3b/H3c/H4/H5a/H5d/H6/...",\n'
            '  "note": "brief reason"\n'
            "}\n"
            "MATCHING CRITERIA:\n"
            "1. Title must match COMPLETELY. Ignore only case, punctuation, articles.\n"
            "2. Authors must match STRICTLY in order. Treat 'et al'/'Others' as omission markers.\n"
            "   - Without 'et al': citation and candidate must have the SAME number of authors.\n"
            "   - With 'et al': compare only the explicit authors before 'et al'.\n"
            "3. Venue: arXiv allows any venue difference. Same conference required otherwise.\n"
            "4. Year differences acceptable if same work (e.g., arXiv 2020, conference 2021).\n"
            "\n"
            "TAXONOMY — classify the result into one of these subtypes:\n"
            "REAL (match=true):\n"
            "  R1: Exact match — all fields verified\n"
            "  R2: Format variant — only case/abbreviation/initial differences\n"
            "  R3: Et al. abbreviation — author list truncated with et al., listed authors correct\n"
            "POTENTIAL (match=true, but with caveats):\n"
            "  P1: Version difference — metadata matches an older arXiv version\n"
            "  P2: Author name variant — plausible nickname/transliteration (Katherine→Kate)\n"
            "HALLUCINATED (match=false):\n"
            "  H1: Completely fabricated — no trace of this paper anywhere\n"
            "  H2a: Word substitution — 1-2 title words replaced with synonyms\n"
            "  H2b: Title paraphrase — title rephrased with different wording\n"
            "  H2c: Title fabrication — semantically different title\n"
            "  H3a: Author addition/deletion — wrong number of authors (no et al.)\n"
            "  H3b: Author reordering — author order wrong\n"
            "  H3c: Author fabrication — entirely fake author names\n"
            "  H4: Venue/year fabrication — wrong venue AND year\n"
            "  H5a: DOI fabrication — DOI resolves to different paper\n"
            "  H5c: Date error — year is verifiably wrong\n"
            "  H5d: Venue mismatch — paper exists but at different venue\n"
            "  H5e: Pages/volume fabrication — wrong page numbers\n"
            "  H6: Sibling paper confusion — metadata mixed from two papers\n"
            "\n"
            "IMPORTANT:\n"
            "- match=true only if title+authors+venue all pass. Be strict.\n"
            "- Always provide the most specific taxonomy subtype.\n"
            "Return JSON only."
        )

    @staticmethod
    def _build_google_existence_prompt(citation: CitationRecord, documents: str, source_paper_title: str = "") -> str:
        citation_author_count = len([author for author in citation.authors if str(author or "").strip()])
        source_line = ""
        if source_paper_title:
            source_line = (
                f"IMPORTANT — The citing paper (the paper that contains this citation) is titled: "
                f"\"{source_paper_title}\". If a search result IS this citing paper or contains "
                f"its reference list, that is NOT independent evidence. You must reject such results.\n"
            )
        return (
            "Determine whether the citation corresponds to a real cited work based on search results.\n"
            f"{source_line}"
            f"Citation to verify: Title: {citation.title} Authors: {citation.authors} "
            f"AuthorCount: {citation_author_count} Venue: {citation.venue} Year: {citation.year}\n"
            f"Search evidence:\n{documents}\n"
            "Return JSON:\n"
            "{\n"
            '  "match": true/false,\n'
            '  "matched_result": <number or null>,\n'
            '  "taxonomy": "R1/R2/R3/R4/P1/P2/P3/H1/H2a/H3a/...",\n'
            '  "note": "brief reason"\n'
            "}\n"
            "EXISTENCE CHECK RULES:\n"
            "1. Use ONLY info EXPLICITLY PRESENT in the search result text. Do not infer.\n"
            "2. CIRCULAR REFERENCE: If the search result is a DIFFERENT paper that merely cites the work "
            "in its bibliography, match=false. The result must be the cited work ITSELF.\n"
            "3. Citations can be tweets, blogs, news, GitHub, software packages — not just papers.\n"
            "4. Title must match or directly correspond to the cited work.\n"
            "5. For tweets/social media: if result quotes exact text and attributes to same person, match=true.\n"
            "6. Authors must align. Treat 'et al'/'Others' as omission markers.\n"
            "7. Without 'et al': ALL listed authors must be explicitly present.\n"
            "8. With 'et al': compare only the explicitly listed authors.\n"
            "9. Venue must align when provided. Missing venue acceptable if citation omits it.\n"
            "10. Year must align when provided.\n"
            "\n"
            "TAXONOMY — classify into one subtype:\n"
            "REAL (match=true):\n"
            "  R1: Exact match — all fields verified\n"
            "  R2: Format variant — only case/abbreviation differences\n"
            "  R3: Et al. — author list truncated, listed authors correct\n"
            "  R4: Non-academic source verified — tweet/blog/GitHub/news confirmed to exist\n"
            "POTENTIAL (match=true, with caveats):\n"
            "  P1: Version difference — metadata matches older version\n"
            "  P2: Author name variant — plausible nickname (Katherine→Kate)\n"
            "  P3: Unstable source — source once existed but deleted/moved; indirect evidence found\n"
            "HALLUCINATED (match=false):\n"
            "  H1: Completely fabricated — no trace anywhere\n"
            "  H2a: Word substitution — title words swapped with synonyms\n"
            "  H2b: Title paraphrase — title rephrased\n"
            "  H2c: Title fabrication — semantically different title\n"
            "  H3a: Author addition/deletion — wrong author count (no et al.)\n"
            "  H3b: Author reordering — wrong author order\n"
            "  H3c: Author fabrication — entirely fake names\n"
            "  H4: Venue/year fabrication — wrong venue AND year\n"
            "  H5a: DOI fabrication — DOI points to different paper\n"
            "  H5c: Date error — wrong year\n"
            "  H5d: Venue mismatch — wrong venue only\n"
            "  H5e: Pages/volume fabrication — wrong pages\n"
            "  H6: Sibling paper confusion — metadata mixed from two papers\n"
            "  H7: Non-existent source — URL/tweet/blog doesn't exist\n"
            "\n"
            "Non-academic citations are legitimate. Do NOT reject just because it's not a formal paper.\n"
            "Always provide the most specific taxonomy subtype.\n"
            "Return JSON only."
        )

    def review(
        self,
        citation: CitationRecord,
        candidates: list[CandidateMatch],
        conflicts: list[str],
        proposed_verdict: VerdictLabel,
        secondary_evidence=None,
    ) -> dict[str, Any] | None:
        if not candidates:
            return {
                "label_override": str(proposed_verdict.value),
                "note": "Bedrock strict match=false matched_result=None. No candidate supplied for LLM recheck.",
            }
        candidate = candidates[0]
        documents = self._build_documents(candidates, self.max_candidates)
        if candidate.connector in {"google_search", "web_search"}:
            prompt = self._build_google_existence_prompt(citation, documents, self.source_paper_title)
        else:
            prompt = self._build_strict_match_prompt(citation, documents)

        # Append secondary evidence context if available
        if secondary_evidence:
            evidence_lines = []
            for ev in secondary_evidence:
                status = "EXPLAINED" if ev.evidence_found else "UNEXPLAINED"
                evidence_lines.append(f"  - {ev.field} ({ev.discrepancy_type}): [{status}] {ev.explanation}")
            prompt += (
                "\n\nSECONDARY VERIFICATION EVIDENCE:\n"
                + "\n".join(evidence_lines)
                + "\n\nIMPORTANT: Only upgrade to POTENTIAL_REFERENCE if ALL discrepancies above "
                "are marked EXPLAINED with concrete evidence. If ANY discrepancy is UNEXPLAINED, "
                "maintain FAKE_REFERENCE.\n"
            )

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
        taxonomy = str(parsed.get("taxonomy", "") or "").strip()

        if match:
            return {
                "label_override": "VALID",
                "taxonomy": taxonomy,
                "note": f"Bedrock strict match=true matched_result={matched_result}. {note}".strip(),
            }
        return {
            "label_override": str(proposed_verdict.value),
            "taxonomy": taxonomy,
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
    if args.max_files > 0:
        json_files = json_files[:args.max_files]

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
        f"web_search_provider={cfg.connectors.web_search_provider} "
        f"google_api_key={'<set>' if cfg.connectors.google_api_key else '<empty>'} "
        f"google_cse_id={'<set>' if cfg.connectors.google_cse_id else '<empty>'} "
        f"serpapi_key={'<set>' if cfg.connectors.serpapi_key else '<empty>'} "
        f"tavily_api_key={'<set>' if cfg.connectors.tavily_api_key else '<empty>'} "
        f"enabled_sources={list(cfg.connectors.enabled_sources) if cfg.connectors.enabled_sources else '<all>'}",
        flush=True,
    )

    orchestrator = default_orchestrator(
        cache_path=final_cache_path,
        dblp_mirror_path=final_mirror_path,
        dblp_sqlite_path=Path(final_dblp_sqlite) if final_dblp_sqlite else None,
        enabled_sources=cfg.connectors.enabled_sources,
        acl_anthology_data_dir=cfg.connectors.acl_anthology_data_dir,
        acl_anthology_repo_path=cfg.connectors.acl_anthology_repo_path,
        semantic_scholar_api_key=final_semscholar_key,
        govinfo_api_key=cfg.connectors.govinfo_api_key,
        searxng_base_url=cfg.connectors.searxng_base_url,
        ncbi_api_key=cfg.connectors.ncbi_api_key,
        ncbi_email=cfg.connectors.ncbi_email,
        web_search_provider=cfg.connectors.web_search_provider,
        google_api_key=cfg.connectors.google_api_key,
        google_cse_id=cfg.connectors.google_cse_id,
        serpapi_key=cfg.connectors.serpapi_key,
        tavily_api_key=cfg.connectors.tavily_api_key,
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

    # Build secondary verifier from available connectors
    from packages.connectors.arxiv import ArxivConnector
    from packages.connectors.openalex import OpenAlexConnector
    from packages.connectors.semanticscholar import SemanticScholarConnector
    from packages.core.secondary_verification import SecondaryVerifier

    from packages.connectors.url_direct import URLDirectConnector as _URLDirectConnector

    arxiv_conn = None
    semscholar_conn = None
    openalex_conn = None
    url_direct_conn = None
    for conn in orchestrator.connectors:
        if isinstance(conn, ArxivConnector):
            arxiv_conn = conn
        elif isinstance(conn, SemanticScholarConnector):
            semscholar_conn = conn
        elif isinstance(conn, OpenAlexConnector):
            openalex_conn = conn
        elif isinstance(conn, _URLDirectConnector):
            url_direct_conn = conn

    secondary_verifier = SecondaryVerifier(
        arxiv_connector=arxiv_conn,
        semantic_scholar_connector=semscholar_conn,
        openalex_connector=openalex_conn,
        url_direct_connector=url_direct_conn,
    )
    print(
        f"[setup] secondary_verifier enabled "
        f"arxiv={'yes' if arxiv_conn else 'no'} "
        f"semantic_scholar={'yes' if semscholar_conn else 'no'} "
        f"openalex={'yes' if openalex_conn else 'no'}",
        flush=True,
    )

    # Build multi-agent judges
    # Build cascading 3-agent system
    from packages.core.bedrock_agents import BedrockValidAgent, BedrockPotentialAgent, BedrockHallucinatedAgent
    from packages.core.extractor_agent import BedrockExtractorAgent

    valid_agent = None
    potential_agent = None
    hallucinated_agent = None
    extractor_agent = None
    if cfg.verification_llm.enabled and cfg.verification_llm.provider == "bedrock":
        bedrock_client = _build_bedrock_client(
            region=cfg.verification_llm.bedrock.region,
            bearer_token=cfg.verification_llm.bedrock.bearer_token,
        )
        model_id = str(cfg.verification_llm.bedrock.model_id)
        valid_agent = BedrockValidAgent(bedrock_client, model_id)
        potential_agent = BedrockPotentialAgent(bedrock_client, model_id)
        hallucinated_agent = BedrockHallucinatedAgent(bedrock_client, model_id)
        extractor_agent = BedrockExtractorAgent(bedrock_client, model_id)
        print(f"[setup] cascading 3-agent + extractor enabled (model={model_id})", flush=True)
    else:
        print("[setup] cascading agents disabled (no verification LLM)", flush=True)

    verifier = CitationVerifier(
        orchestrator=orchestrator,
        llm_resolver=llm_resolver,
        secondary_verifier=secondary_verifier,
        valid_agent=valid_agent,
        potential_agent=potential_agent,
        hallucinated_agent=hallucinated_agent,
        extractor_agent=extractor_agent,
    )

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
        # Tell LLM resolver and existence judge which paper we're verifying
        paper_title = src.stem.replace("_", " ")
        if llm_resolver and hasattr(llm_resolver, "source_paper_title"):
            llm_resolver.source_paper_title = paper_title
        for citation_idx, citation in enumerate(citations, start=1):
            print(
                f"[run] [{idx}/{total}] paper={src.stem} citation={citation_idx}/{total_citations} "
                f"id={citation.citation_id}",
                flush=True,
            )
            verdict = verifier.verify_citation(
                citation,
                extraction_quality=quality_map.get(citation.citation_id, ExtractionQuality.UNKNOWN),
                source_paper_title=src.stem.replace("_", " "),
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
                f"verdict={verdict.verdict.value} taxonomy={verdict.taxonomy_subtype} "
                f"title={citation.title[:50]!r} out={out_path}",
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
