from __future__ import annotations

import json
import os
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ConnectorRuntimeConfig:
    cache_path: Path
    dblp_mirror_path: Path
    dblp_sqlite_path: Path | None
    enabled_sources: tuple[str, ...] | None
    acl_anthology_data_dir: Path | None
    acl_anthology_repo_path: Path | None
    semantic_scholar_api_key: str | None
    govinfo_api_key: str | None
    searxng_base_url: str | None
    ncbi_api_key: str | None
    ncbi_email: str | None
    web_search_provider: str
    google_api_key: str | None
    google_cse_id: str | None
    serpapi_key: str | None
    tavily_api_key: str | None


@dataclass(frozen=True)
class BedrockAuthConfig:
    region: str
    model_id: str | None
    bearer_token: str | None


@dataclass(frozen=True)
class LocalModelConfig:
    model_path: str | None
    inference_backend: str


@dataclass(frozen=True)
class PDFEntryExtractionConfig:
    mode: str
    provider: str
    chunk_chars: int
    bedrock: BedrockAuthConfig
    local: LocalModelConfig


@dataclass(frozen=True)
class PDFCheckerConfig:
    connectors: ConnectorRuntimeConfig
    entry_extraction: PDFEntryExtractionConfig
    citation_reparse: "CitationReparseConfig"
    verification_llm: "VerificationLLMConfig"
    ocr_llm_extract: "OCRLLMExtractConfig"


@dataclass(frozen=True)
class CitationReparseConfig:
    enabled: bool
    provider: str
    model_path: str | None
    max_new_tokens: int
    temperature: float
    bedrock: BedrockAuthConfig


@dataclass(frozen=True)
class VerificationLLMConfig:
    enabled: bool
    provider: str
    max_candidates: int
    bedrock: BedrockAuthConfig


@dataclass(frozen=True)
class OCRLLMExtractConfig:
    enabled: bool
    max_new_tokens: int
    temperature: float
    force_all_entries: bool
    entry_extraction: PDFEntryExtractionConfig
    bedrock: BedrockAuthConfig


def _get_nested(payload: Mapping[str, Any], keys: list[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, MappingABC):
            return None
        current = current.get(key)
    return current


def _optional_payload_str(payload: Mapping[str, Any], keys: list[str]) -> str | None:
    value = _get_nested(payload, keys)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_str(env: Mapping[str, str], key: str) -> str | None:
    value = env.get(key, "").strip()
    return value or None


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_entry_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode in {"heuristic", "model"}:
        return mode
    return "heuristic"


def _parse_chunk_chars(value: Any) -> int:
    try:
        return max(2000, int(str(value).strip()))
    except (TypeError, ValueError):
        return 12000


def _parse_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _parse_temperature(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def _parse_model_provider(value: Any) -> str:
    provider = str(value).strip().lower()
    if provider in {"bedrock", "local"}:
        return provider
    return "bedrock"


def _parse_local_inference_backend(value: Any) -> str:
    backend = str(value).strip().lower()
    if backend in {"hf", "vllm"}:
        return backend
    return "hf"


def _parse_verification_provider(value: Any) -> str:
    provider = str(value).strip().lower()
    if provider in {"bedrock"}:
        return provider
    return "bedrock"


def _parse_reparse_provider(value: Any) -> str:
    provider = str(value).strip().lower()
    if provider in {"local", "bedrock"}:
        return provider
    return "local"


def _parse_web_search_provider(value: Any) -> str | None:
    provider = str(value).strip().lower().replace("-", "_")
    if provider in {"", "auto"}:
        return "auto"
    if provider in {"serpapi", "tavily", "google_cse"}:
        return provider
    if provider in {"google", "google_custom_search", "cse"}:
        return "google_cse"
    return None


def _resolve_web_search_provider(
    explicit_provider: Any,
    google_api_key: str | None,
    google_cse_id: str | None,
    serpapi_key: str | None,
    tavily_api_key: str | None,
) -> str:
    parsed = _parse_web_search_provider(explicit_provider)
    if parsed is not None:
        return parsed
    if google_api_key and google_cse_id:
        return "google_cse"
    if serpapi_key:
        return "serpapi"
    if tavily_api_key:
        return "tavily"
    return "auto"


def _parse_enabled_sources(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    items: list[str] = []
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        items = [str(part).strip() for part in value if str(part).strip()]
    else:
        return None
    if not items:
        return None
    return tuple(dict.fromkeys(items))


def _load_json_payload(env: Mapping[str, str]) -> Mapping[str, Any]:
    config_path = _optional_str(env, "CITATION_CHECKER_CONFIG_PATH") or "config.json"
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in config file {path}: {exc}") from exc

    if not isinstance(payload, MappingABC):
        raise RuntimeError(f"Config file {path} must contain a JSON object.")
    return payload


def _resolve_entry_mode(
    explicit_mode: str | None,
    bearer_token: str | None,
    local_model_path: str | None,
) -> str:
    if explicit_mode is not None:
        return _parse_entry_mode(explicit_mode)
    if bearer_token or local_model_path:
        return "model"
    return "heuristic"


def _resolve_model_id(explicit_model_id: str | None, bearer_token: str | None) -> str | None:
    explicit = (explicit_model_id or "").strip()
    if explicit:
        return explicit
    if bearer_token:
        return "qwen.qwen3-vl-235b-a22b"
    return None


def _resolve_model_provider(
    explicit_provider: str | None,
    bearer_token: str | None,
    local_model_path: str | None,
) -> str:
    if explicit_provider:
        return _parse_model_provider(explicit_provider)
    if local_model_path:
        return "local"
    return "bedrock" if bearer_token else "bedrock"


def load_pdf_checker_config(env: Mapping[str, str] | None = None) -> PDFCheckerConfig:
    env = env or os.environ
    payload = _load_json_payload(env)

    cache_path = (
        _optional_str(env, "CITATION_CHECKER_CACHE_PATH")
        or _optional_payload_str(payload, ["connectors", "cache_path"])
        or "data/cache/connector_cache.sqlite"
    )
    dblp_mirror_path = (
        _optional_str(env, "CITATION_CHECKER_DBLP_MIRROR_PATH")
        or _optional_payload_str(payload, ["connectors", "dblp_mirror_path"])
        or "data/cache/dblp_mirror.jsonl"
    )
    acl_anthology_data_dir = (
        _optional_str(env, "CITATION_CHECKER_ACL_ANTHOLOGY_DATA_DIR")
        or _optional_payload_str(payload, ["connectors", "acl_anthology_data_dir"])
    )
    acl_anthology_repo_path = (
        _optional_str(env, "CITATION_CHECKER_ACL_ANTHOLOGY_REPO_PATH")
        or _optional_payload_str(payload, ["connectors", "acl_anthology_repo_path"])
    )
    semantic_scholar_api_key = (
        _optional_str(env, "SEMANTIC_SCHOLAR_API_KEY")
        or _optional_payload_str(payload, ["connectors", "semantic_scholar_api_key"])
    )
    govinfo_api_key = (
        _optional_str(env, "GOVINFO_API_KEY")
        or _optional_str(env, "CITATION_CHECKER_GOVINFO_API_KEY")
        or _optional_payload_str(payload, ["connectors", "govinfo_api_key"])
    )
    searxng_base_url = (
        _optional_str(env, "CITATION_CHECKER_SEARXNG_BASE_URL")
        or _optional_payload_str(payload, ["connectors", "searxng_base_url"])
    )
    ncbi_api_key = (
        _optional_str(env, "NCBI_API_KEY")
        or _optional_str(env, "CITATION_CHECKER_NCBI_API_KEY")
        or _optional_payload_str(payload, ["connectors", "ncbi_api_key"])
    )
    ncbi_email = (
        _optional_str(env, "NCBI_EMAIL")
        or _optional_str(env, "CITATION_CHECKER_NCBI_EMAIL")
        or _optional_payload_str(payload, ["connectors", "ncbi_email"])
    )
    tavily_api_key = (
        _optional_str(env, "TAVILY_API_KEY")
        or _optional_str(env, "CITATION_CHECKER_TAVILY_API_KEY")
        or _optional_payload_str(payload, ["connectors", "tavily_api_key"])
    )
    google_api_key = (
        _optional_str(env, "GOOGLE_API_KEY")
        or _optional_str(env, "CITATION_CHECKER_GOOGLE_API_KEY")
        or _optional_payload_str(payload, ["connectors", "google_api_key"])
    )
    google_cse_id = (
        _optional_str(env, "GOOGLE_CSE_ID")
        or _optional_str(env, "CITATION_CHECKER_GOOGLE_CSE_ID")
        or _optional_payload_str(payload, ["connectors", "google_cse_id"])
    )
    serpapi_key = (
        _optional_str(env, "SERPAPI_KEY")
        or _optional_str(env, "CITATION_CHECKER_SERPAPI_KEY")
        or _optional_payload_str(payload, ["connectors", "serpapi_key"])
    )
    web_search_provider = _resolve_web_search_provider(
        _optional_str(env, "CITATION_CHECKER_WEB_SEARCH_PROVIDER")
        or _optional_payload_str(payload, ["connectors", "web_search_provider"]),
        google_api_key=google_api_key,
        google_cse_id=google_cse_id,
        serpapi_key=serpapi_key,
        tavily_api_key=tavily_api_key,
    )
    dblp_sqlite_path = (
        _optional_str(env, "CITATION_CHECKER_DBLP_SQLITE_PATH")
        or _optional_payload_str(payload, ["connectors", "dblp_sqlite_path"])
    )
    enabled_sources = _parse_enabled_sources(
        _optional_str(env, "CITATION_CHECKER_ENABLED_SOURCES")
        or _get_nested(payload, ["connectors", "enabled_sources"])
    )

    bearer_token = (
        _optional_str(env, "AWS_BEARER_TOKEN_BEDROCK")
        or _optional_payload_str(payload, ["entry_extraction", "bedrock", "bearer_token"])
    )
    explicit_mode = (
        _optional_str(env, "CITATION_CHECKER_PDF_ENTRY_EXTRACTION")
        or _optional_payload_str(payload, ["entry_extraction", "mode"])
    )
    explicit_provider = (
        _optional_str(env, "CITATION_CHECKER_PDF_MODEL_PROVIDER")
        or _optional_payload_str(payload, ["entry_extraction", "provider"])
    )
    chunk_chars_value = (
        _optional_str(env, "CITATION_CHECKER_PDF_ENTRY_CHUNK_CHARS")
        or _get_nested(payload, ["entry_extraction", "chunk_chars"])
        or "12000"
    )
    region = (
        _optional_str(env, "AWS_REGION")
        or _optional_payload_str(payload, ["entry_extraction", "bedrock", "region"])
        or "us-east-1"
    )
    explicit_model_id = (
        _optional_str(env, "CITATION_CHECKER_BEDROCK_MODEL_ID")
        or _optional_payload_str(payload, ["entry_extraction", "bedrock", "model_id"])
    )
    local_model_path = (
        _optional_str(env, "CITATION_CHECKER_LOCAL_MODEL_PATH")
        or _optional_payload_str(payload, ["entry_extraction", "local", "model_path"])
    )
    local_inference_backend = _parse_local_inference_backend(
        _optional_str(env, "CITATION_CHECKER_LOCAL_INFERENCE_BACKEND")
        or _optional_payload_str(payload, ["entry_extraction", "local", "inference_backend"])
        or "hf"
    )

    reparse_enabled = _parse_bool(
        _optional_str(env, "CITATION_CHECKER_LLM_REPARSE_ENABLED")
        or _get_nested(payload, ["citation_reparse", "enabled"]),
        default=False,
    )
    reparse_model_path = (
        _optional_str(env, "CITATION_CHECKER_LLM_REPARSE_MODEL_PATH")
        or _optional_payload_str(payload, ["citation_reparse", "model_path"])
    )
    reparse_provider = _parse_reparse_provider(
        _optional_str(env, "CITATION_CHECKER_LLM_REPARSE_PROVIDER")
        or _optional_payload_str(payload, ["citation_reparse", "provider"])
        or "local"
    )
    reparse_max_new_tokens = _parse_positive_int(
        _optional_str(env, "CITATION_CHECKER_LLM_REPARSE_MAX_NEW_TOKENS")
        or _get_nested(payload, ["citation_reparse", "max_new_tokens"])
        or 32768,
        default=32768,
    )
    reparse_temperature = _parse_temperature(
        _optional_str(env, "CITATION_CHECKER_LLM_REPARSE_TEMPERATURE")
        or _get_nested(payload, ["citation_reparse", "temperature"])
        or 0.0,
        default=0.0,
    )
    reparse_bedrock_region = (
        _optional_str(env, "CITATION_CHECKER_LLM_REPARSE_BEDROCK_REGION")
        or _optional_payload_str(payload, ["citation_reparse", "bedrock", "region"])
        or region
    )
    reparse_bedrock_model_id = (
        _optional_str(env, "CITATION_CHECKER_LLM_REPARSE_BEDROCK_MODEL_ID")
        or _optional_payload_str(payload, ["citation_reparse", "bedrock", "model_id"])
    )
    reparse_bedrock_bearer_token = (
        _optional_str(env, "CITATION_CHECKER_LLM_REPARSE_BEDROCK_BEARER_TOKEN")
        or _optional_payload_str(payload, ["citation_reparse", "bedrock", "bearer_token"])
        or bearer_token
    )
    verification_enabled = _parse_bool(
        _optional_str(env, "CITATION_CHECKER_VERIFICATION_LLM_ENABLED")
        or _get_nested(payload, ["verification_llm", "enabled"]),
        default=False,
    )
    verification_provider = _parse_verification_provider(
        _optional_str(env, "CITATION_CHECKER_VERIFICATION_LLM_PROVIDER")
        or _optional_payload_str(payload, ["verification_llm", "provider"])
        or "bedrock"
    )
    verification_max_candidates = _parse_positive_int(
        _optional_str(env, "CITATION_CHECKER_VERIFICATION_LLM_MAX_CANDIDATES")
        or _get_nested(payload, ["verification_llm", "max_candidates"])
        or 5,
        default=5,
    )
    verification_region = (
        _optional_str(env, "CITATION_CHECKER_VERIFICATION_BEDROCK_REGION")
        or _optional_payload_str(payload, ["verification_llm", "bedrock", "region"])
        or region
    )
    verification_model_id = (
        _optional_str(env, "CITATION_CHECKER_VERIFICATION_BEDROCK_MODEL_ID")
        or _optional_payload_str(payload, ["verification_llm", "bedrock", "model_id"])
    )
    verification_bearer_token = (
        _optional_str(env, "CITATION_CHECKER_VERIFICATION_BEDROCK_BEARER_TOKEN")
        or _optional_payload_str(payload, ["verification_llm", "bedrock", "bearer_token"])
        or bearer_token
    )
    ocr_llm_extract_enabled = _parse_bool(
        _optional_str(env, "CITATION_CHECKER_OCR_LLM_EXTRACT_ENABLED")
        or _get_nested(payload, ["ocr_llm_extract", "enabled"]),
        default=False,
    )
    ocr_llm_extract_max_new_tokens = _parse_positive_int(
        _optional_str(env, "CITATION_CHECKER_OCR_LLM_EXTRACT_MAX_NEW_TOKENS")
        or _get_nested(payload, ["ocr_llm_extract", "max_new_tokens"])
        or reparse_max_new_tokens,
        default=reparse_max_new_tokens,
    )
    ocr_llm_extract_temperature = _parse_temperature(
        _optional_str(env, "CITATION_CHECKER_OCR_LLM_EXTRACT_TEMPERATURE")
        or _get_nested(payload, ["ocr_llm_extract", "temperature"])
        or reparse_temperature,
        default=reparse_temperature,
    )
    ocr_llm_extract_force_all_entries = _parse_bool(
        _optional_str(env, "CITATION_CHECKER_OCR_LLM_EXTRACT_FORCE_ALL_ENTRIES")
        or _get_nested(payload, ["ocr_llm_extract", "force_all_entries"]),
        default=True,
    )
    ocr_llm_extract_region = (
        _optional_str(env, "CITATION_CHECKER_OCR_LLM_EXTRACT_BEDROCK_REGION")
        or _optional_payload_str(payload, ["ocr_llm_extract", "bedrock", "region"])
        or reparse_bedrock_region
    )
    ocr_llm_extract_model_id = (
        _optional_str(env, "CITATION_CHECKER_OCR_LLM_EXTRACT_BEDROCK_MODEL_ID")
        or _optional_payload_str(payload, ["ocr_llm_extract", "bedrock", "model_id"])
        or reparse_bedrock_model_id
    )
    ocr_llm_extract_bearer_token = (
        _optional_str(env, "CITATION_CHECKER_OCR_LLM_EXTRACT_BEDROCK_BEARER_TOKEN")
        or _optional_payload_str(payload, ["ocr_llm_extract", "bedrock", "bearer_token"])
        or reparse_bedrock_bearer_token
    )
    ocr_entry_mode = _parse_entry_mode(
        _optional_str(env, "CITATION_CHECKER_OCR_LLM_EXTRACT_ENTRY_MODE")
        or _optional_payload_str(payload, ["ocr_llm_extract", "entry_extraction", "mode"])
        or _resolve_entry_mode(explicit_mode, bearer_token, local_model_path)
    )
    ocr_entry_provider = _parse_model_provider(
        _optional_str(env, "CITATION_CHECKER_OCR_LLM_EXTRACT_ENTRY_PROVIDER")
        or _optional_payload_str(payload, ["ocr_llm_extract", "entry_extraction", "provider"])
        or _resolve_model_provider(explicit_provider, bearer_token, local_model_path)
    )
    ocr_entry_chunk_chars = _parse_chunk_chars(
        _optional_str(env, "CITATION_CHECKER_OCR_LLM_EXTRACT_ENTRY_CHUNK_CHARS")
        or _get_nested(payload, ["ocr_llm_extract", "entry_extraction", "chunk_chars"])
        or chunk_chars_value
    )
    ocr_entry_local_model_path = (
        _optional_str(env, "CITATION_CHECKER_OCR_LLM_EXTRACT_LOCAL_MODEL_PATH")
        or _optional_payload_str(payload, ["ocr_llm_extract", "entry_extraction", "local", "model_path"])
        or local_model_path
    )
    ocr_entry_local_inference_backend = _parse_local_inference_backend(
        _optional_str(env, "CITATION_CHECKER_OCR_LLM_EXTRACT_LOCAL_INFERENCE_BACKEND")
        or _optional_payload_str(payload, ["ocr_llm_extract", "entry_extraction", "local", "inference_backend"])
        or local_inference_backend
        or "hf"
    )
    ocr_entry_region = (
        _optional_str(env, "CITATION_CHECKER_OCR_LLM_EXTRACT_ENTRY_BEDROCK_REGION")
        or _optional_payload_str(payload, ["ocr_llm_extract", "entry_extraction", "bedrock", "region"])
        or region
    )
    ocr_entry_model_id = (
        _optional_str(env, "CITATION_CHECKER_OCR_LLM_EXTRACT_ENTRY_BEDROCK_MODEL_ID")
        or _optional_payload_str(payload, ["ocr_llm_extract", "entry_extraction", "bedrock", "model_id"])
        or _resolve_model_id(explicit_model_id, bearer_token)
    )
    ocr_entry_bearer_token = (
        _optional_str(env, "CITATION_CHECKER_OCR_LLM_EXTRACT_ENTRY_BEDROCK_BEARER_TOKEN")
        or _optional_payload_str(payload, ["ocr_llm_extract", "entry_extraction", "bedrock", "bearer_token"])
        or bearer_token
    )

    return PDFCheckerConfig(
        connectors=ConnectorRuntimeConfig(
            cache_path=Path(cache_path),
            dblp_mirror_path=Path(dblp_mirror_path),
            dblp_sqlite_path=Path(dblp_sqlite_path) if dblp_sqlite_path else None,
            enabled_sources=enabled_sources,
            acl_anthology_data_dir=Path(acl_anthology_data_dir) if acl_anthology_data_dir else None,
            acl_anthology_repo_path=Path(acl_anthology_repo_path) if acl_anthology_repo_path else None,
            semantic_scholar_api_key=semantic_scholar_api_key,
            govinfo_api_key=govinfo_api_key,
            searxng_base_url=searxng_base_url,
            ncbi_api_key=ncbi_api_key,
            ncbi_email=ncbi_email,
            web_search_provider=web_search_provider,
            google_api_key=google_api_key,
            google_cse_id=google_cse_id,
            serpapi_key=serpapi_key,
            tavily_api_key=tavily_api_key,
        ),
        entry_extraction=PDFEntryExtractionConfig(
            mode=_resolve_entry_mode(explicit_mode, bearer_token, local_model_path),
            provider=_resolve_model_provider(explicit_provider, bearer_token, local_model_path),
            chunk_chars=_parse_chunk_chars(chunk_chars_value),
            bedrock=BedrockAuthConfig(
                region=region,
                model_id=_resolve_model_id(explicit_model_id, bearer_token),
                bearer_token=bearer_token,
            ),
            local=LocalModelConfig(
                model_path=local_model_path,
                inference_backend=local_inference_backend,
            ),
        ),
        citation_reparse=CitationReparseConfig(
            enabled=(
                reparse_enabled
                and (
                    bool(reparse_model_path)
                    if reparse_provider == "local"
                    else bool(reparse_bedrock_model_id)
                )
            ),
            provider=reparse_provider,
            model_path=reparse_model_path,
            max_new_tokens=reparse_max_new_tokens,
            temperature=reparse_temperature,
            bedrock=BedrockAuthConfig(
                region=reparse_bedrock_region,
                model_id=reparse_bedrock_model_id,
                bearer_token=reparse_bedrock_bearer_token,
            ),
        ),
        verification_llm=VerificationLLMConfig(
            enabled=verification_enabled and bool(verification_model_id),
            provider=verification_provider,
            max_candidates=max(1, verification_max_candidates),
            bedrock=BedrockAuthConfig(
                region=verification_region,
                model_id=verification_model_id,
                bearer_token=verification_bearer_token,
            ),
        ),
        ocr_llm_extract=OCRLLMExtractConfig(
            enabled=ocr_llm_extract_enabled and bool(ocr_llm_extract_model_id),
            max_new_tokens=ocr_llm_extract_max_new_tokens,
            temperature=ocr_llm_extract_temperature,
            force_all_entries=ocr_llm_extract_force_all_entries,
            entry_extraction=PDFEntryExtractionConfig(
                mode=ocr_entry_mode,
                provider=ocr_entry_provider,
                chunk_chars=ocr_entry_chunk_chars,
                bedrock=BedrockAuthConfig(
                    region=ocr_entry_region,
                    model_id=ocr_entry_model_id,
                    bearer_token=ocr_entry_bearer_token,
                ),
                local=LocalModelConfig(
                    model_path=ocr_entry_local_model_path,
                    inference_backend=ocr_entry_local_inference_backend,
                ),
            ),
            bedrock=BedrockAuthConfig(
                region=ocr_llm_extract_region,
                model_id=ocr_llm_extract_model_id,
                bearer_token=ocr_llm_extract_bearer_token,
            ),
        ),
    )
