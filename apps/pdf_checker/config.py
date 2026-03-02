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
    semantic_scholar_api_key: str | None


@dataclass(frozen=True)
class BedrockAuthConfig:
    region: str
    model_id: str | None
    bearer_token: str | None


@dataclass(frozen=True)
class LocalModelConfig:
    model_path: str | None


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


@dataclass(frozen=True)
class CitationReparseConfig:
    enabled: bool
    model_path: str | None
    max_new_tokens: int
    temperature: float


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
    semantic_scholar_api_key = (
        _optional_str(env, "SEMANTIC_SCHOLAR_API_KEY")
        or _optional_payload_str(payload, ["connectors", "semantic_scholar_api_key"])
    )
    dblp_sqlite_path = (
        _optional_str(env, "CITATION_CHECKER_DBLP_SQLITE_PATH")
        or _optional_payload_str(payload, ["connectors", "dblp_sqlite_path"])
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

    reparse_enabled = _parse_bool(
        _optional_str(env, "CITATION_CHECKER_LLM_REPARSE_ENABLED")
        or _get_nested(payload, ["citation_reparse", "enabled"]),
        default=False,
    )
    reparse_model_path = (
        _optional_str(env, "CITATION_CHECKER_LLM_REPARSE_MODEL_PATH")
        or _optional_payload_str(payload, ["citation_reparse", "model_path"])
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

    return PDFCheckerConfig(
        connectors=ConnectorRuntimeConfig(
            cache_path=Path(cache_path),
            dblp_mirror_path=Path(dblp_mirror_path),
            dblp_sqlite_path=Path(dblp_sqlite_path) if dblp_sqlite_path else None,
            semantic_scholar_api_key=semantic_scholar_api_key,
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
            local=LocalModelConfig(model_path=local_model_path),
        ),
        citation_reparse=CitationReparseConfig(
            enabled=reparse_enabled and bool(reparse_model_path),
            model_path=reparse_model_path,
            max_new_tokens=reparse_max_new_tokens,
            temperature=reparse_temperature,
        ),
    )
