from __future__ import annotations

import hashlib
import html
import json
import random
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from packages.core.adjudicate import AdjudicationPolicy
from packages.core.matching import build_candidate_match
from packages.core.models import CitationRecord


@dataclass
class RequestPolicy:
    timeout_s: float = 6.0
    max_retries: int = 2
    backoff_base_s: float = 0.25
    health_recover_step: float = 0.03
    health_decay_step: float = 0.15


@dataclass
class ConnectorResult:
    connector: str
    records: list[dict[str, Any]]
    latency_ms: float
    cache_hit: bool
    source_health: float
    error: str | None = None


class SourceHealth:
    def __init__(self) -> None:
        self._scores: dict[str, float] = {}

    def get(self, connector: str) -> float:
        return self._scores.get(connector, 1.0)

    def success(self, connector: str, policy: RequestPolicy) -> float:
        score = min(1.0, self.get(connector) + policy.health_recover_step)
        self._scores[connector] = score
        return score

    def failure(self, connector: str, policy: RequestPolicy) -> float:
        score = max(0.0, self.get(connector) - policy.health_decay_step)
        self._scores[connector] = score
        return score


class SQLiteCache:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS connector_cache (
                    cache_key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    expires_at INTEGER NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_connector_cache_expires ON connector_cache(expires_at)")

    def get(self, cache_key: str) -> list[dict[str, Any]] | None:
        now = int(time.time())
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT payload, expires_at FROM connector_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if not row:
                return None
            payload, expires_at = row
            if expires_at < now:
                conn.execute("DELETE FROM connector_cache WHERE cache_key = ?", (cache_key,))
                return None
            return json.loads(payload)

    def set(self, cache_key: str, records: list[dict[str, Any]], ttl_s: int) -> None:
        expires_at = int(time.time()) + max(ttl_s, 1)
        payload = json.dumps(records)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO connector_cache(cache_key, payload, expires_at)
                VALUES(?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload = excluded.payload,
                    expires_at = excluded.expires_at
                """,
                (cache_key, payload, expires_at),
            )


class VerifiedReferenceCache:
    """Cache for references that have been verified as VALID (R1/R3/R4).

    This caches the *reference* (not individual candidates). When the same
    reference appears in another paper, the cached verdict is returned
    immediately, skipping the entire verification pipeline.
    """

    # Verified references are unlikely to become invalid; use a long TTL.
    DEFAULT_TTL_S = 60 * 60 * 24 * 30  # 30 days

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS verified_references (
                    ref_key TEXT PRIMARY KEY,
                    verdict_json TEXT NOT NULL,
                    expires_at INTEGER NOT NULL
                )
                """
            )

    @staticmethod
    def _ref_key(citation: CitationRecord) -> str:
        """Build a cache key from title + year."""
        payload = {
            "title": (citation.title or "").strip().lower(),
            "year": citation.year,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return f"ref:{digest}"

    def get(self, citation: CitationRecord) -> dict[str, Any] | None:
        key = self._ref_key(citation)
        now = int(time.time())
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT verdict_json, expires_at FROM verified_references WHERE ref_key = ?",
                (key,),
            ).fetchone()
            if not row:
                return None
            verdict_json, expires_at = row
            if expires_at < now:
                conn.execute("DELETE FROM verified_references WHERE ref_key = ?", (key,))
                return None
            return json.loads(verdict_json)

    def set(self, citation: CitationRecord, verdict_dict: dict[str, Any], ttl_s: int | None = None) -> None:
        key = self._ref_key(citation)
        expires_at = int(time.time()) + (ttl_s or self.DEFAULT_TTL_S)
        payload = json.dumps(verdict_dict, ensure_ascii=False)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO verified_references(ref_key, verdict_json, expires_at)
                VALUES(?, ?, ?)
                ON CONFLICT(ref_key) DO UPDATE SET
                    verdict_json = excluded.verdict_json,
                    expires_at = excluded.expires_at
                """,
                (key, payload, expires_at),
            )


def cache_key_for(connector_name: str, citation: CitationRecord) -> str:
    payload = {
        "connector": connector_name,
        "title": citation.title,
        "authors": citation.authors,
        "venue": citation.venue,
        "year": citation.year,
        "doi": citation.doi,
        "arxiv_id": citation.arxiv_id,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"{connector_name}:{digest}"


class BaseConnector:
    name: str = "base"
    ttl_s: int = 60 * 60 * 24
    cache_identity: str | None = None

    def search(self, citation: CitationRecord, policy: RequestPolicy) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _request_json(self, url: str, params: dict[str, Any], policy: RequestPolicy, headers: dict[str, str] | None = None) -> dict[str, Any]:
        query = urlencode({k: v for k, v in params.items() if v is not None and v != ""})
        target = f"{url}?{query}" if query else url
        headers = headers or {}
        request = Request(target, headers={"User-Agent": "citation-checker/1.0", **headers})
        last_error: Exception | None = None
        for attempt in range(policy.max_retries + 1):
            try:
                with urlopen(request, timeout=policy.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= policy.max_retries:
                    break
                time.sleep(policy.backoff_base_s * (2**attempt) + random.uniform(0.0, 0.05))
        raise RuntimeError(f"{self.name} request failed: {last_error}")

    def _request_json_post(
        self,
        url: str,
        payload: dict[str, Any],
        policy: RequestPolicy,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        headers = headers or {}
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"User-Agent": "citation-checker/1.0", "Content-Type": "application/json", **headers},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(policy.max_retries + 1):
            try:
                with urlopen(request, timeout=policy.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= policy.max_retries:
                    break
                time.sleep(policy.backoff_base_s * (2**attempt) + random.uniform(0.0, 0.05))
        raise RuntimeError(f"{self.name} request failed: {last_error}")

    def _request_text(
        self,
        url: str,
        params: dict[str, Any],
        policy: RequestPolicy,
        headers: dict[str, str] | None = None,
    ) -> str:
        query = urlencode({k: v for k, v in params.items() if v is not None and v != ""})
        target = f"{url}?{query}" if query else url
        headers = headers or {}
        request = Request(target, headers={"User-Agent": "citation-checker/1.0", **headers})
        last_error: Exception | None = None
        for attempt in range(policy.max_retries + 1):
            try:
                with urlopen(request, timeout=policy.timeout_s) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    return response.read().decode(charset, errors="replace")
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                if attempt >= policy.max_retries:
                    break
                time.sleep(policy.backoff_base_s * (2**attempt) + random.uniform(0.0, 0.05))
        raise RuntimeError(f"{self.name} request failed: {last_error}")

    @staticmethod
    def _clean_text(value: str) -> str:
        text = html.unescape(str(value or ""))
        text = re.sub(r"<[^>]+>", " ", text)
        return " ".join(text.split()).strip()


class ConnectorOrchestrator:
    def __init__(
        self,
        connectors: list[BaseConnector],
        cache: SQLiteCache,
        policy: RequestPolicy | None = None,
        source_health: SourceHealth | None = None,
    ) -> None:
        self.connectors = connectors
        self.cache = cache
        self.policy = policy or RequestPolicy()
        self.source_health = source_health or SourceHealth()

    @staticmethod
    def _norm_text(value: str) -> str:
        return " ".join(str(value or "").lower().split())

    @classmethod
    def _has_exact_identifier_hit(cls, citation: CitationRecord, records: list[dict[str, Any]]) -> bool:
        doi = str(citation.doi or "").strip().lower()
        arxiv_id = str(citation.arxiv_id or "").strip().lower()
        for record in records:
            if doi and str(record.get("doi", "") or "").strip().lower() == doi:
                return True
            if arxiv_id and str(record.get("arxiv_id", "") or "").strip().lower() == arxiv_id:
                return True
        return False

    @classmethod
    def _has_exact_title_hit(cls, citation: CitationRecord, records: list[dict[str, Any]]) -> bool:
        title = cls._norm_text(citation.title)
        if not title:
            return False
        return any(cls._norm_text(record.get("title", "")) == title for record in records)

    def query(
        self,
        citation: CitationRecord,
        max_connectors: int | None = None,
    ) -> list[ConnectorResult]:
        results: list[ConnectorResult] = []
        for connector in self.connectors:
            if max_connectors is not None and len(results) >= max_connectors:
                break
            started = time.perf_counter()
            cache_identity = getattr(connector, "cache_identity", None) or connector.name
            key = cache_key_for(cache_identity, citation)
            cached = self.cache.get(key)
            if cached is not None:
                latency_ms = (time.perf_counter() - started) * 1000
                results.append(
                    ConnectorResult(
                        connector=connector.name,
                        records=cached,
                        latency_ms=latency_ms,
                        cache_hit=True,
                        source_health=self.source_health.get(connector.name),
                        error=None,
                    )
                )
                continue

            error: str | None = None
            records: list[dict[str, Any]] = []
            try:
                records = connector.search(citation, self.policy)
                if records:
                    self.cache.set(key, records, connector.ttl_s)
                health = self.source_health.success(connector.name, self.policy)
            except Exception as exc:  # noqa: BLE001 - connector failures should not crash the paper run
                error = str(exc)
                health = self.source_health.failure(connector.name, self.policy)

            latency_ms = (time.perf_counter() - started) * 1000
            results.append(
                ConnectorResult(
                    connector=connector.name,
                    records=records,
                    latency_ms=latency_ms,
                    cache_hit=False,
                    source_health=health,
                    error=error,
                )
            )
        return results
