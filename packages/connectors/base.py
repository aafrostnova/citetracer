from __future__ import annotations

import hashlib
import html
import json
import random
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
    # 429 rate-limit retries use a separate, longer schedule because rate-limited
    # endpoints typically need seconds of breathing room, not 0.25s. Server's
    # Retry-After header wins when present.
    rate_limit_max_retries: int = 4       # 5 attempts total (initial + 4 retries)
    rate_limit_backoff_base_s: float = 2.0 # 2s, 4s, 8s, 16s exponential
    rate_limit_backoff_max_s: float = 20.0 # cap per-sleep (caps Retry-After too)
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
        self._lock = threading.Lock()

    def get(self, connector: str) -> float:
        with self._lock:
            return self._scores.get(connector, 1.0)

    def success(self, connector: str, policy: RequestPolicy) -> float:
        with self._lock:
            score = min(1.0, self._scores.get(connector, 1.0) + policy.health_recover_step)
            self._scores[connector] = score
            return score

    def failure(self, connector: str, policy: RequestPolicy) -> float:
        with self._lock:
            score = max(0.0, self._scores.get(connector, 1.0) - policy.health_decay_step)
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
    """Two-tier cache for references that have been verified as VALID (R1/R3).

    Tier 1 (exact): keyed by a hash of ALL citation fields. Hit = the exact
    same citation string (identical formatting) was verified before → return
    the cached verdict as-is, bypassing re-validation.

    Tier 2 (loose): keyed by title + year only. Hit = some form of this paper
    was verified before → use cached matched_candidate but re-run the
    verification flow on the current citation (catches format variants with
    current logic).
    """

    DEFAULT_TTL_S = 60 * 60 * 24 * 30  # 30 days

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            # Detect legacy schema (ref_key PK, no full_key/loose_key columns).
            # If found, drop and recreate — legacy cache is inherently narrower
            # and values can be recomputed on next run.
            cols = [row[1] for row in conn.execute("PRAGMA table_info(verified_references)")]
            if cols and "full_key" not in cols:
                conn.execute("DROP TABLE verified_references")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS verified_references (
                    full_key TEXT PRIMARY KEY,
                    loose_key TEXT NOT NULL,
                    verdict_json TEXT NOT NULL,
                    expires_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vr_loose ON verified_references(loose_key)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_vr_expires ON verified_references(expires_at)"
            )

    @staticmethod
    def _full_key(citation: CitationRecord) -> str:
        """Exact-match key: all user-facing citation fields."""
        authors = [str(a).strip() for a in (citation.authors or [])]
        payload = {
            "title": (citation.title or "").strip(),
            "authors": authors,
            "venue": (citation.venue or "").strip(),
            "year": citation.year,
            "doi": (citation.doi or "").strip().lower(),
            "arxiv_id": (citation.arxiv_id or "").strip().lower(),
            "pages": (citation.pages or "").strip(),
            "volume": (citation.volume or "").strip(),
            "publisher": (citation.publisher or "").strip(),
            "location": (citation.location or "").strip(),
            "url": (citation.url or "").strip(),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return f"full:{digest}"

    @staticmethod
    def _loose_key(citation: CitationRecord) -> str:
        """Loose-match key: title (lowercased) + year."""
        payload = {
            "title": (citation.title or "").strip().lower(),
            "year": citation.year,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return f"loose:{digest}"

    def get_exact(self, citation: CitationRecord) -> dict[str, Any] | None:
        """Tier 1: exact field-for-field match on the full citation."""
        key = self._full_key(citation)
        now = int(time.time())
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT verdict_json, expires_at FROM verified_references WHERE full_key = ?",
                (key,),
            ).fetchone()
            if not row:
                return None
            verdict_json, expires_at = row
            if expires_at < now:
                conn.execute("DELETE FROM verified_references WHERE full_key = ?", (key,))
                return None
            return json.loads(verdict_json)

    def get_loose(self, citation: CitationRecord) -> dict[str, Any] | None:
        """Tier 2: title+year match (any formatting). Caller should re-validate."""
        key = self._loose_key(citation)
        now = int(time.time())
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT verdict_json, expires_at FROM verified_references "
                "WHERE loose_key = ? ORDER BY expires_at DESC LIMIT 1",
                (key,),
            ).fetchone()
            if not row:
                return None
            verdict_json, expires_at = row
            if expires_at < now:
                return None
            return json.loads(verdict_json)

    # Back-compat alias (was the only method before two-tier design)
    def get(self, citation: CitationRecord) -> dict[str, Any] | None:
        return self.get_loose(citation)

    def set(self, citation: CitationRecord, verdict_dict: dict[str, Any], ttl_s: int | None = None) -> None:
        full_key = self._full_key(citation)
        loose_key = self._loose_key(citation)
        expires_at = int(time.time()) + (ttl_s or self.DEFAULT_TTL_S)
        payload = json.dumps(verdict_dict, ensure_ascii=False)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO verified_references(full_key, loose_key, verdict_json, expires_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(full_key) DO UPDATE SET
                    loose_key = excluded.loose_key,
                    verdict_json = excluded.verdict_json,
                    expires_at = excluded.expires_at
                """,
                (full_key, loose_key, payload, expires_at),
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

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        return isinstance(exc, HTTPError) and exc.code == 429

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        """Parse server's Retry-After header (seconds or HTTP-date); return None if absent/unusable."""
        if not isinstance(exc, HTTPError):
            return None
        hdr = exc.headers.get("Retry-After") if exc.headers else None
        if not hdr:
            return None
        hdr = hdr.strip()
        try:
            return max(0.0, float(hdr))
        except ValueError:
            pass
        try:
            from email.utils import parsedate_to_datetime
            from datetime import datetime, timezone
            dt = parsedate_to_datetime(hdr)
            if dt is None:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds())
        except Exception:
            return None

    @classmethod
    def _sleep_for_retry(
        cls, exc: Exception, attempt: int, policy: RequestPolicy,
        url: str = "", connector_name: str = "",
    ) -> None:
        """Sleep an appropriate backoff before the next retry.

        - 429 responses use rate_limit_backoff (honors Retry-After header if present).
        - Other errors use the regular short exponential backoff.

        Always logs 429 events to stderr so callers can see rate-limit pressure
        regardless of which API is being throttled.
        """
        import sys as _sys
        if cls._is_rate_limited(exc):
            server_hint = cls._retry_after_seconds(exc)
            if server_hint is not None:
                delay = min(server_hint, policy.rate_limit_backoff_max_s)
                hint_src = f"Retry-After={server_hint:.1f}s"
            else:
                delay = min(
                    policy.rate_limit_backoff_base_s * (2**attempt),
                    policy.rate_limit_backoff_max_s,
                )
                hint_src = "exp-backoff"
            delay += random.uniform(0.0, 0.5)
            # ALWAYS surface 429 to stderr (don't swallow silently)
            _src = connector_name or cls.__name__
            _url_short = (url[:120] + "...") if len(url) > 120 else url
            print(
                f"[rate-limit] {_src} 429 (attempt {attempt+1}/"
                f"{policy.rate_limit_max_retries+1}): sleeping {delay:.1f}s "
                f"[{hint_src}] url={_url_short}",
                file=_sys.stderr, flush=True,
            )
        else:
            delay = policy.backoff_base_s * (2**attempt) + random.uniform(0.0, 0.05)
        time.sleep(delay)

    @staticmethod
    def _max_retries_for(exc: Exception, policy: RequestPolicy) -> int:
        return policy.rate_limit_max_retries if BaseConnector._is_rate_limited(exc) else policy.max_retries

    def _log_final_failure(self, url: str, exc: Exception) -> None:
        """Print a stderr line when all retries are exhausted — especially
        important for 429 so callers see sustained rate-limit exhaustion."""
        import sys as _sys
        if self._is_rate_limited(exc):
            _url_short = (url[:120] + "...") if len(url) > 120 else url
            print(
                f"[rate-limit] {self.name} 429 FINAL: all retries exhausted, "
                f"giving up. url={_url_short}",
                file=_sys.stderr, flush=True,
            )

    def _request_json(self, url: str, params: dict[str, Any], policy: RequestPolicy, headers: dict[str, str] | None = None) -> dict[str, Any]:
        query = urlencode({k: v for k, v in params.items() if v is not None and v != ""})
        target = f"{url}?{query}" if query else url
        headers = headers or {}
        request = Request(target, headers={"User-Agent": "citation-checker/1.0", **headers})
        last_error: Exception | None = None
        max_attempts = max(policy.max_retries, policy.rate_limit_max_retries) + 1
        for attempt in range(max_attempts):
            try:
                with urlopen(request, timeout=policy.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self._max_retries_for(exc, policy):
                    break
                self._sleep_for_retry(exc, attempt, policy, url=target, connector_name=self.name)
        self._log_final_failure(target, last_error)
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
        max_attempts = max(policy.max_retries, policy.rate_limit_max_retries) + 1
        for attempt in range(max_attempts):
            try:
                with urlopen(request, timeout=policy.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self._max_retries_for(exc, policy):
                    break
                self._sleep_for_retry(exc, attempt, policy, url=url, connector_name=self.name)
        self._log_final_failure(url, last_error)
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
        max_attempts = max(policy.max_retries, policy.rate_limit_max_retries) + 1
        for attempt in range(max_attempts):
            try:
                with urlopen(request, timeout=policy.timeout_s) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    return response.read().decode(charset, errors="replace")
            except (HTTPError, URLError, TimeoutError) as exc:
                last_error = exc
                if attempt >= self._max_retries_for(exc, policy):
                    break
                self._sleep_for_retry(exc, attempt, policy, url=target, connector_name=self.name)
        self._log_final_failure(target, last_error)
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
        max_workers: int = 8,
    ) -> None:
        self.connectors = connectors
        self.cache = cache
        self.policy = policy or RequestPolicy()
        self.source_health = source_health or SourceHealth()
        self.max_workers = max(1, max_workers)

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
        # Slice connector list up front so max_connectors is honored before parallel dispatch
        connectors_to_query = self.connectors
        if max_connectors is not None:
            connectors_to_query = list(self.connectors)[:max_connectors]

        def _query_one(connector: BaseConnector) -> ConnectorResult:
            started = time.perf_counter()
            error: str | None = None
            records: list[dict[str, Any]] = []
            try:
                records = connector.search(citation, self.policy)
                health = self.source_health.success(connector.name, self.policy)
            except Exception as exc:  # noqa: BLE001 - connector failures should not crash the paper run
                error = str(exc)
                health = self.source_health.failure(connector.name, self.policy)

            latency_ms = (time.perf_counter() - started) * 1000
            return ConnectorResult(
                connector=connector.name,
                records=records,
                latency_ms=latency_ms,
                cache_hit=False,
                source_health=health,
                error=error,
            )

        # Parallel dispatch with order preserved by pool.map
        worker_count = min(self.max_workers, len(connectors_to_query)) or 1
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            return list(pool.map(_query_one, connectors_to_query))
