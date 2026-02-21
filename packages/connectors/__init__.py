from __future__ import annotations

import os
from pathlib import Path

from .arxiv import ArxivConnector
from .base import ConnectorOrchestrator, RequestPolicy, SQLiteCache
from .crossref import CrossrefConnector
from .dblp_offline import DblpOfflineConnector
from .dblp_sqlite import DblpSQLiteConnector
from .dblp_online import DBLPOnlineConnector
from .openalex import OpenAlexConnector
from .semanticscholar import SemanticScholarConnector


def default_orchestrator(
    cache_path: str | Path,
    dblp_mirror_path: str | Path,
    semantic_scholar_api_key: str | None = None,
    dblp_sqlite_path: str | Path | None = None,
) -> ConnectorOrchestrator:
    configured_sqlite_path = str(dblp_sqlite_path or "").strip()
    env_sqlite_path = (os.getenv("CITATION_CHECKER_DBLP_SQLITE_PATH") or "").strip()
    final_sqlite_path = configured_sqlite_path or env_sqlite_path

    if final_sqlite_path:
        connectors = [DblpSQLiteConnector(final_sqlite_path)]
    else:
        connectors = [DblpOfflineConnector(dblp_mirror_path)]
    if os.getenv("CITATION_CHECKER_OFFLINE_ONLY", "0") != "1":
        connectors.extend(
            [
                DBLPOnlineConnector(),
                CrossrefConnector(),
                ArxivConnector(),
                OpenAlexConnector(),
                SemanticScholarConnector(api_key=semantic_scholar_api_key),
            ]
        )
    cache = SQLiteCache(cache_path)
    policy = RequestPolicy()
    return ConnectorOrchestrator(connectors=connectors, cache=cache, policy=policy)


__all__ = [
    "ConnectorOrchestrator",
    "RequestPolicy",
    "SQLiteCache",
    "default_orchestrator",
]
