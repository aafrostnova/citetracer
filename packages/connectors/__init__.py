from __future__ import annotations

import os
from pathlib import Path

from .acl_anthology import ACLAnthologyConnector
from .arxiv import ArxivConnector
from .base import ConnectorOrchestrator, RequestPolicy, SQLiteCache
from .crossref import CrossrefConnector
from .dblp_offline import DblpOfflineConnector
from .dblp_sqlite import DblpSQLiteConnector
from .dblp_online import DBLPOnlineConnector
from .europepmc import EuropePMCConnector
from .govinfo import GovInfoConnector
from .google_search import WebSearchConnector
from .google_scholar import GoogleScholarConnector
from .openalex import OpenAlexConnector
from .pubmed import PubMedConnector
from .searxng import SearxNGConnector
from .semanticscholar import SemanticScholarConnector
from .url_direct import URLDirectConnector


def default_orchestrator(
    cache_path: str | Path,
    dblp_mirror_path: str | Path,
    semantic_scholar_api_key: str | None = None,
    dblp_sqlite_path: str | Path | None = None,
    enabled_sources: list[str] | tuple[str, ...] | None = None,
    acl_anthology_data_dir: str | Path | None = None,
    acl_anthology_repo_path: str | Path | None = None,
    govinfo_api_key: str | None = None,
    searxng_base_url: str | None = None,
    ncbi_api_key: str | None = None,
    ncbi_email: str | None = None,
    web_search_provider: str | None = None,
    google_api_key: str | None = None,
    google_cse_id: str | None = None,
    serpapi_key: str | None = None,
    tavily_api_key: str | None = None,
) -> ConnectorOrchestrator:
    configured_sqlite_path = str(dblp_sqlite_path or "").strip()
    env_sqlite_path = (os.getenv("CITATION_CHECKER_DBLP_SQLITE_PATH") or "").strip()
    final_sqlite_path = configured_sqlite_path or env_sqlite_path
    enabled = {str(name).strip() for name in (enabled_sources or []) if str(name).strip()}

    if final_sqlite_path:
        connectors = [DblpSQLiteConnector(final_sqlite_path)]
    else:
        connectors = [DblpOfflineConnector(dblp_mirror_path)]
    # NOTE: url_direct is intentionally NOT added here.
    # It is invoked exclusively from verifier.py Phase 0 (before orchestrator.query)
    # so we don't double-fetch the citation URL.
    if os.getenv("CITATION_CHECKER_OFFLINE_ONLY", "0") != "1":
        connectors.extend(
            [
                DBLPOnlineConnector(),
                CrossrefConnector(),
                ArxivConnector(),
                ACLAnthologyConnector(
                    data_dir=acl_anthology_data_dir,
                    repo_path=acl_anthology_repo_path,
                ),
                EuropePMCConnector(),
                PubMedConnector(api_key=ncbi_api_key, email=ncbi_email),
                OpenAlexConnector(),
                SemanticScholarConnector(api_key=semantic_scholar_api_key),
                GovInfoConnector(api_key=govinfo_api_key),
                GoogleScholarConnector(serpapi_key=serpapi_key),
                WebSearchConnector(
                    provider=web_search_provider,
                    api_key=google_api_key,
                    cse_id=google_cse_id,
                    serpapi_key=serpapi_key,
                    tavily_api_key=tavily_api_key,
                ),
                SearxNGConnector(base_url=searxng_base_url),
            ]
        )
    if enabled:
        connectors = [connector for connector in connectors if _connector_enabled(connector.name, enabled)]
    cache = SQLiteCache(cache_path)
    policy = RequestPolicy()
    return ConnectorOrchestrator(connectors=connectors, cache=cache, policy=policy)


def _connector_enabled(connector_name: str, enabled: set[str]) -> bool:
    if connector_name in enabled:
        return True
    aliases = {
        "web_search": {"google_search"},
    }
    return any(alias in enabled for alias in aliases.get(connector_name, set()))


__all__ = [
    "ConnectorOrchestrator",
    "RequestPolicy",
    "SQLiteCache",
    "default_orchestrator",
]
