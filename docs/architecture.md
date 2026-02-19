# Architecture

## Pipelines

- Source checker (`apps/source_checker`):
  - extract citations from `.tex`
  - parse `.bib`
  - build canonical records
  - verify against bibliographic connectors
  - output JSON + Markdown report
- PDF checker (`apps/pdf_checker`):
  - extract text from PDF
  - segment reference entries
  - parse fields from references
  - verify against same connector core
  - output JSON + Markdown report

## Shared Core

`packages/core` contains shared types, normalization, matching, adjudication, and report rendering.

Decision policy is deterministic-first and precision-oriented:
- `VALID`
- `FLAWED_CITATION`
- `SUSPECTED_HALLUCINATION`
- `INSUFFICIENT_EVIDENCE`

## Connector Layer

`packages/connectors` provides:
- online connectors: Crossref, arXiv, DBLP, OpenAlex, Semantic Scholar
- offline connector: DBLP mirror (`data/cache/dblp_mirror.jsonl`)
- retry/backoff policy
- source health scoring
- SQLite cache (`data/cache/connector_cache.sqlite`)
