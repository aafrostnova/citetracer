# Citation Hallucination Detection

Prototype-informed implementation of a citation integrity checker with two pipelines:

- `apps/source_checker`: checks LaTeX + BibTeX source
- `apps/pdf_checker`: checks rendered PDF references

Both share a common verifier in `packages/core` and connector stack in `packages/connectors`.

## Quickstart

```bash
python3 -m apps.source_checker.run --input data/fixtures/sample_source --out artifacts/sample_source_report.json
python3 -m apps.pdf_checker.run --input data/fixtures/sample_pdf/sample.pdf --out artifacts/sample_pdf_report.json
python3 -m packages.eval.run --suite synthetic_stress --out artifacts/eval
```

## ArXiv Seed Data

Fetch initial real-world data from arXiv:

```bash
python3 scripts/fetch_arxiv_seed.py \
  --query "cat:cs.LG" \
  --max-results 3 \
  --metadata-out data/real_world/arxiv_seed_metadata.jsonl \
  --manifest-out data/real_world/arxiv_seed_manifest.json \
  --mirror-out data/real_world/arxiv_seed_mirror.jsonl
```

Download PDFs and run a smoke pipeline pass:

```bash
python3 scripts/fetch_arxiv_seed.py --download-pdfs --output-dir /tmp/arxiv_seed_pdfs
python3 scripts/run_arxiv_seed_smoke.py --manifest data/real_world/arxiv_seed_manifest.json --out artifacts/arxiv_smoke
```

Runtime env knobs:

- `CITATION_CHECKER_OFFLINE_ONLY=1` to disable online connectors
- `CITATION_CHECKER_CACHE_PATH=/tmp/cache.sqlite` to override cache location
- `CITATION_CHECKER_DBLP_MIRROR_PATH=/path/to/mirror.jsonl` to override mirror file

## GitHub Actions

- `.github/workflows/ci.yml`: unit tests + offline source/PDF smoke on push/PR.
- `.github/workflows/arxiv-smoke.yml`: scheduled/manual arXiv seed fetch and pipeline smoke run with uploaded artifacts.

## Project Layout

- `apps/`: source and PDF checker applications
- `packages/core/`: models, normalization, matching, adjudication, report logic
- `packages/connectors/`: bibliographic connectors + cache + request policy
- `packages/eval/`: benchmark runner and metrics
- `data/`: fixtures, manifests, labels, and offline DBLP mirror
- `docs/`: architecture, schema, evaluation protocol, annotation guidelines
