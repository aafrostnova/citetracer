# Citation Hallucination Detection

Prototype-informed implementation of a citation integrity checker with two pipelines:

- `apps/source_checker`: checks LaTeX + BibTeX source
- `apps/pdf_checker`: checks rendered PDF references

Both share a common verifier in `packages/core` and connector stack in `packages/connectors`.

## Quickstart

```bash
python3 -m apps.source_checker.run --input data/fixtures/sample_source --out artifacts/sample_source_report.json
python3 -m apps.pdf_checker.run --input data/fixtures/sample_pdf/sample.pdf --out artifacts/sample_pdf_report.json
python3 -m apps.pdf_checker.run --input data/fixtures/ICLR_2026_hallucinated_papers_pdf/A_superpersuasive_autonomous_policy_debating_system.pdf --out artifacts/A_superpersuasive_autonomous_policy_debating_system.json
python3 -m packages.eval.run --suite synthetic_stress --out artifacts/eval
```

## Real ArXiv Test Data

Fetch real arXiv metadata (and optional PDFs + LaTeX sources):

```bash
python3 scripts/fetch_arxiv_seed.py \
  --query "cat:cs.LG" \
  --max-results 3 \
  --metadata-out data/real_world/arxiv_seed_metadata.jsonl \
  --manifest-out data/real_world/arxiv_seed_manifest.json \
  --mirror-out data/real_world/arxiv_seed_mirror.jsonl
```

Download real PDFs and source archives for local testing:

```bash
python3 scripts/fetch_arxiv_seed.py \
  --query "cat:cs.LG" \
  --max-results 3 \
  --pdf-output-dir data/seed/arxiv_pdfs \
  --source-output-dir data/seed/arxiv_sources \
  --download-pdfs \
  --download-sources \
  --extract-sources
```

Run smoke checks over fetched arXiv assets:

```bash
python3 scripts/run_arxiv_seed_smoke.py \
  --manifest data/real_world/arxiv_seed_manifest.json \
  --out artifacts/arxiv_smoke \
  --pipelines both
```

## Synthetic LaTeX Fixture

Generate a mixed verified/flawed/fabricated citation fixture:

```bash
python3 scripts/generate_synthetic_latex_fixture.py \
  --output-dir data/fixtures/synthetic_mixed_source
```

## Runtime Environment Knobs

- `CITATION_CHECKER_OFFLINE_ONLY=1` to disable online connectors
- `CITATION_CHECKER_CACHE_PATH=/tmp/cache.sqlite` to override cache location
- `CITATION_CHECKER_DBLP_MIRROR_PATH=/path/to/mirror.jsonl` to override mirror file
- `CITATION_CHECKER_DBLP_SQLITE_PATH=/path/to/dblp.sqlite` to use real DBLP SQLite (preferred over mirror JSONL)
- `CITATION_CHECKER_PDF_ENTRY_EXTRACTION=heuristic|model` to control PDF reference entry extraction mode
- `CITATION_CHECKER_PDF_MODEL_PROVIDER=bedrock|local` to choose model backend in `model` mode
- `CITATION_CHECKER_BEDROCK_MODEL_ID=...` to override Bedrock model id (default auto-selected when bearer token is present)
- `CITATION_CHECKER_PDF_ENTRY_CHUNK_CHARS=12000` to tune model extraction chunk size
- `CITATION_CHECKER_LOCAL_MODEL_PATH=/path/to/DeepSeek-OCR-2` to enable local model extraction (OCR over rendered reference-page images)
- `CITATION_CHECKER_LOCAL_OCR_DEBUG=1` to dump per-page local OCR debug artifacts under `.tmp/pdf_checker_local_ocr_pages/.../model_outputs`
- `CITATION_CHECKER_LOCAL_OCR_DEBUG_RAISE=1` to stop swallowing local OCR exceptions and fail fast
- `CITATION_CHECKER_VERBOSE=1` to print pipeline step logs and verification progress bar (`n/total`); set `0` to disable
- `AWS_BEARER_TOKEN_BEDROCK=...` to use bearer-token-only Bedrock auth (without AK/SK); when set, PDF extraction defaults to `model` mode
- `CITATION_CHECKER_CONFIG_PATH=...` to point to a JSON config file (default: `config.json`)

## JSON Config

You can store all PDF checker runtime settings in `config.json` and avoid exporting many env vars.

- Copy template: `cp config.example.json config.json`
- Edit `config.json` and set your token under `entry_extraction.bedrock.bearer_token`
- Runtime priority is: environment variables > `config.json` > built-in defaults

Minimal example:

```json
{
  "connectors": {
    "dblp_sqlite_path": "/project/pi_shiqingma_umass_edu/mingzheli/Ref_Agent/data/dblp.sqlite"
  },
  "entry_extraction": {
    "provider": "local",
    "local": {
      "model_path": "/project/pi_shiqingma_umass_edu/mingzheli/model/DeepSeek-OCR-2"
    },
    "bedrock": {
      "bearer_token": ""
    }
  }
}
```

Bedrock example:

```json
{
  "entry_extraction": {
    "provider": "bedrock",
    "bedrock": {
      "bearer_token": "YOUR_AWS_BEARER_TOKEN_BEDROCK"
    }
  }
}
```

Local provider requirements:
- `torch` + CUDA available
- `transformers` with `trust_remote_code=True` support
- `Pillow`
- `pymupdf` (render reference pages to images before OCR)

Local OCR flow:
- render reference pages from PDF to images
- run DeepSeek-OCR-2 with markdown prompt (`<|grounding|>Convert the document to markdown.`)
- convert markdown to `{"references":[{"raw_reference":"..."}]}` format used by the checker

## Real DBLP Data

The default `data/cache/dblp_mirror.jsonl` is a tiny fixture.  
For real DBLP coverage, point to an official DBLP-derived SQLite file:

```bash
export CITATION_CHECKER_DBLP_SQLITE_PATH=/project/pi_shiqingma_umass_edu/mingzheli/Ref_Agent/data/dblp.sqlite
export CITATION_CHECKER_OFFLINE_ONLY=1
```

When `CITATION_CHECKER_DBLP_SQLITE_PATH` is set, the checker uses the SQLite connector instead of the small mirror JSONL.

## GitHub Actions

- `.github/workflows/ci.yml`: unit tests + offline source/PDF smoke on push/PR.
- `.github/workflows/arxiv-smoke.yml`: scheduled/manual/push arXiv seed fetch and PDF+source smoke checks with uploaded artifacts.

## Project Layout

- `apps/`: source and PDF checker applications
- `packages/core/`: models, normalization, matching, adjudication, report logic
- `packages/connectors/`: bibliographic connectors + cache + request policy
- `packages/eval/`: benchmark runner and metrics
- `data/`: fixtures, manifests, labels, and offline DBLP mirror
- `docs/`: architecture, schema, evaluation protocol, annotation guidelines
