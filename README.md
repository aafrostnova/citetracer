# Citation Hallucination Detection

Prototype implementation of a citation integrity checker with two pipelines:

- `apps/source_checker`: checks LaTeX + BibTeX source references.
- `apps/pdf_checker`: checks rendered PDF references.

Both pipelines share verifier logic in `packages/core` and connectors in `packages/connectors`.

## Data Layout

Paper assets are now organized under two top-level directories:

- `data/latex_papers/`: LaTeX source papers
- `data/pdf_papers/`: PDF papers

Current examples:

- `data/latex_papers/sample_source`
- `data/pdf_papers/sample_pdf/2510.06445v2.pdf`
- `data/pdf_papers/hallucinated_iclr_2026`
- `data/pdf_papers/hallucinated_neurips_2025`
- `data/pdf_papers/benign_icml_oral_2026`
- `data/pdf_papers/benign_iclr_oral_2026`

## Quickstart

```bash
python3 -m apps.source_checker.run \
  --input data/latex_papers/sample_source \
  --out artifacts/sample_source_report.json

python3 -m apps.pdf_checker.run \
  --input /project/pi_shiqingma_umass_edu/mingzheli/Citation_Hallucination_Detection/data/pdf_papers/hallucinated_iclr_2026/A_superpersuasive_autonomous_policy_debating_system.pdf \
  --out artifacts/sample_pdf_report.json

python3 -m apps.pdf_checker.run \
  --input data/pdf_papers/hallucinated_iclr_2026/A_superpersuasive_autonomous_policy_debating_system.pdf \
  --out artifacts/A_superpersuasive_autonomous_policy_debating_system.json
```

## PDF Reference Extraction Demo

Use `pdf_extractor_demo.py` to run extraction/parsing only (without full checker adjudication).

Single PDF:

```bash
python3 pdf_extractor_demo.py \
  --input data/pdf_papers/hallucinated_iclr_2026/C3-OWD_A_Curriculum_Cross-modal_Contrastive_Learning_Framework_for_Open-World_Detection.pdf \
  --out artifacts/C3-OWD_reference_extract.json
```

Batch folder:

```bash
python3 pdf_extractor_demo.py \
  --input data/pdf_papers/hallucinated_iclr_2026 \
  --out artifacts/iclr_2026_reference_extracts \
  --workers 1 \
  --continue-on-error

python3 pdf_extractor_demo.py \
  --input data/pdf_papers/benign_icml_oral_2026 \
  --out artifacts/icml_2025_reference_extracts_1 \
  --workers 1 \
```

Notes:

- `--workers` is forced to `1` for local GPU OCR model mode to avoid contention/OOM.
- Batch mode writes `_batch_manifest.json` in output directory.

## LLM Reparse (Optional)

The parser can run an LLM-based reparsing pass on citations with missing core fields.

Config block (see `config.example.json`):

```json
"citation_reparse": {
  "enabled": false,
  "model_path": "/project/pi_shiqingma_umass_edu/mingzheli/model/Qwen3-0.6B",
  "max_new_tokens": 32768,
  "temperature": 0.0
}
```

Run reparsing test on existing artifact JSON files:

```bash
python3 scripts/test_llm_reparse_on_artifacts.py \
  --input-dir artifacts/icml_oral_2026_reference_extracts_20260301_215912 \
  --output-dir artifacts/icml_oral_2026_reference_extracts_20260301_215912_llm_reparse_test \
  --config-path config.json \
  --model-path /project/pi_shiqingma_umass_edu/mingzheli/model/Qwen3-0.6B \
  --max-new-tokens 32768 \
  --temperature 0
```

## Synthetic LaTeX Fixture

Generate a mixed verified/flawed/fabricated citation fixture:

```bash
python3 scripts/generate_synthetic_latex_fixture.py \
  --output-dir data/latex_papers/synthetic_mixed_source
```

## Real ArXiv Seed Scripts

Fetch real arXiv metadata (and optional PDFs + LaTeX sources):

```bash
python3 scripts/fetch_arxiv_seed.py \
  --query "cat:cs.LG" \
  --max-results 3 \
  --metadata-out data/real_world/arxiv_seed_metadata.jsonl \
  --manifest-out data/real_world/arxiv_seed_manifest.json \
  --mirror-out data/real_world/arxiv_seed_mirror.jsonl
```

Download real PDFs and source archives:

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

Smoke check fetched arXiv assets:

```bash
python3 scripts/run_arxiv_seed_smoke.py \
  --manifest data/real_world/arxiv_seed_manifest.json \
  --out artifacts/arxiv_smoke \
  --pipelines both
```

## Runtime Environment Knobs

- `CITATION_CHECKER_CONFIG_PATH=...`: JSON config path (default `config.json`)
- `CITATION_CHECKER_OFFLINE_ONLY=1`: disable online connectors
- `CITATION_CHECKER_CACHE_PATH=/tmp/cache.sqlite`: override connector cache
- `CITATION_CHECKER_DBLP_MIRROR_PATH=/path/to/mirror.jsonl`: override DBLP mirror
- `CITATION_CHECKER_DBLP_SQLITE_PATH=/path/to/dblp.sqlite`: use DBLP SQLite
- `CITATION_CHECKER_PDF_ENTRY_EXTRACTION=heuristic|model`: extraction mode
- `CITATION_CHECKER_PDF_MODEL_PROVIDER=bedrock|local`: model backend in model mode
- `CITATION_CHECKER_PDF_ENTRY_CHUNK_CHARS=12000`: model extraction chunk size
- `CITATION_CHECKER_BEDROCK_MODEL_ID=...`: override Bedrock model id
- `AWS_BEARER_TOKEN_BEDROCK=...`: bearer token for Bedrock auth
- `CITATION_CHECKER_LOCAL_MODEL_PATH=/path/to/DeepSeek-OCR-2`: local OCR model path
- `CITATION_CHECKER_LOCAL_OCR_DEBUG=1`: dump local OCR debug artifacts under `.tmp/`
- `CITATION_CHECKER_LOCAL_OCR_DEBUG_RAISE=1`: fail fast on local OCR exceptions
- `CITATION_CHECKER_VERBOSE=1`: print pipeline step logs + verification progress
- `CITATION_CHECKER_LLM_REPARSE_ENABLED=1`: enable LLM reparsing pass
- `CITATION_CHECKER_LLM_REPARSE_MODEL_PATH=/path/to/model`: LLM reparse model
- `CITATION_CHECKER_LLM_REPARSE_MAX_NEW_TOKENS=32768`: reparse generation budget
- `CITATION_CHECKER_LLM_REPARSE_TEMPERATURE=0`: reparse temperature

## JSON Config

Copy and edit:

```bash
cp config.example.json config.json
```

Runtime priority:

`environment variables > config.json > code defaults`

Minimal local OCR + optional reparse example:

```json
{
  "connectors": {
    "dblp_sqlite_path": "/project/pi_shiqingma_umass_edu/mingzheli/Ref_Agent/data/dblp.sqlite"
  },
  "entry_extraction": {
    "mode": "model",
    "provider": "local",
    "chunk_chars": 12000,
    "local": {
      "model_path": "/project/pi_shiqingma_umass_edu/mingzheli/model/DeepSeek-OCR-2"
    },
    "bedrock": {
      "region": "us-east-1",
      "bearer_token": ""
    }
  },
  "citation_reparse": {
    "enabled": false,
    "model_path": "/project/pi_shiqingma_umass_edu/mingzheli/model/Qwen3-0.6B",
    "max_new_tokens": 32768,
    "temperature": 0.0
  }
}
```

## Project Layout

- `apps/`: source and PDF checker applications
- `packages/core/`: models, normalization, matching, adjudication, report logic
- `packages/connectors/`: bibliographic connectors, cache, request policy
- `packages/eval/`: benchmark runner and metrics
- `data/`: paper fixtures and cache
- `docs/`: architecture and annotation guidelines

