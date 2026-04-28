# Citation Hallucination Detection

A three-stage cascading multi-agent pipeline that detects fabricated citations
in research papers. Each citation is adjudicated against an **11-code
taxonomy** across three classes (`REAL`, `POTENTIAL`, `HALLUCINATED`), so a
reviewer learns both *whether* a citation is wrong and *which field* is wrong.

- **Stage 1** parses PDF, LaTeX, and HTML sources into structured citation records.
- **Stage 2** collects candidate matches through an ordered cascade: local
  memory cache → URL fetch → ten bibliographic connectors in parallel → web
  search fallback.
- **Stage 3** adjudicates each citation with a rule-based matcher plus three
  specialist judge agents (`Valid`, `Potential`, `Hallucinated`); only
  `Potential` uses an LLM.

See [docs/taxonomy.md](docs/taxonomy.md) for full subtype definitions and
[docs/metric_guide.md](docs/metric_guide.md) for the scoring protocol.

## Taxonomy summary

The full taxonomy enumerates 13 codes across three classes. Two codes
(`R4`, `P2`) are defined for completeness but are **not part of the current
evaluation set**. We explain why below the table.

| Class          | Code | Description                                                                            | Evaluated |
| ----------------| ------| ----------------------------------------------------------------------------------------| :---------:|
| `REAL`         | R1   | Exact field-wise match                                                                 | ✓         |
|                | R2   | Normalizable format variant (initials, abbreviated venue, etc.)                        | ✓         |
|                | R3   | `et al.` truncation; listed authors are correct                                        | ✓         |
|                | R4   | No single candidate covers all fields, but the union across connectors does            | ✗         |
| `POTENTIAL`    | P1   | One author's given name is a plausible nickname variant                                | ✓         |
|                | P2   | Non-academic source (blog post, tweet, GitHub repo, etc.); existence partly verifiable | ✗         |
|                | P3   | Core fields match; peripheral fields fabricated and unverifiable across every source   | ✓         |
| `HALLUCINATED` | H1   | Title error (word substitution, paraphrase, fabrication)                               | ✓         |
|                | H2   | Author error (addition, deletion, reordering, fabrication)                             | ✓         |
|                | H3   | Venue error (paper exists, cited at a different venue)                                 | ✓         |
|                | H4   | Year error                                                                             | ✓         |
|                | H5   | DOI or identifier error (resolves elsewhere or not at all)                             | ✓         |
|                | H6   | Peripheral error verifiable against a source (pages, volume, publisher, location)      | ✓         |

**Why `R4` and `P2` are excluded from the evaluation.**

- `R4` (cross-source coverage): no single targeted dataset exists, because
  the condition depends on the current union of connector indices at query
  time. The same citation can flip between `R1` and `R4` as any connector
  updates its records, which makes reproducible ground truth difficult.
  Our pipeline still recognizes and emits `R4` at runtime; we just do not
  score it.
- `P2` (non-academic source): reliable evaluation requires a curated corpus
  of citations to blog posts, tweets, repositories, and similar web sources,
  with verified live URLs and cached snapshots. Collecting and maintaining
  such a set is out of scope for this release (broken-link rot alone would
  force continuous re-curation). `P2` remains a first-class runtime verdict
  for deployed use, but it has no row in our paper tables.

For scoring we therefore work with **11 evaluated codes** and collapse
`R1`/`R2`/`R3` into a single bucket `R`, giving **9 scoring buckets**:
`R, P1, P3, H1..H6`.

## Synthetic dataset distribution

The current evaluation set contains **2,450 citations** across the 11
evaluated codes. Each subtype has its own JSON file under
`data/synthetic_data/v2/`. The `R*_plus` files are an extended real set
harvested from an independent BibTeX seed corpus; each `_plus` file rolls
up into its parent subtype (`R1_plus` → `R1`, and so on) for every count,
table, and metric.

| Class          | Code          | Count     | File(s)                                                   |
| -------------- | ------------- | --------: | --------------------------------------------------------- |
| `REAL`         | R1            | 338       | `R1.json` (171) + `R1_plus.json` (167)                    |
|                | R2            | 342       | `R2.json` (175) + `R2_plus.json` (167)                    |
|                | R3            | 343       | `R3.json` (177) + `R3_plus.json` (166)                    |
|                | **R (total)** | **1,023** | scoring bucket `R`                                        |
| `POTENTIAL`    | P1            | 91        | `P1.json`                                                 |
|                | P3            | 180       | `P3.json`                                                 |
|                | **P (total)** | **271**   |                                                           |
| `HALLUCINATED` | H1            | 200       | `H1.json`                                                 |
|                | H2            | 198       | `H2.json`                                                 |
|                | H3            | 197       | `H3.json`                                                 |
|                | H4            | 195       | `H4.json`                                                 |
|                | H5            | 200       | `H5.json`                                                 |
|                | H6            | 166       | `H6.json`                                                 |
|                | **H (total)** | **1,156** |                                                           |
| **Total**      |               | **2,450** |                                                           |

Ground truth for `R1`..`R3`, `P1`/`P3`, and `H1`..`H6` lives in
`data/synthetic_data/v2/meta.json`. The `R*_plus` entries carry no meta
entry and default to `REAL` / `VALID` at scoring time.

## Current benchmark accuracy

Per-bucket accuracy of the default pipeline (rule-based `Valid` and
`Hallucinated` judges, LLM `Potential` judge with `P1`/`P3` short-circuit
rules) on the 2,450-citation evaluation set. Correctness is defined per the
[metric guide](docs/metric_guide.md): class-level for the `R` bucket,
subtype-level for every other bucket.

| Subtype | N     | Correct | Accuracy | Errors |
| ------- | ----: | ------: | -------: | -----: |
| R       |  1023 |     965 |    94.3% |     58 |
| P1      |    91 |      91 |   100.0% |      0 |
| P3      |   180 |     179 |    99.4% |      1 |
| H1      |   200 |     200 |   100.0% |      0 |
| H2      |   198 |     196 |    99.0% |      2 |
| H3      |   197 |     196 |    99.5% |      1 |
| H4      |   195 |     186 |    95.4% |      9 |
| H5      |   200 |     200 |   100.0% |      0 |
| H6      |   166 |     166 |   100.0% |      0 |
| **Total** | **2450** | **2379** | **97.1%** | **71** |

The weakest buckets are `R` (58 false alarms on real citations) and `H4` (9
year errors mislabeled as a neighbouring `H*` code). All `P*` and the
remaining `H*` buckets score at or near ceiling.

## Supported bibliographic APIs

Stage 2 queries the ten sources below in parallel. Most are free public APIs;
three require a key for unthrottled throughput, and the web-search fallback
needs whichever provider you configure. All endpoints are called from
`packages/connectors/`; each source is independent and can be toggled via
`connectors.enabled_sources` in `config.json`.

| Connector (`source` name) | Provider       | Endpoint                                                  | Auth required                           |
| ------------------------- | -------------- | --------------------------------------------------------- | --------------------------------------- |
| `dblp_sqlite`             | DBLP           | local SQLite mirror built from `https://dblp.org/xml/`    | none (local file)                       |
| `dblp_online`             | DBLP           | `https://dblp.org/search/publ/api`                        | none                                    |
| `crossref`                | Crossref       | `https://api.crossref.org/works`                          | none (polite pool via user-agent email) |
| `openalex`                | OpenAlex       | `https://api.openalex.org/works`                          | optional `openalex_api_key` + `openalex_mailto` for the polite pool |
| `semantic_scholar`        | Semantic Scholar | `https://api.semanticscholar.org/graph/v1/paper/search` | `semantic_scholar_api_key` recommended (unauthenticated calls are heavily rate-limited) |
| `arxiv`                   | arXiv          | `https://export.arxiv.org/api/query`                      | none                                    |
| `pubmed`                  | NCBI PubMed    | `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`          | optional `ncbi_api_key` + `ncbi_email`  |
| `europepmc`               | Europe PMC     | `https://www.ebi.ac.uk/europepmc/webservices/rest/search` | none                                    |
| `acl_anthology`           | ACL Anthology  | `https://aclanthology.org/search/` and local `.bib` mirror | none (local clone speeds up bulk runs) |
| `url_direct`              | Publisher HTML | follows the citation's explicit URL and reads `<meta name="citation_*">` tags | none                                    |
| `web_search` (fallback)   | Tavily / SerpAPI / Google Custom Search | `https://api.tavily.com/search` or `https://serpapi.com/search.json` or `https://www.googleapis.com/customsearch/v1` | one of `tavily_api_key`, `serpapi_key`, or `google_api_key` + `google_cse_id` |

The pipeline always prefers structured connectors; the web-search fallback
runs only when every structured source returns no qualifying candidate, and
its raw results pass through a small LLM extractor (`packages/core/extractor_agent.py`)
before entering adjudication.

## LLM provider for Stage 1 extraction and the Potential judge

The pipeline invokes an LLM in two places: the entry extractor that turns
OCR output or web-search raw content into structured fields
(`packages/core/extractor_agent.py`), and the Potential judge that
adjudicates `P1` / `P3` cases (`packages/core/bedrock_agents.py`). Both are
configured under `entry_extraction` and `verification_llm` in `config.json`.

**Currently supported.**

- **AWS Bedrock** (default) — any Bedrock-hosted model accessible via the
  Converse API. Configure `region`, `bearer_token` (or AWS credentials), and
  `model_id` (for example `qwen.qwen3-vl-235b-a22b`). This is the path all
  released results use.
- **Local HuggingFace / vLLM** — usable for the entry extractor only by
  setting `entry_extraction.provider: "local"` and pointing
  `entry_extraction.local.model_path` at a DeepSeek-OCR checkpoint.

**Planned extensions.** The LLM interface is a thin wrapper around a single
`_call_bedrock(client, model_id, prompt)` function, so adding a new provider
means writing one parallel adapter and routing through
`verification_llm.provider`. Targets under consideration:

- **OpenAI** — GPT-4o / GPT-4.1 via the Chat Completions API.
- **Anthropic** — Claude 3.5 / 4 family via the Messages API.
- **Google** — Gemini 1.5 / 2 via the Generative Language API.
- **Azure OpenAI** — the same OpenAI API surface behind Azure auth.
- **Self-hosted** — vLLM or TGI endpoints exposing an OpenAI-compatible API.

Once a new adapter lands, switching providers will be a one-line change in
`config.json` (`verification_llm.provider`) plus the matching auth fields;
the rule-based Valid and Hallucinated agents stay unchanged.

## Repository layout

```
apps/
  pdf_checker/          # end-to-end PDF auditor
  source_checker/       # LaTeX-source auditor
packages/
  core/                 # agents, adjudication, taxonomy, cache
  connectors/           # DBLP, Crossref, OpenAlex, ACL Anthology, arXiv, ...
data/
  synthetic_data/v2/    # evaluation dataset (11 JSON files + meta.json)
  bst/                  # BibTeX styles used to render benchmark PDFs
  bib/harvested_seeds_clean.json   # seed pool for R*_plus expansion
pdf_samples_full/       # regenerable benchmark PDFs (5 styles × 10 papers)
scripts/                # evaluation, dataset construction, diagnostic tools
results/                # eval JSONs + aggregate stats (detailed_stats.csv)
docs/
  taxonomy.md           # full taxonomy definitions
  metric_guide.md       # scoring protocol (confusion matrix + tables)
```

## Setup

### 1. Install dependencies

```bash
conda create -n citeaudit python=3.10 -y && conda activate citeaudit
pip install -r requirements.txt
```

The pipeline also needs `pdflatex` and `bibtex` on `PATH` if you plan to
regenerate the benchmark PDFs.

### 2. Configure

Start from the template:

```bash
cp config.example.json config.json
```

`config.json` has four top-level blocks. Below is the **minimum viable
config** with inline comments showing what each field controls; the example
below uses all-Bedrock for both the OCR-extractor LLM and the verification
judge. The sections after walk through each block in detail.

```jsonc
{
  // --- Stage 1 method ---
  "citation_parse_method": "ocr_llm_extract",  // "ocr_llm_extract" (PDF, default) or "citation_reparse"

  // --- Stage 2 connectors ---
  "connectors": {
    "cache_path": "data/cache/connector_cache.sqlite",
    "dblp_sqlite_path": "/abs/path/to/dblp.sqlite",  // built once via scripts/build_dblp_sqlite.py
    "enabled_sources": [
      "dblp_sqlite", "url_direct", "dblp_online", "crossref", "arxiv",
      "acl_anthology", "europepmc", "pubmed", "openalex", "semantic_scholar",
      "web_search"
      // "searxng_search"  // optional: add this entry to enable the local SearxNG fallback
    ],
    "semantic_scholar_api_key": "<your-key>",   // strongly recommended, otherwise heavy throttling
    "openalex_api_key":         "<your-key>",   // optional, polite pool
    "openalex_mailto":          "you@host.edu", // optional, polite pool
    "ncbi_api_key":             "<your-key>",   // optional, lifts PubMed RPS cap
    "ncbi_email":               "you@host.edu", // recommended together with ncbi_api_key
    "web_search_provider":      "tavily",        // "tavily" | "serpapi" | "google_cse"
    "tavily_api_key":           "<your-key>",
    "searxng_base_url":         "http://127.0.0.1:8080"  // optional; only used if "searxng_search" is in enabled_sources
  },

  // --- Stage 1 LLM extractor (OCR → structured fields) ---
  "entry_extraction": {
    "mode": "model",
    "provider": "bedrock",          // "bedrock" or "local"
    "bedrock": {
      "region":       "us-east-1",
      "bearer_token": "<aws-bearer-token>"
    },
    "local": {                      // only used if provider == "local"
      "model_path":       "/abs/path/to/DeepSeek-OCR-2",
      "inference_backend": "vllm"
    }
  },

  // --- Stage 1 OCR + LLM merge (PDF input only) ---
  "ocr_llm_extract": {
    "enabled": true,
    "boundary_merge_mode": "rule",  // "rule" (cheap) | "llm" (Bedrock at boundaries) | "none"
    "bedrock": {
      "region":   "us-east-1",
      "model_id": "qwen.qwen3-vl-235b-a22b"
    }
  },

  // --- Stage 3 verification LLM (Potential-judge + extractor agent) ---
  "verification_llm": {
    "enabled":        true,
    "provider":       "bedrock",
    "max_candidates": 5,
    "bedrock": {
      "region":   "us-east-1",
      "model_id": "qwen.qwen3-vl-235b-a22b"
    }
  }
}
```

#### 2a. Pick your connectors

Drop any source from `enabled_sources` that you cannot authenticate. The
defaults are tuned for ML/NLP papers; for life-science papers keep
`pubmed`/`europepmc`, for math/physics keep `arxiv`. The cascade short-circuits
once a clean candidate appears, so listing more sources is rarely a cost
problem — but **the order in `enabled_sources` matters**: faster/cleaner
sources should come first.

| Field | Purpose | When you need it |
|---|---|---|
| `enabled_sources` | Which connectors run in Stage 2. | Always set. |
| `dblp_sqlite_path` | Path to a local DBLP mirror (fastest connector by ~30×). | Strongly recommended for any large run; otherwise fall back to `dblp_online`. |
| `semantic_scholar_api_key` | Lifts S2 from ~100 req/5min to a usable rate. | Any batch run. |
| `openalex_api_key` + `openalex_mailto` | Joins OpenAlex's polite pool. | Optional but free. |
| `ncbi_api_key` + `ncbi_email` | Lifts PubMed/E-utilities RPS cap. | Life-science papers. |
| `web_search_provider` + `tavily_api_key` (or `serpapi_key`, or `google_api_key` + `google_cse_id`) | Web-search fallback for non-academic citations. | Any paper that may cite blogs/repos/whitepapers. |
| `searxng_base_url` | Self-hosted SearxNG instance for free web fallback. | Optional; add `"searxng_search"` to `enabled_sources` to activate. |

#### 2b. Pick your LLM provider

The pipeline calls an LLM in three places, all controlled by separate blocks:

| Block | What it does | Default model |
|---|---|---|
| `entry_extraction` | Stage 1: turns OCR output (or web-search snippets) into structured citation fields. | `qwen.qwen3-vl-235b-a22b` via Bedrock; or local DeepSeek-OCR |
| `ocr_llm_extract` | Stage 1: column/page-boundary merge judge for PDF input. | `qwen.qwen3-vl-235b-a22b` via Bedrock |
| `verification_llm` | Stage 3: Potential-judge for `P1`/`P3` adjudication, plus the web-result extractor agent. | `qwen.qwen3-vl-235b-a22b` via Bedrock |

For an all-Bedrock setup you only need to fill `region` + `bearer_token`
once per block (or share them across blocks by hand). For a mixed setup
— for example local DeepSeek-OCR for Stage 1, Bedrock Qwen for Stage 3 —
set `entry_extraction.provider: "local"` and leave the other blocks on
`bedrock`. The Stage 3 judge cannot currently run locally; see "Planned
extensions" below for non-Bedrock provider support.

`ocr_llm_extract.boundary_merge_mode` is the one knob worth tuning per run:

- `rule` (default) — deterministic merge heuristic at column/page
  boundaries. No LLM cost. Use this unless you see merge errors in the
  parsed reference list.
- `llm` — Bedrock decides at every ambiguous boundary. Strongest
  correctness on noisy OCR; adds one Bedrock call per boundary
  (~10–30 per paper).
- `none` — trust the layout segmenter as-is. Cheapest; OK for clean
  arXiv/conference PDFs.

#### 2c. Sanity-check your config

```bash
python -m apps.pdf_checker.run --input data/pdf_papers/sample.pdf \
  --out /tmp/sanity_report.json --citation-workers 1 --paper-workers 1
```

If this runs end-to-end on one paper, your config is wired correctly. The
log line `Cascading 3-agent + field classifier + extractor enabled` confirms
that Stage 3 LLM agents booted (i.e., your Bedrock credentials work).

### 3. Pre-fetch caches (optional)

```bash
mkdir -p data/cache
# DBLP SQLite mirror: download from https://dblp.org/xml/ and run an
# offline indexer, then point dblp_sqlite_path at it.
# ACL Anthology: clone https://github.com/acl-org/acl-anthology.git into
# data/cache/acl_anthology_repo (its data/ subdir becomes acl_anthology_data_dir).
```

## Models and hardware

The default pipeline uses two models. Both are configurable in `config.json`.

| Role | Default model | Where it runs | Why |
|---|---|---|---|
| OCR (Stage 1, when input is a PDF) | `DeepSeek-OCR-2` (vLLM backend) | Local A100 GPU (≥40 GB VRAM, we run on 80 GB) | Layout-aware reference parsing; emits per-block bbox tags that the segmenter consumes. |
| Citation reparse + boundary-merge judge + Potential judge | `qwen.qwen3-vl-235b-a22b` via AWS Bedrock | Bedrock Converse API | Vision-language model. Used three places: (a) per-citation reparse to canonicalise fields from raw OCR, (b) merge-or-separate decision at column/page boundaries when `ocr_llm_extract.boundary_merge_mode=llm`, (c) Potential adjudication for `P1`/`P3` cases. |

The default `config.json` ships with these defaults baked in. Change the
provider for either role by editing `entry_extraction.local.model_path`
(OCR) and `verification_llm.bedrock.model_id` / `ocr_llm_extract.bedrock.model_id`
(LLM). Switching the LLM to a non-Bedrock provider needs a small adapter
(see "Planned extensions" above).

### Pipeline invariants

- **`ocr_llm_extract.boundary_merge_mode`** (default `rule`) — three values:
  - `rule`: deterministic content-merge heuristic. Cheapest.
  - `llm`: ask Bedrock at every column/page boundary whose continuation is
    not obviously a fresh entry. Strongest correctness on noisy OCR;
    requires Bedrock credentials.
  - `none`: trust the layout-aware segmenter as-is.
- **Year never gets rewritten by the LLM** ([citation_parser.py:1296-1308](apps/pdf_checker/ingest/citation_parser.py#L1296-L1308)).
  Vision-language models tend to "helpfully" override the printed year
  against the conference year in the venue name (or against their own
  training knowledge), which destroys the H4 hallucination detection
  signal. The pipeline keeps the heuristic-extracted year as ground truth
  for what is on the page; reparse touches title, authors, venue, volume,
  pages, publisher, location, doi, arxiv_id, and url.
- **`with-images` for VLM reparse** is on by default in `apps/pdf_checker/run.py`
  (production verification flow) and off by default in
  `pdf_extractor_demo.py` and `scripts/reprocess_with_llm_merge.py`
  (extraction-only flows). Pass `--with-images` to either to enable the
  rendered page image in the LLM reparse prompt.

## Running the checker on your own paper

### PDF

```bash
python -m apps.pdf_checker.run \
  --input data/pdf_papers/your_paper.pdf \
  --out artifacts/your_paper_report.json
```

### LaTeX source

```bash
python -m apps.source_checker.run \
  --input path/to/your_latex_project \
  --out artifacts/your_paper_report.json
```

The report JSON contains one entry per citation with the full adjudication
trail; see the **Output glossary** section below for every field.

## Running in parallel

The PDF checker exposes a **three-level concurrency model**: papers ×
citations within a paper × connectors per citation. All three are I/O-bound
(HTTP + LLM API calls), so increasing workers gives near-linear speedup
until you hit a remote rate limit. Defaults are tuned to be safe.

| Flag | Default | What it parallelizes |
|---|---:|---|
| `--paper-workers` | 2 | PDFs in flight when `--input` is a directory |
| `--citation-workers` | 4 | Citations within a single paper |
| `--connector-workers` | 8 | Connector queries per citation |

Total in-flight HTTP/LLM calls = `paper × citation × connector`. The
defaults give 2 × 4 × 8 = 64 concurrent calls; on a single beefy machine
this is a comfortable upper bound for free-tier S2/Crossref/OpenAlex keys.
Push higher only if you have authenticated keys and confirmed rate limits.

### One paper, default parallelism

```bash
python -m apps.pdf_checker.run \
  --input data/pdf_papers/your_paper.pdf \
  --out   artifacts/your_paper_report.json
```

### A directory of PDFs

`--input` accepts a directory; the runner recursively collects every
`*.pdf` and writes one report per paper into `--out`:

```bash
python -m apps.pdf_checker.run \
  --input data/pdf_papers/iclr2026_set/ \
  --out   artifacts/iclr2026_reports/ \
  --paper-workers    4 \
  --citation-workers 4 \
  --connector-workers 8
```

Each paper's report lands at
`artifacts/iclr2026_reports/<paper_stem>_report.json`. Reports are written
**incrementally per citation**, so a crash mid-run leaves valid partial
output for every citation that had finished.

### Tuning the workers

| Symptom | Action |
|---|---|
| Many `[rate-limit] X 429` lines in the log | Lower `--connector-workers`, or get an API key for connector `X` (most often Crossref or Semantic Scholar). |
| Bedrock throttling on the Stage 3 judge | Lower `--citation-workers`. Each citation makes at most one LLM call (Potential judge), so this directly throttles LLM RPS. |
| Single paper with 200+ citations runs slowly | Raise `--citation-workers` (try 8–12). Memory cost is negligible — the verifier is shared read-only across threads. |
| Whole machine sits idle waiting | Raise `--paper-workers`. The verifier instance is shared across paper threads, so RAM stays flat. |

### Serial debugging

When a citation produces an unexpected verdict, drop to serial to make
logs readable:

```bash
python -m apps.pdf_checker.run \
  --input data/pdf_papers/your_paper.pdf \
  --out   /tmp/debug.json \
  --paper-workers 1 --citation-workers 1 --connector-workers 1
```

### Skipping verification (extraction only)

For Stage 1 ablations, `--extract-only` runs OCR + reference parsing and
dumps the parsed citations without invoking Stage 2/3:

```bash
python -m apps.pdf_checker.run \
  --input data/pdf_papers/your_paper.pdf \
  --out   /tmp/parsed.json \
  --extract-only
```

### SLURM batch

Two ready-made SLURM scripts run the synthetic-data evaluation as parallel
arrays: `scripts/eval_R_rule_based.sbatch` and
`scripts/eval_H_rule_based.sbatch`. For PDF batch jobs, wrap the directory
form above in your own sbatch script and pin `--paper-workers` to the node
CPU count divided by 4 (one citation worker per CPU is a reasonable
starting point).

## Reproducing the benchmark

### Quickstart: testing the verifier on the RPH (`data/synthetic_data/v2/`) dataset

The evaluation set is the **RPH benchmark** (Real / Potential / Hallucinated)
at `data/synthetic_data/v2/`. Each `*.json` is a flat list of citation
records with the same field schema as the pipeline's own output.

```
data/synthetic_data/v2/
├── R1.json  R2.json  R3.json                 # Real, 523 total (3 subtypes)
├── R1_plus.json  R2_plus.json  R3_plus.json  # Extended Real seed pool, 500 more
├── P1.json  P3.json                          # Potential, 271 total (2 subtypes)
├── H1.json  H2.json  H3.json
├── H4.json  H5.json  H6.json                 # Hallucinated, 1156 total (6 subtypes)
└── meta.json                                 # citation_id → ground-truth label/subtype
```

Each `H*.json` and `P*.json` is generated by mutating a verified seed
citation along exactly one axis (year shift for `H4`, venue swap for `H3`,
peripheral fabrication for `H6`, nickname variant for `P1`, etc.); see
[docs/taxonomy.md](docs/taxonomy.md). Ground-truth labels come from
`meta.json`; the `R*_plus` rows have no entry there and default to `VALID`.

**To run the verifier on a single subtype** (here `H4`, the year-error
subtype, 195 citations):

```bash
python scripts/test_synthetic_samples.py \
  --samples data/synthetic_data/v2/H4.json \
  --meta    data/synthetic_data/v2/meta.json \
  --workers 8 \
  --rule-based-valid-agent --rule-based-hallucinated-agent \
  --out results/eval_H4_rule_based.json
```

The script feeds each citation directly into `CitationVerifier` (so it
uses the full Stage-2 cascade and Stage-3 judges), then writes the verdict
trace plus an aggregate summary.

**To run all 11 subtypes in one shot:**

```bash
for sub in R1 R2 R3 R1_plus R2_plus R3_plus P1 P3 H1 H2 H3 H4 H5 H6; do
  expected=""
  if [[ "$sub" == R*_plus ]]; then
    expected="--expected-verdict VALID"  # _plus rows have no meta entry
  fi
  python scripts/test_synthetic_samples.py \
    --samples data/synthetic_data/v2/${sub}.json \
    --meta    data/synthetic_data/v2/meta.json \
    $expected \
    --workers 8 \
    --rule-based-valid-agent --rule-based-hallucinated-agent \
    --out results/eval_${sub}_rule_based.json
done
python scripts/build_detailed_stats.py
```

Outputs:
- `results/eval_<sub>_rule_based.json` — per-citation verdict trail with
  `predicted`, `taxonomy`, `adjudication_reason`, `evidence_sources`, etc.
- `results/detailed_stats.csv` — one row per citation with `expected_label`,
  `predicted_label`, `expected_subtype`, `predicted_taxonomy`, `match_label`,
  `match_taxonomy`. This is the file the metric guide consumes.
- `results/detailed_stats_summary.json` — per-bucket totals and accuracies.

**To score the PDF extractor end-to-end on the rendered benchmark PDFs**
(after running the OCR + LLM reparse pipeline; see
[`scripts/test_pdf_extractor.sh`](scripts/test_pdf_extractor.sh)):

```bash
# Stage 1+2 extraction over the 50 benchmark PDFs (uses Bedrock + DeepSeek-OCR)
PATH="<your ref_router conda env bin>:$PATH" \
  bash scripts/test_pdf_extractor.sh --out-root results/extraction_ocr

# Score the extracted JSONs against the cleaned bib (drops the 58
# render-omission citations whose mutated field never reached the page).
python scripts/score_extraction_clean.py \
  --ext-root results/extraction_ocr \
  --tag    ocr_clean \
  --exclude-omissions
```

The exclusion list at `docs/pdf_render_omissions.json` is the canonical
58-citation exclude set: 33 `bst_field_suppression` (the bibtex style
hides the mutated field at render time), 18 `latex_accent_escape`
(LaTeX accents that the rendered PDF prints as Unicode glyphs but the GT
bib stores as `\&apos;` / `\\\"u`), and 7 `html_entity_escape`. PDF-input
scores must drop these or they score the extractor for evidence that does
not exist on the page.

### 1. Dataset (already materialized)

The evaluation set lives under `data/synthetic_data/v2/`:

```
R1.json  R2.json  R3.json             # real citations, 523 total
R1_plus.json  R2_plus.json  R3_plus.json   # extended real set, 500 more
P1.json  P3.json                       # potential, 271 total
H1.json  H2.json  H3.json  H4.json  H5.json  H6.json   # hallucinated, 1156 total
meta.json                              # ground-truth labels + seeds
```

Total: **2450 citations**.

### 2. Rendered PDFs (optional)

```bash
python scripts/gen_exhaustive_pdfs.py
```

Produces `pdf_samples_full/{acm,splncs04,iclr,ieee,plain}/paper_XXX/main.pdf`.
Each paper bundles 40–50 citations sampled from the dataset; `refs.bib` and
`main.bbl` align 1:1 (verified).

### 3. Run the pipeline against the dataset

```bash
# R (real): use meta.json as ground truth
python scripts/test_synthetic_samples.py \
  --samples data/synthetic_data/v2/R1.json \
  --meta    data/synthetic_data/v2/meta.json \
  --workers 8 \
  --rule-based-valid-agent --rule-based-hallucinated-agent \
  --out results/eval_R1_rule_based.json

# R_plus (extended real, no meta entry): declare expected verdict directly
python scripts/test_synthetic_samples.py \
  --samples data/synthetic_data/v2/R1_plus.json \
  --expected-verdict VALID \
  --workers 8 \
  --rule-based-valid-agent --rule-based-hallucinated-agent \
  --out results/eval_R1_plus_rule_based.json

# P1, P3, H1..H6 follow the same pattern as R, with --meta.
```

Batched versions live in `scripts/eval_R_rule_based.sbatch` and
`scripts/eval_H_rule_based.sbatch` for SLURM.

### 4. Aggregate results

```bash
python scripts/build_detailed_stats.py
```

Produces:

- `results/detailed_stats.csv` — one row per citation with `expected_label`,
  `predicted_label`, `expected_subtype`, `predicted_taxonomy`, `match_label`,
  `match_taxonomy`.
- `results/detailed_stats_summary.json` — per-bucket totals and accuracies.

### 5. Score against the metric guide

Use [docs/metric_guide.md](docs/metric_guide.md) to produce the three paper
artifacts from `detailed_stats.csv`:

1. **9 × 9 confusion matrix** — rows = true bucket, columns = predicted bucket
   (`R, P1, P3, H1..H6`).
2. **Per-bucket table** — `N, TP, FP, FN, Precision, Recall, F1, Accuracy` for
   each of the 9 buckets (one-versus-rest).
3. **Safe-error analysis** — per-class safe/dangerous error counts, plus
   dataset-wide `SER`, `DER`, and identity `class_accuracy = 1 − SER − DER`.

Our rule-based pipeline reaches `class_accuracy ≈ 97.1%` on the 2450-citation
set, with `DER` dominated by `H4` and `H2` subtype drift (see the per-bucket
table for the full breakdown).

## Output glossary

Every run produces one JSON per citation with the fields below. The pipeline
commits to a single class (`predicted`) and a set of fine-grained codes
(`taxonomy`):

| Field                   | Type      | Meaning                                                                                              |
| ----------------------- | --------- | ---------------------------------------------------------------------------------------------------- |
| `citation_id`           | string    | Stable ID, e.g. `H1-0042`.                                                                           |
| `input_citation`        | object    | Parsed fields from Stage 1.                                                                          |
| `predicted`             | string    | `VALID` / `POTENTIAL_REFERENCE` / `FAKE_REFERENCE`.                                                  |
| `taxonomy`              | string[]  | Fine-grained codes (`R`, `P1`, `P3`, `H1`..`H6`); multi-code sets appear when several fields mismatch. |
| `adjudication_reason`   | string    | One-sentence explanation from whichever judge closed the case.                                       |
| `evidence_sources`      | string[]  | Connectors that returned at least one candidate.                                                     |
| `conflicts`             | string[]  | Fields that mismatched at least one qualifying candidate.                                            |
| `field_status`          | object    | Per-field verdict (`match`, `mismatch`, `candidate_missing`, `reference_missing`, `both_missing`).   |
| `matched_candidate`     | object?   | Best-matching candidate (or null if none qualified).                                                 |
| `candidate_evaluations` | object[]  | Full per-connector evaluation trail for the citation.                                                |
| `needs_human_review`    | bool      | `true` when the pipeline is uncertain.                                                               |

`predicted` maps one-to-one to the taxonomy classes: `VALID` ↔ `REAL`,
`POTENTIAL_REFERENCE` ↔ `POTENTIAL`, `FAKE_REFERENCE` ↔ `HALLUCINATED`.

## Citing and contributing

The benchmark, taxonomy, and pipeline are released together. Issues and pull
requests are welcome, especially new connectors, additional subtypes, and
independent scorers against the metric guide.
