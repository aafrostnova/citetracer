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

Copy the example config and fill in your API keys and local paths:

```bash
cp config.example.json config.json
```

Key fields:

| Field | Purpose |
| ----- | ------- |
| `connectors.enabled_sources` | Which bibliographic connectors to query. Remove any source you cannot authenticate. |
| `connectors.dblp_sqlite_path` | Path to a local DBLP SQLite mirror (fastest connector). Leave empty to skip. |
| `connectors.semantic_scholar_api_key` | Semantic Scholar API key. Without it, rate limits throttle the connector heavily. |
| `connectors.openalex_api_key`, `openalex_mailto` | Authenticated OpenAlex access (polite pool). |
| `connectors.web_search_provider` | `tavily`, `serpapi`, or `google_cse`. Only needed when structured connectors miss a citation. |
| `connectors.tavily_api_key` / `serpapi_key` / `google_api_key` + `google_cse_id` | Credentials for the chosen web search provider. |
| `connectors.ncbi_api_key`, `ncbi_email` | PubMed/Europe PMC credentials. |
| `entry_extraction.local.model_path` | Path to the local OCR/extractor model. Set to a DeepSeek-OCR checkpoint for the `local` provider. |
| `entry_extraction.bedrock.*` | AWS Bedrock model id + bearer token for the `bedrock` provider. |
| `verification_llm.bedrock.model_id` | The LLM used by the `Potential` judge. |

Example provider selection in `config.json`:

```json
{
  "entry_extraction": {
    "mode": "model",
    "provider": "bedrock",
    "bedrock": {
      "region": "us-east-1",
      "bearer_token": "AWSV2-..."
    }
  },
  "verification_llm": {
    "enabled": true,
    "provider": "bedrock",
    "bedrock": {
      "region": "us-east-1",
      "model_id": "qwen.qwen3-vl-235b-a22b"
    }
  }
}
```

### 3. Pre-fetch caches (optional)

```bash
mkdir -p data/cache
# DBLP SQLite mirror: download from https://dblp.org/xml/ and run an
# offline indexer, then point dblp_sqlite_path at it.
# ACL Anthology: clone https://github.com/acl-org/acl-anthology.git into
# data/cache/acl_anthology_repo (its data/ subdir becomes acl_anthology_data_dir).
```

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

## Reproducing the benchmark

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
