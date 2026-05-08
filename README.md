# CITETRACER: Cascading Multi-Agent Citation Hallucination Detection

CITETRACER detects fabricated citations in research papers and routes each
citation to one of an **11-code taxonomy** (R, P1, P3, H1..H6) so reviewers
see *which* field is wrong, not just whether the citation is fake. The
pipeline parses PDF or BibTeX input, retrieves evidence through a four-stage
cascade (Memory cache, URL Fetch, eight Scholar Connectors in parallel, Web
Agent fallback), runs deterministic field matching, and routes residual
cases to class-specialist judge agents.

![CITETRACER overview](figs/overview.png)

On a 2,450-citation synthetic benchmark CITETRACER attains 97.1% accuracy
and class-level F1 of 97.0 / 95.8 / 98.5 for Real / Potential /
Hallucinated. On 957 real-world fabricated citations from ICLR 2026 and
ACM CCS 2026 desk-rejected submissions it catches 97.1% with no
abstentions. See [docs/taxonomy.md](docs/taxonomy.md) for the full
taxonomy and [docs/metric_guide.md](docs/metric_guide.md) for the scoring
protocol.

## Setup

```bash
# 1) clone
git clone https://github.com/aaFrostnova/Citation_Hallucination_Detection.git
cd Citation_Hallucination_Detection

# 2) install (Python 3.10+)
pip install -r requirements.txt

# 3) configure
cp config.example.json config.json
# edit config.json: fill in API keys (Bedrock bearer token or OpenAI/Azure
# key, plus Semantic Scholar / OpenAlex / NCBI / Tavily as needed) and the
# DBLP sqlite path. See "Configuration" below for the full list.
```

Optional: download `DeepSeek-OCR-2` to a local path and point
`entry_extraction.local.model_path` at it if you want OCR to run on a
local GPU instead of Bedrock.

## Quick start: verify one paper end-to-end

```bash
mkdir -p artifacts/demo
python -m apps.pdf_checker.run \
  --input path/to/paper.pdf \
  --out  artifacts/demo \
  --paper-workers 1 --citation-workers 8 --connector-workers 6
```

Output per paper:

- `<stem>_report.json` — structured per-citation verdicts
- `<stem>_report.md`   — reviewer-friendly markdown report
- `<stem>_report.timing.json` — per-phase latency

`--input` accepts a single PDF or a directory of PDFs. `--resume` skips
papers whose `_report.md` already exists, useful for batch jobs.

## Verifier-only mode (skip PDF extraction)

If your input is structured BibTeX records or already-parsed citations,
skip Stage 1 and run only the cascade plus judges. Inputs are JSON files
with the schema produced by Stage 1 (`results/eval_*_rule_based.json`).

```bash
python scripts/eval_judge_only.py \
  --inputs   results/eval_*_rule_based.json \
  --out      results/judge_only_run \
  --model    qwen.qwen3-vl-235b-a22b \
  --workers  16
```

This also accepts `--exclude-connectors` to ablate any connector group
(e.g. `--exclude-connectors web_search`).

## Datasets

Two datasets ship with the repository:

| Path                                       | What it is                                                                                                                | Size  |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | ----- |
| `data/synthetic_data/v2/`                  | 2,450 synthetic citations across the 11 evaluated codes (one JSON per subtype: `R1*.json`, `R2*.json`, `R3*.json`, `P1.json`, `P3.json`, `H1.json`..`H6.json`), with `meta.json` carrying the per-citation ground-truth label. Built from real BibTeX seeds with controlled LLM mutations. | 2,450 |
| `data/iclr2026_hallucinated/`              | 957 real-world fabricated citations from ICLR 2026 desk-rejected submissions. `hallucinated_refs.json` is the raw list; `hallucinated_refs_structured.json` is the parsed structured-record version used by the verifier. | 957   |

The synthetic set is the primary benchmark used in every paper table; the
real-world set is the out-of-distribution test in Section 4.4.

## Reproduce paper numbers

```bash
# main verification on the 2,450-citation synthetic benchmark
bash scripts/eval_H_rule_based.sh
bash scripts/eval_R_rule_based.sh
python scripts/build_detailed_stats.py    # rolls up per-subtype acc

# wall-clock benchmark
python scripts/bench_full_pipeline_bib.py \
  --workers 32 --per-subtype 15 --limit 200 \
  --out results/bench_full_bib_smoke
```

## Configuration

The pipeline reads `config.json`. Every key can also be overridden by an
environment variable of the same name prefixed with `CITATION_CHECKER_`.

| Block                   | What it controls                                                                                                       |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `entry_extraction`      | OCR model for bibliography-region detection (default: DeepSeek-OCR via local vLLM)                                     |
| `ocr_llm_extract`       | Parser Agent that re-extracts structured fields from each cropped citation block (default: Kimi K2.5 via Bedrock)      |
| `verification_llm`      | Matcher Agent + Class-Specialist Judgers (default: Qwen3-VL-235B via Bedrock); `max_candidates` caps top-K per source  |
| `connectors`            | Connector cache path, DBLP mirror paths, eight academic sources, web-search provider, all related API keys             |
| `citation_parse_method` | `"ocr_llm_extract"` (default, recommended) or `"citation_reparse"`                                                     |

CLI flags worth knowing (`python -m apps.pdf_checker.run --help` for the full list):

| flag                   | default | purpose                                                  |
| ---------------------- | ------- | -------------------------------------------------------- |
| `--paper-workers`      | 2       | parallel papers when `--input` is a directory            |
| `--citation-workers`   | 4       | parallel citations within each paper                     |
| `--connector-workers`  | 8       | parallel connector calls per citation                    |
| `--offline-only`       | off     | skip every online connector (use local DBLP only)        |
| `--extract-only`       | off     | run Stages 1-2 only and dump parsed citations            |
| `--resume`             | off     | skip papers whose `_report.md` already exists            |
| `--save-ocr-artifacts` | off     | persist OCR debug artifacts under `<out>/ocr_artifacts/` |

## Repository layout

```
apps/pdf_checker/         end-to-end PDF entry point (run.py, config.py, ingest/)
packages/connectors/      bibliographic connectors + cache + orchestrator
packages/core/            verifier, cascading agents, field matcher, judges
packages/llm/             provider-agnostic LLM client (bedrock / openai / azure / local vLLM)
scripts/                  evaluation, benchmarking, scoring, sbatch templates
data/synthetic_data/v2/   2,450-citation synthetic benchmark, one JSON per subtype
data/iclr2026_hallucinated/  957 real-world fabricated citations from ICLR 2026 desk rejections
docs/                     taxonomy, metric guide, render-omission notes
tests/                    pytest unit tests for parser, normalizer, agents
```

## Citing

If you use CITETRACER, please cite:

```bibtex
@article{citetracer2026,
  title  = {CITETRACER: Cascading Multi-Agent Citation Hallucination Detection},
  author = {<authors>},
  year   = {2026},
}
```

## License

See [LICENSE](LICENSE).
