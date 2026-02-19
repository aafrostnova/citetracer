# Citation Hallucination Detection

Prototype-informed implementation of a citation integrity checker with two pipelines:

- `apps/source_checker`: checks LaTeX + BibTeX source
- `apps/pdf_checker`: checks rendered PDF references

Both share a common verifier in `packages/core` and connector stack in `packages/connectors`.

## Quickstart

```bash
python -m apps.source_checker.run --input data/fixtures/sample_source --out artifacts/sample_source_report.json
python -m apps.pdf_checker.run --input data/fixtures/sample_pdf/sample.pdf --out artifacts/sample_pdf_report.json
python -m packages.eval.run --suite synthetic_stress --out artifacts/eval
```

## Project Layout

- `apps/`: source and PDF checker applications
- `packages/core/`: models, normalization, matching, adjudication, report logic
- `packages/connectors/`: bibliographic connectors + cache + request policy
- `packages/eval/`: benchmark runner and metrics
- `data/`: fixtures, manifests, labels, and offline DBLP mirror
- `docs/`: architecture, schema, evaluation protocol, annotation guidelines
