# Evaluation Protocol

## Suites

- `neurips_iclr_subset`
- `arxiv_rolling_weekly`
- `synthetic_stress`

## Metrics

- Hallucination: precision / recall / F1 (`SUSPECTED_HALLUCINATION`)
- Flawed citation: precision / recall / F1 (`FLAWED_CITATION`)
- Calibration: Expected Calibration Error (ECE)
- Operational: unresolved rate, manual review load, top-k alert precision

## Acceptance Targets (v1)

- Hallucination precision >= 0.92
- Top-10 alert precision >= 0.85
- Source unresolved rate <= 0.05
- PDF extraction field F1 >= 0.85 on clean PDFs
