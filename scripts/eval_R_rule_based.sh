#!/bin/bash
# Evaluate R1, R2, R3 with the rule-based Valid + Hallucinated agents.
# (R*_plus split was merged back into the parent files; no separate _plus runs.)
# Outputs: results/eval_R<idx>_rule_based.json
set -euo pipefail

cd "$(dirname "$0")/.."

for R_IDX in 1 2 3; do
    echo "[$(date)] Running R${R_IDX}"
    python scripts/test_synthetic_samples.py \
        --samples data/synthetic_data/v2/R${R_IDX}.json \
        --meta    data/synthetic_data/v2/meta.json \
        --workers 8 \
        --rule-based-valid-agent \
        --rule-based-hallucinated-agent \
        --out     results/eval_R${R_IDX}_rule_based.json
    echo "[$(date)] Finished R${R_IDX}"
done
