#!/bin/bash
# Evaluate H1..H6 with the rule-based Valid + Hallucinated agents.
# Outputs: results/eval_H<idx>_rule_based.json
set -euo pipefail

cd "$(dirname "$0")/.."

for H_IDX in 1 2 3 4 5 6; do
    echo "[$(date)] Running H${H_IDX}"
    python scripts/test_synthetic_samples.py \
        --samples data/synthetic_data/v2/H${H_IDX}.json \
        --meta    data/synthetic_data/v2/meta.json \
        --workers 8 \
        --rule-based-valid-agent \
        --rule-based-hallucinated-agent \
        --out     results/eval_H${H_IDX}_rule_based.json
    echo "[$(date)] Finished H${H_IDX}"
done
