#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -p cpu
#SBATCH -t 12:00:00
#SBATCH -o slurm-eval-R-%j.out
#SBATCH -J eval_R_rule_based

cd /project/pi_shiqingma_umass_edu/mingzheli/Citation_Hallucination_Detection

set -e

# R*_plus: no meta entries, use --expected-verdict VALID
for f in R2_plus R3_plus; do
  echo "=== Running ${f} ==="
  python scripts/test_synthetic_samples.py \
    --samples data/synthetic_data/v2/${f}.json \
    --expected-verdict VALID \
    --workers 8 \
    --rule-based-valid-agent \
    --rule-based-hallucinated-agent \
    --out results/eval_${f}_rule_based.json
done

# R1/R2/R3: have meta entries
for f in R1 R2 R3; do
  echo "=== Running ${f} ==="
  python scripts/test_synthetic_samples.py \
    --samples data/synthetic_data/v2/${f}.json \
    --meta data/synthetic_data/v2/meta.json \
    --workers 8 \
    --rule-based-valid-agent \
    --rule-based-hallucinated-agent \
    --out results/eval_${f}_rule_based.json
done

echo "=== All done ==="
