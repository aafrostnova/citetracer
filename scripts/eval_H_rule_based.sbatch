#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=64g  # Requested Memory
#SBATCH -p superpod-a100  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 2-0  # Job time limit
#SBATCH --array=1-6
#SBATCH -o slurm-%A_%a.out  # %A = array job id, %a = array task id

# Run H1..H6 using RuleBasedValidAgent + RuleBasedHallucinatedAgent.
#
# Submit as array (parallel):   sbatch scripts/eval_H_rule_based.sbatch
# Run sequentially (no slurm):  bash   scripts/eval_H_rule_based.sbatch
#
# Results: results/eval_H<idx>_rule_based.json

cd /project/pi_shiqingma_umass_edu/mingzheli/Citation_Hallucination_Detection

run_one() {
    local H_IDX=$1
    echo "[$(date)] Starting H${H_IDX}"
    python scripts/test_synthetic_samples.py \
        --samples data/synthetic_data/v2/H${H_IDX}.json \
        --meta data/synthetic_data/v2/meta.json \
        --workers 4 \
        --rule-based-valid-agent \
        --rule-based-hallucinated-agent \
        --out results/eval_H${H_IDX}_rule_based.json
    echo "[$(date)] Finished H${H_IDX}"
}

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # Running under sbatch as an array task: handle one index
    run_one "${SLURM_ARRAY_TASK_ID}"
else
    # Running directly (bash scripts/eval_H_rule_based.sbatch): loop 1..6
    for H_IDX in 1 2 3 4 5 6; do
        run_one "${H_IDX}"
    done
fi
