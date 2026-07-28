#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J residual_dnn_shards
#SBATCH -o ./logs/residual_dnn_shard_%A_%a.out
#SBATCH -e ./logs/residual_dnn_shard_%A_%a.err
#SBATCH -p gpu
#SBATCH --gres=gpu:A100

# One array task per shard. Each shard config runs a nested-CV HPO sweep over
# its own ~25 phenotypes. Submit with the array range printed by
# scripts/generate_phenotype_shards.py, e.g.:
#     sbatch --array=0-9 batch_scripts/residual_dnn_shards.sh
# Add %K to cap concurrent tasks (A100s), e.g. --array=0-9%5.

module load Python/3.12.9

source ./pasteur/bin/activate
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

CONFIG="experiments/phenotypes/shards/residual_dnn_all_phenotypes_shard_${SLURM_ARRAY_TASK_ID}.yaml"

echo "Current working directory: $(pwd)"
echo "Shard task ${SLURM_ARRAY_TASK_ID}: config ${CONFIG}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: shard config not found: ${CONFIG}" >&2
    echo "Did you run scripts/generate_phenotype_shards.py first?" >&2
    exit 1
fi

python3.12 main.py --config "${CONFIG}"

echo "Residual DNN shard ${SLURM_ARRAY_TASK_ID} completed successfully."
