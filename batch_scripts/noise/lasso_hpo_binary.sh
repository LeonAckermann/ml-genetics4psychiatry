#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J lasso_hpo_binary
#SBATCH -o ./logs/lasso_hpo_binary_%A_%a.out
#SBATCH -e ./logs/lasso_hpo_binary_%A_%a.err

module load Python/3.12.9

source ./pasteur/bin/activate
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

#print working directory
echo "Current working directory: $(pwd)"

echo "Starting lasso HPO for illness ${i} ..."
python3.12 main.py --config experiments/hpo/lasso_hpo_binary.yaml

#print done
echo "Lasso HPO binary script completed successfully."