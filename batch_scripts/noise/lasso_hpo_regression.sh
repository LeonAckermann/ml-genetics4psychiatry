#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J lasso_noise_regression
#SBATCH -o ./logs/lasso_noise_%A_%a.out
#SBATCH -e ./logs/lasso_noise_%A_%a.err

module load Python/3.12.9

source ./pasteur/bin/activate
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

#print working directory
echo "Current working directory: $(pwd)"

echo "Starting lasso HPO for illness ${i} ..."
python3.12 main.py --config experiments/noise/lasso_noise_regression.yaml

#print done
echo "Lasso HPO regression script completed successfully."