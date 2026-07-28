#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J ridge_hpo_regression
#SBATCH -o ./logs/ridge_hpo_regression_%A_%a.out
#SBATCH -e ./logs/ridge_hpo_regression_%A_%a.err

module load Python/3.12.9

source ./pasteur/bin/activate
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

#print working directory
echo "Current working directory: $(pwd)"

echo "Starting ridge HPO for illness ${i} ..."
python3.12 main.py --config experiments/hpo/ridge_hpo_regression.yaml

#print done
echo "Ridge HPO regression script completed successfully."