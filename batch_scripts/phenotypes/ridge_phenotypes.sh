#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J ridge_phenotypes
#SBATCH -o ./logs/ridge_phenotypes_%A_%a.out
#SBATCH -e ./logs/ridge_phenotypes_%A_%a.err

module load Python/3.12.9

source ./pasteur/bin/activate
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

#print working directory
echo "Current working directory: $(pwd)"

echo "Starting ridge regression for illness ${i} ..."
python3.12 main.py --config experiments/phenotypes/ridge_all_phenotypes.yaml

#print done
echo "Ridge regression script completed successfully."