#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J linear_phenotypes
#SBATCH -o ./logs/linear_phenotypes_%A_%a.out
#SBATCH -e ./logs/linear_phenotypes_%A_%a.err

module load Python/3.12.9

source ./pasteur/bin/activate
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

#print working directory
echo "Current working directory: $(pwd)"

echo "Starting linear regression for illness ${i} ..."
python3.12 main.py --config experiments/phenotypes/linear_regression_all_phenotypes_pca.yaml

#print done
echo "Linear regression script completed successfully."