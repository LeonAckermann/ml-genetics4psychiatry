#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J lasso_phenotypes_pca
#SBATCH -o ./logs/lasso_phenotypes_pca_%A_%a.out
#SBATCH -e ./logs/lasso_phenotypes_pca_%A_%a.err

module load Python/3.12.9

source ./pasteur/bin/activate
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

#print working directory
echo "Current working directory: $(pwd)"

echo "Starting lasso for illness ${i} ..."
python3.12 main.py --config experiments/phenotypes/lasso_all_phenotypes_pca.yaml

#print done
echo "Lasso script completed successfully."