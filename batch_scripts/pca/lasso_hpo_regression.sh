#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J lasso_pca_regression
#SBATCH -o ./logs/lasso_pca_regression_%A_%a.out
#SBATCH -e ./logs/lasso_pca_regression_%A_%a.err

module load Python/3.12.9

source ./pasteur/bin/activate
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

#print working directory
echo "Current working directory: $(pwd)"

echo "Starting lasso PCA for illness ${i} ..."
python3.12 main.py --config experiments/pca/lasso_pca_regression.yaml

#print done
echo "Lasso PCA regression script completed successfully."