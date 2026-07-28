#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J ridge_col_ratio_regression
#SBATCH -o ./logs/ridge_col_ratio_regression_%A_%a.out
#SBATCH -e ./logs/ridge_col_ratio_regression_%A_%a.err

module load Python/3.12.9

source ./pasteur/bin/activate
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

#print working directory
echo "Current working directory: $(pwd)"

echo "Starting ridge Col Ratio for illness ${i} ..."
python3.12 main.py --config experiments/col_ratio/ridge_col_ratio_regression.yaml

#print done
echo "Ridge Col Ratio regression script completed successfully."