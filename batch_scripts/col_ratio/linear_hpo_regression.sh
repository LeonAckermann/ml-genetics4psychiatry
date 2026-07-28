#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J linear_col_ratio_regression
#SBATCH -o ./logs/linear_col_ratio_regression_%A_%a.out
#SBATCH -e ./logs/linear_col_ratio_regression_%A_%a.err

module load Python/3.12.9

source ./pasteur/bin/activate
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

#print working directory
echo "Current working directory: $(pwd)"

echo "Starting linear Col Ratio for illness ${i} ..."
python3.12 main.py --config experiments/col_ratio/linear_col_ratio_regression.yaml

#print done
echo "Linear Col Ratio regression script completed successfully."