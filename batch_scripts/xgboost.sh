#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J xgboost
#SBATCH -o ./logs/xgboost_%A_%a.out
#SBATCH -e ./logs/xgboost_%A_%a.err

module load Python/3.12.9

source ./pasteur/bin/activate
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

#print working directory
echo "Current working directory: $(pwd)"

echo "Starting xgboost for illness ${i} ..."
python3.12 main.py --config experiments/xgboost.yaml
echo "Pipeline for ${i} completed successfully."

#print done
echo "XGBoost script completed successfully."

