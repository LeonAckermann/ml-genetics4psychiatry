#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J xgboost_hpo_lucas
#SBATCH -o ./logs/xgboost_hpo_lucas_%A_%a.out
#SBATCH -e ./logs/xgboost_hpo_lucas_%A_%a.err
#SBATCH -p gpu
#SBATCH --gres=gpu:A100


module load Python/3.12.9

source ./pasteur/bin/activate
#export TABPFN_ALLOW_CPU_LARGE_DATASET=1

#print working directory
echo "Current working directory: $(pwd)"

echo "Starting xgboost HPO for illness ${i} ..."
python3.12 main.py --config experiments/lucas/xgboost_hpo_lucas.yaml

#print done
echo "XGBoost HPO lucas script completed successfully."