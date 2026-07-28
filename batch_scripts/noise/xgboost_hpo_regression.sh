#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J xgboost_noise_regression
#SBATCH -o ./logs/xgboost_noise_regression_%A_%a.out
#SBATCH -e ./logs/xgboost_noise_regression_%A_%a.err
#SBATCH -p gpu
#SBATCH --gres=gpu:A100


module load Python/3.12.9

source ./pasteur/bin/activate
#export TABPFN_ALLOW_CPU_LARGE_DATASET=1

#print working directory
echo "Current working directory: $(pwd)"

echo "Starting xgboost Noise for illness ${i} ..."
python3.12 main.py --config experiments/noise/xgboost_noise_regression.yaml

#print done
echo "XGBoost Noise regression script completed successfully."