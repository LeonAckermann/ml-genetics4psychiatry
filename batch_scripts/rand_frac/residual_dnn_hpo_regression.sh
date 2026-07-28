#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J residual_dnn_rand_regression
#SBATCH -o ./logs/residual_dnn_rand_regression_%A_%a.out
#SBATCH -e ./logs/residual_dnn_rand_regression_%A_%a.err
#SBATCH -p gpu
#SBATCH --gres=gpu:A100

module load Python/3.12.9

source ./pasteur/bin/activate
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

#print working directory
echo "Current working directory: $(pwd)"

echo "Starting residual DNN Rand for illness ${i} ..."
python3.12 main.py --config experiments/rand_frac/residual_dnn_rand_regression.yaml

#print done
echo "Residual DNN Rand regression script completed successfully."