#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J tabpfn_all_phenotypes
#SBATCH -o ./logs/tabpfn_all_phenotypes_%A_%a.out
#SBATCH -e ./logs/tabpfn_all_phenotypes_%A_%a.err
#SBATCH -p gpu
#SBATCH --gres=gpu:A100

module load Python/3.12.9

source ./pasteur/bin/activate
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

#print working directory
echo "Current working directory: $(pwd)"

echo "Starting tabpfn HPO for illness ${i} ..."
python3.12 main.py --config experiments/phenotypes/tabpfn_all_phenotypes.yaml

#print done
echo "TabPFN HPO regression script completed successfully."