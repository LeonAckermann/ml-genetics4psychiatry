#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J generate_shards
#SBATCH -o ./logs/generate_shards_%A_%a.out
#SBATCH -e ./logs/generate_shards_%A_%a.err


module load Python/3.12.9

source ./pasteur/bin/activate
export TABPFN_ALLOW_CPU_LARGE_DATASET=1


echo "Current working directory: $(pwd)"

python3.12 script/generate_phenotype_shards.py --config experiments/phenotypes/residual_dnn_all_phenotypes.yaml --shard-size 25


echo "Shard generation script completed successfully."
