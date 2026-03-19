#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J unzip
#SBATCH -o ../logs/unzip_%A_%a.out
#SBATCH -e ../logs/unzip_%A_%a.err

echo "Current working directory: $(pwd)"

# unzip the data
tar -xzf ../data/tmpDATA-Leon/data_clumping_and_sumstats_for_leon.tar.gz -C ../data/tmpDATA-Leon

echo "Unzip completed successfully."

