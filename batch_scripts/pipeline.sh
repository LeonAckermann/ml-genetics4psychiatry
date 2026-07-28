#!/usr/bin/env bash

#SBATCH --mem=200000
#SBATCH -J pipeline_scz
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH -o ./logs/pipeline_scz_%A_%a.out
#SBATCH -e ./logs/pipeline_scz_%A_%a.err


module load Python/3.12.9
module load plink/2.0.0-a.6.11

source ./pasteur/bin/activate

illness=(scz)

for i in "${illness[@]}"
do
  echo "Starting pipeline for ${i} data processing..."
  echo "Current working directory: $(pwd)"
  # unzip the data
  python3.12 ./main.py --config ./experiments/pipeline_scz.yaml
  echo "Pipeline for ${i} completed successfully."
done


