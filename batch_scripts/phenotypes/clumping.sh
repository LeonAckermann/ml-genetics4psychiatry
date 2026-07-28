#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J clumping
#SBATCH -o ./logs/clumping_%A_%a.out
#SBATCH -e ./logs/clumping_%A_%a.err

module load Python/3.12.9
module load plink/2.0.0-a.6.11

source ./pasteur/bin/activate
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

#print working directory
echo "Current working directory: $(pwd)"

echo "Starting clumping "
python3.12 main.py --config experiments/phenotypes/phenotype_clumping.yaml
echo "Pipeline for clumping completed successfully."

#print done
echo "Clumping script completed successfully."

