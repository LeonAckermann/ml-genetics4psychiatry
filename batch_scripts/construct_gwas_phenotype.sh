#!/usr/bin/env bash

#SBATCH --mem=200000
#SBATCH -J construct_gwas_phenotype
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH -o ./logs/construct_gwas_phenotype_%A_%a.out
#SBATCH -e ./logs/construct_gwas_phenotype_%A_%a.err


module load Python/3.12.9
module load plink/2.0.0-a.6.11

source ./pasteur/bin/activate

# record run time
start_time=$(date +%s)

echo "Starting pipeline for ${i} data processing..."
echo "Current working directory: $(pwd)"
# unzip the data
python3.12 ./main.py --config ./experiments/pipeline/construct_gwas_phenotype.yaml
echo "Pipeline for ${i} completed successfully."

# record end time and calculate elapsed time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Total time taken: $elapsed_time seconds"
