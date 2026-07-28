#!/usr/bin/env bash

#SBATCH --mem=300000
#SBATCH -J dense_submatrix
#SBATCH -N 1
#SBATCH --cpus-per-task=12
#SBATCH -o ./logs/dense_submatrix_%A_%a.out
#SBATCH -e ./logs/dense_submatrix_%A_%a.err


module load Python/3.12.9
module load plink/2.0.0-a.6.11

source ./pasteur/bin/activate

# record run time
start_time=$(date +%s)

echo "Starting dense submatrix construction..."
# unzip the data

python3.12 -m script.alignment.dense_submatrix \
  --matrix data/pipeline/input/gwas_pheno/all_z_scores_imputed.txt \
  --cache  data/pipeline/analysis/pattern_hist.npz \
  --out    data/pipeline/analysis/dense_frontier \
  --min-features 100

# record end time and calculate elapsed time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Total time taken: $elapsed_time seconds"



