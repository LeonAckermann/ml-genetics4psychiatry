#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J tabpfn
#SBATCH -o ./logs/tabpfn_%A_%a.out
#SBATCH -e ./logs/tabpfn_%A_%a.err
#SBATCH -p gpu           
#SBATCH --gres=gpu:A100

module load Python/3.12.9

source ./pasteur/bin/activate
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

#print working directory
echo "Current working directory: $(pwd)"

illness=(scz)
for i in "${illness[@]}"
do
  echo "Starting tabpfn for illness ${i} ..."
  python3.12 main.py --config experiments/tabpfn_${i}.yaml
  echo "Pipeline for ${i} completed successfully."
done

#print done
echo "TabPFN script completed successfully."

