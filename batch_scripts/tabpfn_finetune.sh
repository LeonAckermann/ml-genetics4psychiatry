#!/usr/bin/env bash

#SBATCH --mem=100000
#SBATCH -J tabpfn_finetune
#SBATCH -o ./logs/tabpfn_finetune_%A_%a.out
#SBATCH -e ./logs/tabpfn_finetune_%A_%a.err
#SBATCH -p gpu           
#SBATCH --gres=gpu:A100

module load Python/3.12.9

source ./pasteur/bin/activate
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

#print working directory
echo "Current working directory: $(pwd)"


echo "Starting tabpfn finetune"
python3.12 main.py --config experiments/tabpfn_finetune.yaml
echo "Pipeline for tabpfn finetune completed successfully."


#print done
echo "TabPFN finetune script completed successfully."

