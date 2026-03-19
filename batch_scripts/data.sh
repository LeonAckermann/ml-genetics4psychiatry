#!/usr/bin/env bash

#SBATCH --mem=10000
#SBATCH -J data
#SBATCH -o ../logs/data_%A_%a.out
#SBATCH -e ../logs/data_%A_%a.err

echo "Current working directory: $(pwd)"

# unzip the data
module load Python/3.12.9

source ../pasteur/bin/activate

#print working directory
echo "Current working directory: $(pwd)"

python3.12 ../scripts/pyscripts/data.py 

#print done
echo "TabPFN script completed successfully."

