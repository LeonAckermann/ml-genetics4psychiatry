#!/usr/bin/env bash

#SBATCH --mem=32000
#SBATCH -J whitening_preflight
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH -o ./logs/whitening_preflight_%A_%a.out
#SBATCH -e ./logs/whitening_preflight_%A_%a.err

# Whitening pre-flight diagnostics — checks Sigma/the intercept matrix
# (alignment, symmetry, missing entries, diagonal, spectrum, PD-ness, sign
# consistency, null calibration) before it is used to whiten anything.
# See whitening/README.md. Config defaults to the full 192-trait intercept
# matrix; override with: sbatch whitening_preflight.sh <config.yaml>

module load Python/3.12.9

source ./pasteur/bin/activate

CONFIG="${1:-whitening/configs/full_intercept_matrix.yaml}"

start_time=$(date +%s)

echo "Running whitening pre-flight with config: ${CONFIG}"
python3.12 -m whitening.run --config "${CONFIG}"
status=$?

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Total time taken: $elapsed_time seconds"

if [ $status -ne 0 ]; then
  echo "Whitening pre-flight reported failures (exit ${status}) — see results/whitening/*/whitening_report.txt"
fi
exit $status
