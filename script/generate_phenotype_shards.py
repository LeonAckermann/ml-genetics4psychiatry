#!/usr/bin/env python3
"""Generate per-shard experiment configs for a phenotype sweep.

Resolves the full phenotype list (``include=1`` in info.csv and present as a
column in the gwas_pheno matrix), splits it into fixed-size chunks, and writes
one config file per chunk with ``data.illness`` set to that chunk's traits.
An explicit ``data.illness`` list short-circuits the density/info.csv selection
in main.py, so each file runs an independent nested-CV sweep over just its own
~25 phenotypes — ideal for a SLURM array job (embarrassingly parallel).

Run from the repository root (so the relative data paths in the config
resolve), with the cluster venv active (needs torch via the dataloader import):

    python scripts/generate_phenotype_shards.py \
        --config experiments/phenotypes/residual_dnn_all_phenotypes.yaml \
        --shard-size 25
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

# Allow running as `python scripts/generate_phenotype_shards.py` from the repo
# root: put the repo root (this file's parent's parent) on the import path so
# the `dataloader` package resolves regardless of the invocation directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader import included_phenotype_columns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Base experiment YAML.")
    ap.add_argument("--shard-size", type=int, default=25,
                    help="Phenotypes per shard (default: 25).")
    ap.add_argument("--out-dir", default=None,
                    help="Output dir for shard configs (default: <config_dir>/shards).")
    args = ap.parse_args()

    base_path = Path(args.config)
    cfg = yaml.safe_load(base_path.read_text())
    data = cfg["data"]

    phenos = included_phenotype_columns(data["gwas_pheno_path"], data["info_csv_path"])
    if not phenos:
        raise SystemExit(
            "No phenotypes resolved from info.csv — check gwas_pheno_path / "
            "info_csv_path and the include column."
        )

    out_dir = Path(args.out_dir) if args.out_dir else base_path.parent / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = [phenos[i:i + args.shard_size]
              for i in range(0, len(phenos), args.shard_size)]
    stem = base_path.stem

    for i, chunk in enumerate(chunks):
        shard = copy.deepcopy(cfg)
        shard["data"]["illness"] = chunk        # explicit list -> skips density/info.csv selection
        shard["data"].pop("min_density", None)  # unambiguous: illness drives the list
        (out_dir / f"{stem}_shard_{i}.yaml").write_text(
            yaml.safe_dump(shard, sort_keys=False)
        )

    n = len(chunks)
    print(f"{len(phenos)} phenotypes -> {n} shards of up to {args.shard_size} "
          f"in {out_dir}/")
    print("Submit with:")
    print(f"  sbatch --array=0-{n - 1} batch_scripts/residual_dnn_shards.sh")


if __name__ == "__main__":
    main()
