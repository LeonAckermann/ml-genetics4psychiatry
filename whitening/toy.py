"""Generate the toy 2-phenotype matrix + simulated Z used as the default,
fast target for this module — matching the source report's own worked
example (kappa ~= 2.3, no regularisation needed).

Also generates a corresponding second estimate (a noisy perturbation of the
same estimand) so the two-estimate probe has something to run on by default,
and writes both to disk so ``whitening/run.py --config whitening/configs/toy.yaml``
is reproducible without any external data. Scaling to the full 192-trait
intercept matrix is exactly the config swap documented in
whitening/configs/full_intercept_matrix.yaml.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_toy(
    out_dir: str | Path,
    rho: float = 0.55,
    n_snps: int = 5000,
    noise_scale: float = 0.03,
    seed: int = 0,
) -> dict[str, Path]:
    """Write Sigma1.csv, Sigma2.csv, Z.txt (+ groups.csv) for a 2-trait toy.

    Sigma1 is the exact 2x2 correlation matrix [[1, rho], [rho, 1]]
    (kappa = (1+rho)/(1-rho) ~= 2.3 at rho=0.4, matching the source report).
    Sigma2 is Sigma1 perturbed by symmetric noise of scale ``noise_scale``,
    standing in for a second pairwise estimate — small enough that the
    two-estimate probe reports "identified" by default, and easy to dial up to
    see the probe trip.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = ["PHENO_A", "PHENO_B"]

    Sigma1 = np.array([[1.0, rho], [rho, 1.0]])

    rng = np.random.default_rng(seed)
    E = rng.normal(scale=noise_scale, size=(2, 2))
    E = 0.5 * (E + E.T)
    np.fill_diagonal(E, 0.0)
    Sigma2 = Sigma1 + E

    z = rng.multivariate_normal(np.zeros(2), Sigma1, size=n_snps)
    z_df = pd.DataFrame(z, columns=labels)
    z_df.insert(0, "ID", [f"rs{i}" for i in range(n_snps)])
    z_df.insert(1, "A0", rng.choice(list("ACGT"), size=n_snps))
    # Half the SNPs get a strand-ambiguous allele pair (A/T), by construction,
    # so whitening/sign.py's allele-ambiguity probe has something to find.
    a1 = np.where(
        np.arange(n_snps) % 2 == 0,
        np.where(z_df["A0"] == "A", "T", np.where(z_df["A0"] == "T", "A",
                 np.where(z_df["A0"] == "C", "G", "C"))),
        rng.choice(list("ACGT"), size=n_snps),
    )
    z_df["A1"] = a1

    sigma1_path = out_dir / "Sigma1.csv"
    sigma2_path = out_dir / "Sigma2.csv"
    z_path = out_dir / "Z.txt"
    groups_path = out_dir / "groups.csv"

    pd.DataFrame(Sigma1, index=labels, columns=labels).to_csv(sigma1_path)
    pd.DataFrame(Sigma2, index=labels, columns=labels).to_csv(sigma2_path)
    z_df.to_csv(z_path, sep="\t", index=False)
    pd.DataFrame({"trait": labels, "group": ["TOY", "TOY"]}).to_csv(groups_path, index=False)

    return {
        "sigma1": sigma1_path, "sigma2": sigma2_path,
        "z": z_path, "groups": groups_path,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="data/pipeline/input/whitening_toy")
    p.add_argument("--rho", type=float, default=0.55)
    p.add_argument("--n-snps", type=int, default=5000)
    p.add_argument("--noise-scale", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    paths = make_toy(args.out_dir, args.rho, args.n_snps, args.noise_scale, args.seed)
    for k, v in paths.items():
        print(f"{k}: {v}")
