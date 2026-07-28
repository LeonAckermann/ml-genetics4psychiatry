"""
Largest dense submatrix analysis for the joined GWAS Z-score matrix.

Given the intermediary wide matrix produced by
``dataloader.pipeline.construct_gwas_phenotype`` (rows = SNPs, columns =
phenotype Z-scores, cells possibly null), this finds column subsets that
maximize the density of the resulting matrix.

Problem framing
---------------
Let ``M`` be the presence matrix (1 = non-null Z, 0 = null). Two objectives are
supported for choosing a column subset ``S`` (``--objective``):

* ``complete`` (default) - largest fully-dense submatrix. Keep only rows that
  are non-null across *all* of ``S``; this gives a complete-case block of shape
  ``n(S) x |S|`` where ``n(S) = |intersection of each column's non-null row
  set|``. ``n(S)`` shrinks monotonically as ``S`` grows, so this is a trade-off
  curve. It is the classic maximum-area all-ones submatrix problem (NP-hard at
  193 columns), so we trace a strong *heuristic* Pareto frontier.

* ``density`` - most non-null entries. Keep *all* rows for the chosen columns
  (the submatrix still has null cells) and maximize the count of non-null
  entries. Because that count is separable over columns (it is just the sum of
  the columns' non-null counts), the optimal k-subset is exactly the ``k``
  densest columns - no search needed. Total entries rises with ``k`` while the
  filled *fraction* (density) falls, so the frontier reports both.

Two-stage approach
------------------
1. Row collapse (the one expensive pass): fold all rows into a histogram over
   distinct null-patterns (a bitmask over the phenotype columns). GWAS
   missingness is structured, so the number of distinct patterns is far
   smaller than the row count. Cached to ``.npz`` so the search is instant and
   re-runnable.
2. Frontier search on the pattern histogram: backward elimination (drop the
   feature costing the most rows) + forward selection (add the feature keeping
   the most rows), taking the better of the two at each feature count. An
   optional local-swap refinement (``--swap-rounds``, off by default) squeezes
   a little more out but costs O(C^3) evaluations — slow at ~193 features.

Runtime scales with the number of *distinct null-patterns* P (not the row
count) and the feature count C: the histogram build is one linear pass over
the file, then backward+forward is ~C^2 subset evaluations, each O(P/64 * C).
At C=193 with P~100k patterns the default frontier takes ~1 minute.

Usage
-----
    # Build the pattern histogram cache from the joined matrix, then analyse:
    python -m script.alignment.dense_submatrix \
        --matrix data/pipeline/input/gwas_pheno/all_z_scores_imputed.txt \
        --cache  data/pipeline/analysis/pattern_hist.npz \
        --out    data/pipeline/analysis/dense_frontier

    # Force a phenotype to always be present (anchor the search):
        ... --require SCZ ADHD

    # Recommend a single subset given a floor on rows or features:
        ... --min-rows 500000        # largest feature set keeping >= 500k rows
        ... --min-features 20        # subset of >= 20 features keeping most rows

    # Maximize non-null entries instead (keep all rows, densest columns):
        ... --objective density
        ... --objective density --min-density 0.8   # most entries at >= 80% filled
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

# Columns of the joined matrix that are NOT phenotype Z-scores.
META_COLS = ("ID", "chrom", "pos", "A0", "A1")

# Tokens the joined matrix may use for a missing cell. "Null" is what
# ``construct_gwas_phenotype`` writes; the rest are common GWAS conventions.
NULL_VALS = ["Null", ".", "NA", "N/A", "NaN", "nan", "NULL", "null", ""]


# ---------------------------------------------------------------------------
# Stage 1: collapse rows into a null-pattern histogram
# ---------------------------------------------------------------------------

def _present_mask(col: pl.Series) -> pl.Series:
    """Boolean 'has a usable Z' mask: not null, and not NaN for float columns."""
    mask = col.is_not_null()
    if col.dtype in (pl.Float32, pl.Float64):
        mask = mask & col.is_not_nan()
    return mask


def build_pattern_histogram(matrix_path: Path, batch_size: int = 500_000,
                            verbose: bool = True):
    """Stream the joined matrix and fold its rows into a histogram over
    distinct null-patterns.

    Returns ``(features, patterns, counts)`` where ``features`` is the list of
    phenotype column names (length C), ``patterns`` is a ``(P, ceil(C/8))``
    uint8 array of bit-packed presence masks (one row per distinct pattern),
    and ``counts`` is a ``(P,)`` int64 array of how many matrix rows share each
    pattern. Bit order matches ``np.packbits`` (big-endian: feature 0 is the
    most-significant bit of byte 0).
    """
    header = pl.read_csv(matrix_path, separator="\t", n_rows=0,
                         null_values=NULL_VALS).columns
    features = [c for c in header if c not in META_COLS]
    del header
    if not features:
        raise ValueError(f"No phenotype columns found in {matrix_path} "
                         f"(header={header}, meta={META_COLS})")
    if verbose:
        print(f"{len(features)} phenotype columns: {', '.join(features[:6])}"
              f"{' ...' if len(features) > 6 else ''}")

    hist: dict[bytes, int] = {}
    total_rows = 0
    lf = pl.scan_csv(matrix_path, separator="\t", null_values=NULL_VALS).select(features)
    for chunk in lf.collect_batches(chunk_size=batch_size):
        # (h, C) bool presence mask -> (h, ceil(C/8)) bit-packed uint8.
        # A cell counts as present only if it is neither null nor NaN
        # (polars treats NaN as distinct from null, but both mean "no Z").
        present = np.column_stack(
            [_present_mask(chunk[c]).to_numpy() for c in features]
        )
        packed = np.packbits(present, axis=1)
        uniq, cnts = np.unique(packed, axis=0, return_counts=True)
        for key, cnt in zip(uniq, cnts):
            b = key.tobytes()
            hist[b] = hist.get(b, 0) + int(cnt)
        total_rows += chunk.height
        if verbose:
            print(f"  ...{total_rows:,} rows, {len(hist):,} distinct patterns so far")

    n_bytes = (len(features) + 7) // 8
    patterns = np.frombuffer(b"".join(hist.keys()), dtype=np.uint8).reshape(-1, n_bytes)
    counts = np.fromiter(hist.values(), dtype=np.int64, count=len(hist))
    if verbose:
        print(f"Collapsed {total_rows:,} rows -> {len(counts):,} distinct patterns")
    return features, patterns, counts


def save_cache(path: Path, features, patterns, counts) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=np.array(features, dtype=object),
                        patterns=patterns, counts=counts)


def load_cache(path: Path):
    data = np.load(path, allow_pickle=True)
    return list(data["features"]), data["patterns"], data["counts"]


# ---------------------------------------------------------------------------
# Stage 2: evaluate subsets and trace the frontier
# ---------------------------------------------------------------------------

class DenseSearch:
    """Evaluate complete-row counts for column subsets against a cached
    pattern histogram, and trace a heuristic (features -> max rows) frontier."""

    def __init__(self, features, patterns, counts):
        self.features = list(features)
        self.C = len(self.features)
        self.counts = counts                        # (P,) int64
        self.total_rows = int(counts.sum())
        # Repack the bit-packed patterns from bytes (P, n_bytes) into 64-bit
        # words (P, W) so each superset test is ~8x fewer element ops. The
        # word layout is self-consistent: subset masks go through the same
        # padding+view, so bitwise AND/equality stay correct.
        self.pat_words = self._to_words(patterns)   # (P, W) uint64
        self.W = self.pat_words.shape[1]
        # Per-feature non-null row counts, folded straight from the histogram:
        # unpack each pattern's bits and weight by how many rows share it.
        bits = np.unpackbits(patterns, axis=1)[:, :self.C]   # (P, C) uint8
        self.present_counts = (counts.astype(np.int64) @ bits).astype(np.int64)

    @staticmethod
    def _to_words(packed_u8: np.ndarray) -> np.ndarray:
        """Pad a (..., n_bytes) uint8 bit-packed array to a multiple of 8
        bytes and reinterpret each 8-byte group as one uint64 word."""
        n_bytes = packed_u8.shape[-1]
        W = (n_bytes + 7) // 8
        padded = np.zeros(packed_u8.shape[:-1] + (W * 8,), dtype=np.uint8)
        padded[..., :n_bytes] = packed_u8
        return np.ascontiguousarray(padded).view(np.uint64).reshape(
            packed_u8.shape[:-1] + (W,))

    def _pack(self, S_bool: np.ndarray) -> np.ndarray:
        """Bit-pack a length-C boolean subset mask into the same word layout."""
        return self._to_words(np.packbits(S_bool))

    def n_complete(self, S_bool: np.ndarray) -> int:
        """Number of matrix rows non-null across *every* selected feature."""
        if not S_bool.any():
            return self.total_rows
        S = self._pack(S_bool)
        superset = np.all((self.pat_words & S) == S, axis=1)
        return int(self.counts[superset].sum())

    def per_feature_present(self) -> np.ndarray:
        """Rows present (non-null) for each single feature -> (C,) int64."""
        return self.present_counts.copy()

    def nonzero_entries(self, S_bool: np.ndarray) -> int:
        """Total non-null cells in the (all-rows x S) submatrix. Because the
        columns don't interact, this is just the sum of the selected columns'
        non-null counts."""
        return int(self.present_counts[S_bool].sum())

    def rows_nonempty(self, S_bool: np.ndarray) -> int:
        """Rows with at least one non-null cell among the selected columns
        (i.e. rows that survive after dropping all-null rows)."""
        if not S_bool.any():
            return 0
        S = self._pack(S_bool)
        any_present = np.any((self.pat_words & S) != 0, axis=1)
        return int(self.counts[any_present].sum())

    def density_frontier(self, required: np.ndarray) -> dict[int, tuple]:
        """Frontier for the 'most non-null entries' objective (all rows kept).

        Total entries is separable over columns, so the optimum k-subset is
        simply the k densest columns (with any ``required`` columns forced in).
        Returns ``{k: (n_entries, subset_bool)}`` for k from |required|..C."""
        rest = [c for c in np.argsort(-self.present_counts, kind="stable")
                if not required[c]]
        order = list(np.where(required)[0]) + rest
        env, S, entries = {}, np.zeros(self.C, dtype=bool), 0
        k0 = int(required.sum())
        for idx, c in enumerate(order, start=1):
            S[c] = True
            entries += int(self.present_counts[c])
            if idx >= max(1, k0):            # only record once all required are in
                env[idx] = (entries, S.copy())
        return env

    # -- frontier tracers ---------------------------------------------------

    def backward_frontier(self, required: np.ndarray, verbose=True) -> dict[int, tuple]:
        """From the full set, repeatedly drop the (non-required) feature whose
        removal recovers the most complete rows."""
        S = np.ones(self.C, dtype=bool)
        frontier = {int(S.sum()): (self.n_complete(S), S.copy())}
        while S.sum() > max(1, required.sum()):
            candidates = np.where(S & ~required)[0]
            if candidates.size == 0:
                break
            best_c, best_n = -1, -1
            for c in candidates:
                S[c] = False
                n = self.n_complete(S)
                if n > best_n:
                    best_n, best_c = n, c
                S[c] = True
            S[best_c] = False
            frontier[int(S.sum())] = (best_n, S.copy())
            if verbose:
                print(f"  backward: {int(S.sum())} features -> {best_n:,} rows")
        return frontier

    def forward_frontier(self, required: np.ndarray, verbose=True) -> dict[int, tuple]:
        """From the required set (or empty), repeatedly add the feature that
        keeps the most complete rows."""
        S = required.copy()
        frontier = {}
        if S.any():
            frontier[int(S.sum())] = (self.n_complete(S), S.copy())
        while S.sum() < self.C:
            candidates = np.where(~S)[0]
            best_c, best_n = -1, -1
            for c in candidates:
                S[c] = True
                n = self.n_complete(S)
                if n > best_n:
                    best_n, best_c = n, c
                S[c] = False
            S[best_c] = True
            frontier[int(S.sum())] = (best_n, S.copy())
            if verbose:
                print(f"  forward: {int(S.sum())} features -> {best_n:,} rows")
        return frontier

    def swap_improve(self, S_bool: np.ndarray, required: np.ndarray,
                     rounds: int = 2) -> tuple[int, np.ndarray]:
        """Local search at fixed feature count: swap one out-feature for one
        in-feature if it increases the complete-row count."""
        S = S_bool.copy()
        best_n = self.n_complete(S)
        for _ in range(rounds):
            improved = False
            ins = np.where(S & ~required)[0]
            outs = np.where(~S)[0]
            for i in ins:
                for o in outs:
                    S[i], S[o] = False, True
                    n = self.n_complete(S)
                    if n > best_n:
                        best_n, improved = n, True
                    else:
                        S[i], S[o] = True, False
            if not improved:
                break
        return best_n, S

    def frontier(self, required: np.ndarray, swap_rounds: int = 1,
                 verbose=True) -> dict[int, tuple]:
        """Upper envelope of backward + forward frontiers, refined by local
        swaps. Returns ``{k: (n_rows, subset_bool)}``."""
        if verbose:
            print("Backward elimination pass...")
        env = self.backward_frontier(required, verbose)
        if verbose:
            print("Forward selection pass...")
        fwd = self.forward_frontier(required, verbose)
        for k, (n, S) in fwd.items():
            if k not in env or n > env[k][0]:
                env[k] = (n, S)
        if swap_rounds > 0:
            if verbose:
                print("Local-swap refinement...")
            for k in list(env):
                n, S = env[k]
                n2, S2 = self.swap_improve(S, required, rounds=swap_rounds)
                if n2 > n:
                    env[k] = (n2, S2)
        return env


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def frontier_records(search: DenseSearch, env: dict[int, tuple]) -> list[dict]:
    """Flatten a frontier into JSON-friendly records, one per feature count."""
    R = search.total_rows
    records = []
    for k in sorted(env):
        n_rows, S = env[k]
        feats = [search.features[i] for i in np.where(S)[0]]
        records.append({
            "n_features": k,
            "n_rows": int(n_rows),
            "row_coverage": n_rows / R if R else 0.0,   # fraction of all SNPs kept
            "area": int(n_rows) * k,                     # dense cells (rows x features)
            "features": feats,
        })
    return records


def pick_by_floor(records: list[dict], min_rows=None, min_features=None):
    """Pick a single recommended subset given a floor on rows or features."""
    if min_rows is not None:
        ok = [r for r in records if r["n_rows"] >= min_rows]
        return max(ok, key=lambda r: r["n_features"]) if ok else None
    if min_features is not None:
        ok = [r for r in records if r["n_features"] >= min_features]
        return max(ok, key=lambda r: r["n_rows"]) if ok else None
    return max(records, key=lambda r: r["area"])   # default: max dense area


def density_records(search: DenseSearch, env: dict[int, tuple]) -> list[dict]:
    """Flatten a density frontier into JSON-friendly records, one per k."""
    R = search.total_rows
    records = []
    for k in sorted(env):
        entries, S = env[k]
        n_eff = search.rows_nonempty(S)          # rows with >=1 non-null cell
        feats = [search.features[i] for i in np.where(S)[0]]
        records.append({
            "n_features": k,
            "n_entries": int(entries),                       # non-null cells
            "n_cells_all": R * k,                            # cells over all rows
            "density_all": entries / (R * k) if R and k else 0.0,
            "n_rows_nonempty": n_eff,                        # rows kept after drop
            "density_nonempty": entries / (n_eff * k) if n_eff and k else 0.0,
            "features": feats,
        })
    return records


def pick_by_density(records: list[dict], min_density=None, min_features=None):
    """Pick a single subset for the density objective. With ``min_density`` (a
    floor on the fraction of non-null cells over all rows) pick the feature set
    with the most non-null entries that still clears the floor; with
    ``min_features`` pick the densest set of at least that many features."""
    if min_density is not None:
        ok = [r for r in records if r["density_all"] >= min_density]
        return max(ok, key=lambda r: r["n_entries"]) if ok else None
    if min_features is not None:
        ok = [r for r in records if r["n_features"] >= min_features]
        return max(ok, key=lambda r: r["density_all"]) if ok else None
    return max(records, key=lambda r: r["n_entries"])   # trivially all features


def save_density_outputs(out_prefix: Path, search: DenseSearch,
                         records: list[dict]) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(out_prefix.with_suffix(".json"), "w") as fh:
        json.dump({
            "objective": "max_nonzero_entries",
            "total_rows": search.total_rows,
            "total_features": search.C,
            "all_features": search.features,
            "frontier": records,
        }, fh, indent=2)

    lines = ["n_features\tn_entries\tdensity_all\tn_rows_nonempty\tdensity_nonempty"]
    lines += [f"{r['n_features']}\t{r['n_entries']}\t{r['density_all']:.6f}\t"
              f"{r['n_rows_nonempty']}\t{r['density_nonempty']:.6f}" for r in records]
    out_prefix.with_suffix(".tsv").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = [r["n_features"] for r in records]
        ent = [r["n_entries"] for r in records]
        den = [r["density_all"] for r in records]
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(ks, ent, marker="o", ms=3, color="C0", label="non-null entries")
        ax1.set_xlabel("Number of features kept")
        ax1.set_ylabel("Non-null entries", color="C0")
        ax1.tick_params(axis="y", labelcolor="C0")
        ax2 = ax1.twinx()
        ax2.plot(ks, den, marker="s", ms=3, color="C3", label="density (all rows)")
        ax2.set_ylabel("Density = entries / (rows x features)", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")
        ax1.set_title("Sparsity trade-off: total non-null entries vs. density")
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_prefix.with_suffix(".png"), dpi=150)
        plt.close(fig)
    except Exception as e:   # plotting is best-effort
        print(f"(skipped plot: {e})")


def save_outputs(out_prefix: Path, search: DenseSearch, records: list[dict]) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(out_prefix.with_suffix(".json"), "w") as fh:
        json.dump({
            "total_rows": search.total_rows,
            "total_features": search.C,
            "all_features": search.features,
            "frontier": records,
        }, fh, indent=2)

    # Compact TSV of the frontier (features x rows), easy to eyeball / plot.
    lines = ["n_features\tn_rows\trow_coverage\tarea"]
    lines += [f"{r['n_features']}\t{r['n_rows']}\t{r['row_coverage']:.6f}\t{r['area']}"
              for r in records]
    out_prefix.with_suffix(".tsv").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = [r["n_features"] for r in records]
        ns = [r["n_rows"] for r in records]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ks, ns, marker="o", ms=3)
        ax.set_xlabel("Number of features kept")
        ax.set_ylabel("Complete-case rows (SNPs)")
        ax.set_title("Density trade-off: features vs. rows with no nulls")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_prefix.with_suffix(".png"), dpi=150)
        plt.close(fig)
    except Exception as e:   # plotting is best-effort
        print(f"(skipped plot: {e})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, help="Joined GWAS Z-score matrix (TSV)")
    p.add_argument("--cache", type=Path, required=True,
                   help=".npz pattern-histogram cache (built from --matrix if missing)")
    p.add_argument("--out", type=Path, required=True,
                   help="Output path prefix (writes .json/.tsv/.png)")
    p.add_argument("--rebuild", action="store_true",
                   help="Rebuild the cache even if it exists")
    p.add_argument("--batch-size", type=int, default=500_000)
    p.add_argument("--objective", choices=["complete", "density"], default="complete",
                   help="'complete' (default): largest fully-non-null submatrix "
                        "(maximize complete-case rows per feature count). 'density': "
                        "keep all rows and maximize the number of non-null entries "
                        "in the chosen columns (the k densest columns).")
    p.add_argument("--require", nargs="*", default=[],
                   help="Feature names forced to always be included (anchor)")
    p.add_argument("--swap-rounds", type=int, default=0,
                   help="Local-swap refinement rounds (default 0 = off). Polishes "
                        "the frontier but costs O(C^3) evaluations — slow at ~193 "
                        "features; enable only if you want the extra squeeze.")
    p.add_argument("--min-rows", type=int, default=None,
                   help="Recommend the largest feature set keeping >= this many rows")
    p.add_argument("--min-features", type=int, default=None,
                   help="Recommend the subset of >= this many features keeping most rows")
    p.add_argument("--min-density", type=float, default=None,
                   help="(density objective) Recommend the feature set with the most "
                        "non-null entries whose density over all rows is >= this floor")
    return p.parse_args()


def main():
    args = parse_args()

    if args.cache.exists() and not args.rebuild:
        print(f"Loading cached pattern histogram from {args.cache}")
        features, patterns, counts = load_cache(args.cache)
    else:
        if not args.matrix:
            raise SystemExit("--matrix is required to build the cache")
        features, patterns, counts = build_pattern_histogram(
            args.matrix, batch_size=args.batch_size)
        save_cache(args.cache, features, patterns, counts)
        print(f"Saved pattern histogram cache to {args.cache}")

    search = DenseSearch(features, patterns, counts)
    print(f"{search.total_rows:,} rows x {search.C} features, "
          f"{len(counts):,} distinct null-patterns")

    unknown = [r for r in args.require if r not in features]
    if unknown:
        raise SystemExit(f"--require names not in matrix: {unknown}")
    required = np.zeros(search.C, dtype=bool)
    for r in args.require:
        required[features.index(r)] = True

    if args.objective == "density":
        env = search.density_frontier(required)
        records = density_records(search, env)
        save_density_outputs(args.out, search, records)
        print(f"\nDensity frontier written to {args.out}.json / .tsv / .png")

        pick = pick_by_density(records, args.min_density, args.min_features)
        if pick:
            print(f"\nRecommended subset ({pick['n_features']} features, "
                  f"{pick['n_entries']:,} non-null entries, "
                  f"density {pick['density_all']:.1%} over all rows / "
                  f"{pick['density_nonempty']:.1%} over non-empty rows):")
            print("  " + ", ".join(pick["features"]))
    else:
        env = search.frontier(required, swap_rounds=args.swap_rounds)
        records = frontier_records(search, env)
        save_outputs(args.out, search, records)
        print(f"\nFrontier written to {args.out}.json / .tsv / .png")

        pick = pick_by_floor(records, args.min_rows, args.min_features)
        if pick:
            print(f"\nRecommended subset ({pick['n_features']} features, "
                  f"{pick['n_rows']:,} rows, coverage {pick['row_coverage']:.1%}):")
            print("  " + ", ".join(pick["features"]))


if __name__ == "__main__":
    main()
