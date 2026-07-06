import numpy as np
from pathlib import Path
import functools
import gc
import shlex
import shutil
import subprocess
import sys

import pandas as pd
import polars as pl


from dataloader.preprocess import load_txt, load_txt_polars

def construct_gwas_mri(path, output_path, chunk_size=10000, total_chunks=None, polars=False, value="T_STAT"):
    path = Path(path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    allres_files = sorted(path.rglob("allRES.txt"))
    if not allres_files:
        raise FileNotFoundError(f"No allRES.txt files found under {path}")

    print(f"Found {len(allres_files)} allRES.txt files")

    NULL_VALS = [".", "NA", "N/A", "NaN", "nan", "NULL", "null", ""]

    # --- Pass 1: find the intersection of (ID, A1, OMITTED) keys ---
    print("Pass 1: finding common SNPs across all files...")
    common_keys: set | None = None
    for file in allres_files:
        df = (
            pl.scan_csv(str(file), separator="\t", null_values=NULL_VALS)
            .select(["ID", "A1", "OMITTED"])
            .collect()
            .unique(subset=["ID", "A1", "OMITTED"])
        )
        keys = set(zip(df["ID"].to_list(), df["A1"].to_list(), df["OMITTED"].to_list()))
        del df
        gc.collect()
        if common_keys is None:
            common_keys = keys
        else:
            common_keys &= keys

    n_common = len(common_keys)
    print(f"Common SNPs: {n_common}")

    # Build a sorted reference frame for stable row alignment
    common_ids      = [k[0] for k in common_keys]
    common_a1       = [k[1] for k in common_keys]
    common_omitted  = [k[2] for k in common_keys]
    del common_keys
    gc.collect()

    common_df = pl.DataFrame({
        "ID":      common_ids,
        "A1":      common_a1,
        "OMITTED": common_omitted,
    }).sort(["ID", "A1", "OMITTED"])
    del common_ids, common_a1, common_omitted
    gc.collect()

    # --- Pass 2: pre-allocate matrix and fill one column per file ---
    print(f"Pass 2: extracting {value} columns...")
    n_files = len(allres_files)
    t_stat_matrix = np.empty((n_common, n_files), dtype=np.float64)
    phenotype_names = []

    duplicates_log: list[pl.DataFrame] = []

    for i, file in enumerate(allres_files):
        phenotype = file.parent.name
        phenotype_names.append(phenotype)

        raw = (
            pl.scan_csv(str(file), separator="\t", null_values=NULL_VALS)
            .select(["ID", "A1", "OMITTED", value])
            .collect()
        )

        is_dup = raw.select(pl.struct("ID", "A1", "OMITTED").is_duplicated()).to_series()
        dups = raw.filter(is_dup).with_columns(pl.lit(phenotype).alias("phenotype"))
        if len(dups) > 0:
            duplicates_log.append(dups)

        df = (
            raw.group_by(["ID", "A1", "OMITTED"])
            .agg(pl.col(value).sort_by(pl.col(value).abs()).last())
            .join(common_df, on=["ID", "A1", "OMITTED"], how="inner")
            .sort(["ID", "A1", "OMITTED"])
        )
        del raw
        t_stat_matrix[:, i] = df[value].to_numpy()
        del df
        gc.collect()

        if (i + 1) % 100 == 0 or i == n_files - 1:
            print(f"  Processed {i + 1}/{n_files} files")

    # Save duplicate SNPs log
    if duplicates_log:
        dup_path = output_path.parent / "duplicate_snps.tsv"
        pl.concat(duplicates_log).write_csv(str(dup_path), separator="\t")
        print(f"Saved {sum(len(d) for d in duplicates_log)} duplicate rows to {dup_path}")
    del duplicates_log
    gc.collect()

    # A1 (allRES) → A0, OMITTED (allRES) → A1
    merged = common_df.rename({"A1": "A0", "OMITTED": "A1"})
    for i, name in enumerate(phenotype_names):
        merged = merged.with_columns(pl.Series(name=name, values=t_stat_matrix[:, i]))

    del t_stat_matrix, common_df
    gc.collect()

    n_rows_merged = merged.shape[0]
    n_unique_ids = merged.select(pl.col("ID").n_unique()).item()

    stats = {
        "n_files": n_files,
        "n_rows_merged": n_rows_merged,
        "n_unique_ids": n_unique_ids,
    }

    print(f"Final merged shape: {merged.shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_csv(str(output_path), separator="\t")
    del merged
    gc.collect()
    print(f"Saved merged data to {output_path}")
    return stats

def _gwas_phenotype_illness_name(file: Path) -> str:
    return file.stem[2:] if file.stem.startswith("z_") else file.stem


def construct_gwas_phenotype(input_path, output_path, how="inner", join_key="chrom", verbose=True):
    """Join per-illness GWAS summary statistic files into one Z-score matrix.

    Reads every ``z_<ILLNESS>.txt`` file under ``input_path``, harmonizes the
    A0/A1 allele coding against the first illness (alphabetically) found, and
    writes a single wide matrix with one Z-score column named after each
    illness (e.g. ``ADHD``, ``SCZ``) plus the reference illness's rsID
    (as ``ID``), chrom, pos, A0 and A1.

    ``join_key`` controls how SNPs are matched across files:
      - "chrom": join on (chrom, pos) [default]
      - "rs":    join on rsID

    Comparing the row count between the two modes is a useful diagnostic:
    a large gap (e.g. millions via "rs" vs. a handful via "chrom") points to
    a genome build mismatch between input files, since rsIDs are build
    independent while chrom/pos are not.

    Alleles reported in swapped order relative to the reference have their
    Z score sign flipped; SNPs whose alleles can't be reconciled in either
    order are dropped from the output.
    """
    if join_key not in ("chrom", "rs"):
        raise ValueError(f"join_key must be 'chrom' or 'rs', got {join_key!r}")

    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    files = sorted(input_path.glob("z_*.txt"))
    if not files:
        raise FileNotFoundError(f"No z_*.txt files found in {input_path}")

    illnesses = [_gwas_phenotype_illness_name(f) for f in files]
    join_how = "inner" if how == "inner" else "full"
    key_cols = ["chrom", "pos"] if join_key == "chrom" else ["ID"]
    dedupe_subset = ["chrom", "pos"] if join_key == "chrom" else ["rsID"]

    if verbose:
        print(f"Found {len(files)} illness files: {', '.join(illnesses)}")
        print(f"Joining on: {join_key}")

    per_file_frames = {}
    line_counts = {}
    dupe_counts = {}
    for file, illness in zip(files, illnesses):
        df = pl.read_csv(
            file,
            separator="\t",
            columns=["chrom", "rsID", "pos", "A0", "A1", "Z"],
            schema_overrides={
                "chrom": pl.Int8, "pos": pl.Int32,
                "rsID": pl.Utf8, "A0": pl.Utf8, "A1": pl.Utf8, "Z": pl.Float64,
            },
        )
        n_rows = df.height
        df = df.unique(subset=dedupe_subset, keep="first", maintain_order=True)
        n_dupes = n_rows - df.height

        per_file_frames[illness] = df
        line_counts[illness] = n_rows
        dupe_counts[illness] = n_dupes
        if verbose:
            note = f" ({n_dupes} duplicate {join_key} rows dropped)" if n_dupes else ""
            print(f"  {illness}: {n_rows} lines{note}")

    reference = illnesses[0]
    ref_df = per_file_frames[reference].rename({"rsID": "ID", "Z": reference})

    renamed_frames = [ref_df]
    for illness in illnesses[1:]:
        df = per_file_frames[illness].rename(
            {"rsID": "ID", "A0": f"A0_{illness}", "A1": f"A1_{illness}", "Z": illness}
        )
        df = df.drop("ID") if join_key == "chrom" else df.drop(["chrom", "pos"])
        renamed_frames.append(df)

    merged = functools.reduce(
        lambda left, right: left.join(right, on=key_cols, how=join_how, coalesce=True),
        renamed_frames,
    )

    swap_counts = {}
    mismatch_counts = {}
    drop_mask = pl.Series([False] * merged.height)

    for illness in illnesses[1:]:
        a0_col, a1_col, z_col = f"A0_{illness}", f"A1_{illness}", illness

        is_match = (pl.col(a0_col) == pl.col("A0")) & (pl.col(a1_col) == pl.col("A1"))
        is_swap = (pl.col(a0_col) == pl.col("A1")) & (pl.col(a1_col) == pl.col("A0"))
        is_present = pl.col(z_col).is_not_null()

        flags = merged.select(
            swap=(is_present & is_swap),
            mismatch=(is_present & ~is_match & ~is_swap),
        )

        swap_counts[illness] = int(flags["swap"].sum())
        mismatch_counts[illness] = int(flags["mismatch"].sum())

        merged = merged.with_columns(
            pl.when(flags["swap"]).then(-pl.col(z_col)).otherwise(pl.col(z_col)).alias(z_col)
        ).drop([a0_col, a1_col])

        drop_mask = drop_mask | flags["mismatch"]

    n_mismatched_rows = int(drop_mask.sum())
    merged = merged.filter(~drop_mask)

    final_cols = ["ID", "chrom", "pos", "A0", "A1"] + illnesses
    merged = merged.select(final_cols)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_csv(str(output_path), separator="\t")
    if verbose:
        print(f"Saved joined GWAS phenotype matrix to {output_path}")

    stats = {
        "n_files": len(files),
        "illnesses": illnesses,
        "reference_illness": reference,
        "join_key": join_key,
        "line_counts": line_counts,
        "duplicate_counts": dupe_counts,
        "swap_counts": swap_counts,
        "mismatch_counts": mismatch_counts,
        "n_rows_dropped_mismatch": n_mismatched_rows,
        "n_rows_joined": merged.height,
    }
    return stats


def merge_gwas_illness_mri(df_illness, df_mri):
    # Step 1: direct match on [ID, A0, A1]
    direct = df_illness.merge(df_mri, on=["ID", "A0", "A1"], how="inner")
    n_rows_direct = direct.shape[0]

    # Step 2: flip alleles for unmatched rows, invert Z-score (exclude palindromic SNPs)
    direct_ids = set(direct["ID"])
    illness_remaining = df_illness[~df_illness["ID"].isin(direct_ids)].copy()

    palindromic = illness_remaining.apply(
        lambda r: frozenset([r["A0"], r["A1"]]) in (frozenset(["A", "T"]), frozenset(["C", "G"])),
        axis=1,
    )
    illness_remaining = illness_remaining[~palindromic].copy()
    illness_remaining[["A0", "A1"]] = illness_remaining[["A1", "A0"]].values
    illness_remaining["Z"] = -illness_remaining["Z"]

    flipped = illness_remaining.merge(df_mri, on=["ID", "A0", "A1"], how="inner")
    n_rows_flipped = flipped.shape[0]

    # Step 3: concatenate
    aligned = pd.concat([direct, flipped], ignore_index=True)
    return aligned, n_rows_direct, n_rows_flipped, int(palindromic.sum())



def aligne_illness_mri(illness, verbose=True, chunk_size=10000, total_chunks=None, mri_path=None, polars=False, output_suffix=""):
    
    if verbose:
        print(f"Loading GWAS data for MRI")
    if mri_path is None:
        mri_path = Path(f"../data/pipeline/input/gwas_mri/all_z_scores.txt")
    if polars:
        df_mri = load_txt_polars(Path(mri_path), chunk_size=chunk_size, total_chunks=total_chunks)
    else:
        df_mri = load_txt(Path(mri_path), chunk_size=chunk_size, total_chunks=total_chunks)
    # drop columns only needed for standalone use (e.g. gwas_pheno's all_z_scores.txt),
    # not for the ID/A0/A1 merge below — avoids _x/_y suffix collisions with df_illness
    df_mri = df_mri.drop(columns=[c for c in ["chrom", "pos"] if c in df_mri.columns])
    # get number of rows
    n_rows_mri = df_mri.shape[0]
    # drop rows with missing values
    df_mri.dropna(inplace=True)
    n_rows_mri_after_drop = df_mri.shape[0]
    if verbose:
        print(f"Number of rows in MRI data: {n_rows_mri}")
        print(f"Number of rows in MRI rows dropped: {n_rows_mri - n_rows_mri_after_drop}")
    if verbose:
        print(f"Loading GWAS data for illness {illness}")
    if polars:
        df_illness = load_txt_polars(Path(f"./data/pipeline/input/gwas_illness/z_{illness}.txt"), chunk_size=chunk_size)
    else:
        df_illness = load_txt(Path(f"./data/pipeline/input/gwas_illness/z_{illness}.txt"), chunk_size=chunk_size)
    df_illness.rename(columns={"rsID": "ID"}, inplace=True)
    n_rows_illness = df_illness.shape[0]
    # drop rows with missing values    df_illness.dropna(inplace=True)
    n_rows_illness_after_drop = df_illness.shape[0]
    if verbose:
        print(f"Number of rows in illness data: {n_rows_illness}")
        print(f"Number of rows in illness data rows dropped: {n_rows_illness - n_rows_illness_after_drop}")
    if verbose:
        print(f"Aligning data for illness {illness} with MRI data")

    aligned, n_rows_direct, n_rows_flipped, n_palindromic = merge_gwas_illness_mri(df_illness, df_mri)

    ## Step 1: direct match on [ID, A0, A1]
    #direct = df_illness.merge(df_mri, on=["ID", "A0", "A1"], how="inner")
    #n_rows_direct = direct.shape[0]
#
    ## Step 2: flip alleles for unmatched rows, invert Z-score (exclude palindromic SNPs)
    #direct_ids = set(direct["ID"])
    #illness_remaining = df_illness[~df_illness["ID"].isin(direct_ids)].copy()
#
    #palindromic = illness_remaining.apply(
    #    lambda r: frozenset([r["A0"], r["A1"]]) in (frozenset(["A", "T"]), frozenset(["C", "G"])),
    #    axis=1,
    #)
    #n_palindromic = palindromic.sum()
    #illness_remaining = illness_remaining[~palindromic].copy()
    #illness_remaining[["A0", "A1"]] = illness_remaining[["A1", "A0"]].values
    #illness_remaining["Z"] = -illness_remaining["Z"]
#
    #flipped = illness_remaining.merge(df_mri, on=["ID", "A0", "A1"], how="inner")
    #n_rows_flipped = flipped.shape[0]
#
    ## Step 3: concatenate
    #aligned = pd.concat([direct, flipped], ignore_index=True)

    #if verbose:
    #    print(f"  Direct matches:           {n_rows_direct}")
    #    print(f"  Flipped allele matches:   {n_rows_flipped}")
    #    print(f"  Palindromic SNPs skipped: {n_palindromic}")
    #    print(f"  Total aligned:            {aligned.shape[0]}")

    # remove all columns except ID, and P
    # get all columns except ID and P
    cols_to_keep = [col for col in aligned.columns if col not in ["P"]]
    mri = aligned[cols_to_keep]
    aligned = aligned[["ID", "P"]]
    n_rows_aligned = aligned.shape[0]
    if verbose:
        print(f"Number of rows in aligned data: {n_rows_aligned}")
    # save it as txt file
    output_path_illness= Path(f"./data/pipeline/intermediate/aligned_{illness}{output_suffix}.txt").expanduser().resolve()
    aligned.to_csv(output_path_illness, sep="\t", index=False)
    if verbose:
        print(f"Saved aligned data for illness {illness} at {output_path_illness}")
    #output_path_mri = Path(f"../data/pipeline/intermediate/mri_{illness}.txt").expanduser().resolve()
    #mri.to_csv(output_path_mri, sep="\t", index=False)
    output = {"n_rows_mri": n_rows_mri, 
              "n_rows_mri_dropped": n_rows_mri - n_rows_mri_after_drop, 
              "n_rows_illness": n_rows_illness, 
              "n_rows_illness_dropped": n_rows_illness - n_rows_illness_after_drop, 
              "n_rows_direct_match": n_rows_direct,
              "n_rows_flipped_match": n_rows_flipped,
              "n_palindromic_skipped": n_palindromic,
              "n_rows_aligned": n_rows_aligned}
    return output



def call_plink2(cfg: dict[str, str]) -> None:
    cmd: list[str] = ["plink2"]
    for key, value in cfg.items():
        cmd.append(key)
        if value is not None and str(value) != "":
            cmd.append(str(value))

    # print each element of the command on a new line for debugging
    print(f"Calling plink2 with command: {shlex.join(cmd)}")

    #subprocess.run("ln -sf $HOME/tools/plink2/plink2 $HOME/tools/bin/plink2", shell=True)
    # run this command with subprocess echo 'export PATH="$HOME/tools/bin:$PATH"' >> ~/.bashrc
    #subprocess.run("echo 'export PATH=\"$HOME/tools/bin:$PATH\"' >> ~/.bashrc", shell=True)
    #subprocess.run("source ~/.bashrc", shell=True)

    plink2_path = shutil.which("plink2")
    if not plink2_path:
        raise FileNotFoundError(
            "plink2 is not available. Install plink2 and ensure it's on your PATH."
        )

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error calling plink2: {result.stderr}")
        sys.exit(1)

    print(f"plink2 output: {result.stdout}")


def aligne_clumped_illness_mri(illness, verbose=True, chunk_size=10000, total_chunks=None, polars=False, mri_path=None, output_suffix=""):
    if verbose:
        print(f"Loading clumped data for illness {illness}")
    if polars:
        df_clumped = load_txt_polars(Path(f"./data/pipeline/output/clumped_{illness}{output_suffix}.clumps"), chunk_size=chunk_size, total_chunks=total_chunks)
    else:
        df_clumped = load_txt(Path(f"./data/pipeline/output/clumped_{illness}{output_suffix}.clumps"), chunk_size=chunk_size)
    df_clumped.rename(columns={"ID": "ID"}, inplace=True)
    n_rows_clumped = df_clumped.shape[0]
     # drop rows with missing values
    #df_clumped.dropna(inplace=True) 
    n_rows_clumped_after_drop = df_clumped.shape[0]
    if verbose:
        print(f"Number of rows in clumped data: {n_rows_clumped}")
        print(f"Number of rows in clumped data after dropping missing values: {n_rows_clumped_after_drop}")
    if verbose:
        print(f"Loading aligned MRI data for illness {illness}")
    
    if mri_path is None:
        mri_path = Path(f"./data/pipeline/input/gwas_mri/all_z_scores.txt")
    if polars:
        df_mri = load_txt_polars(Path(mri_path), chunk_size=chunk_size, total_chunks=total_chunks)
    else:
        df_mri = load_txt(Path(mri_path), chunk_size=chunk_size, total_chunks=total_chunks)
    # drop columns only needed for standalone use (e.g. gwas_pheno's all_z_scores.txt),
    # not for the ID/A0/A1 merge below — avoids _x/_y suffix collisions with df_illness
    df_mri = df_mri.drop(columns=[c for c in ["chrom", "pos"] if c in df_mri.columns])
    n_rows_mri = df_mri.shape[0]
    
    if verbose:
        print(f"Aligning clumped data for illness {illness} with MRI data")
    #aligned_clumped = df_clumped.merge(df_mri, on="ID", how="inner")

    if verbose:
        print(f"Loading GWAS data for illness {illness}")
    if polars:
        df_illness = load_txt_polars(Path(f"./data/pipeline/input/gwas_illness/z_{illness}.txt"), chunk_size=chunk_size)
    else:
        df_illness = load_txt(Path(f"./data/pipeline/input/gwas_illness/z_{illness}.txt"), chunk_size=chunk_size)
    df_illness.rename(columns={"rsID": "ID"}, inplace=True)
    n_rows_illness = df_illness.shape[0]
    # drop rows with missing values    df_illness.dropna(inplace=True)
    n_rows_illness_after_drop = df_illness.shape[0]
    if verbose:
        print(f"Number of rows in illness data: {n_rows_illness}")
        print(f"Number of rows in illness data rows dropped: {n_rows_illness - n_rows_illness_after_drop}")
    if verbose:
        print(f"Aligning data for illness {illness} with MRI data")
    #aligned_illness = df_illness.merge(df_mri, on="ID, ", how="inner")
    aligned_illness, _, _, _ = merge_gwas_illness_mri(df_illness, df_mri)

    cols_to_keep = [col for col in aligned_illness.columns if col not in ["P"]]
    aligned_illness = aligned_illness[cols_to_keep]
    aligned = df_clumped.merge(aligned_illness, on="ID", how="inner")
    
    if verbose:
        print(f"Number of rows in aligned clumped data: {aligned.shape[0]}")

    
    # remove columns chrom	pos	A0	A1	N
    #CHROM	POS	P	TOTAL	NONSIG	S0.05	S0.01	S0.001	S0.0001	SP2	chrom	pos	A0	A1	N
    aligned = aligned.drop(columns=["#CHROM","POS","TOTAL","NONSIG","S0.05","S0.01","S0.001","S0.0001","SP2","chrom","pos","A0","A1","N"])
    output_path = Path(f"./data/pipeline/final/aligned_clumped_{illness}{output_suffix}.txt").expanduser().resolve()
    aligned.to_csv(output_path, sep="\t", index=False)
    if verbose:
        print(f"Saved aligned clumped data for illness {illness} at {output_path}")
    output = {"n_rows_clumped": n_rows_clumped, 
              "n_rows_clumped_dropped": n_rows_clumped - n_rows_clumped_after_drop, 
              "n_rows_aligned_illness": aligned_illness.shape[0],
              "n_rows_mri": n_rows_mri,
              "n_rows_aligned": aligned.shape[0]}
    return output