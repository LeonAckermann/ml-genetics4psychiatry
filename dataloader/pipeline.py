from pathlib import Path
import gc
import shlex
import shutil
import subprocess
import sys

import pandas as pd
import polars as pl


from dataloader.preprocess import load_txt, load_txt_polars

def construct_gwas_mri(path, output_path, chunk_size=10000, total_chunks=None, polars=False):
    path = Path(path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    # Collect all allRES.txt files recursively
    allres_files = sorted(path.rglob("allRES.txt"))
    if not allres_files:
        raise FileNotFoundError(f"No allRES.txt files found under {path}")

    print(f"Found {len(allres_files)} allRES.txt files")

    merged: pl.DataFrame | None = None
    for file in allres_files:
        phenotype = file.parent.name

        # Use lazy scan so only the 4 needed columns are read from disk
        df: pl.DataFrame = (
            pl.scan_csv(str(file), separator="\t", null_values=[".", "NA", "N/A", "NaN", "nan", "NULL", "null", ""])
            .select(["ID", "A1", "PROVISIONAL_REF?", "T_STAT"])
            .rename({"T_STAT": phenotype})
            .collect()
        )

        if merged is None:
            merged = df
        else:
            old_merged = merged
            merged = merged.join(df, on=["ID", "A1", "PROVISIONAL_REF?"], how="inner")
            del old_merged

        del df
        gc.collect()
        #print(f"  Merged {phenotype}: {merged.shape[0]} rows remaining")

    # rename A1 -> A0 and PROVISIONAL_REF? -> A1
    merged = merged.rename({"A1": "A0", "PROVISIONAL_REF?": "A1"})

    n_rows_merged = merged.shape[0]
    n_unique_ids = merged.select(pl.col("ID").n_unique()).item()

    stats = {
        "n_files": len(allres_files),
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



def aligne_illness_mri(illness, verbose=True, chunk_size=10000, total_chunks=None, mri_path=None, polars=False):
    
    if verbose:
        print(f"Loading GWAS data for MRI")
    if mri_path is None:
        mri_path = Path(f"../data/pipeline/input/gwas_mri/all_z_scores.txt")
    if polars:
        df_mri = load_txt_polars(Path(mri_path), chunk_size=chunk_size, total_chunks=total_chunks)
    else:
        df_mri = load_txt(Path(mri_path), chunk_size=chunk_size, total_chunks=total_chunks)
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
        df_illness = load_txt_polars(Path(f"../data/pipeline/input/gwas_illness/z_{illness}.txt"), chunk_size=chunk_size)
    else:
        df_illness = load_txt(Path(f"../data/pipeline/input/gwas_illness/z_{illness}.txt"), chunk_size=chunk_size)
    df_illness.rename(columns={"rsID": "ID"}, inplace=True)
    n_rows_illness = df_illness.shape[0]
    # drop rows with missing values    df_illness.dropna(inplace=True)
    n_rows_illness_after_drop = df_illness.shape[0]
    if verbose:
        print(f"Number of rows in illness data: {n_rows_illness}")
        print(f"Number of rows in illness data rows dropped: {n_rows_illness - n_rows_illness_after_drop}")
    if verbose:
        print(f"Aligning data for illness {illness} with MRI data")

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
    n_palindromic = palindromic.sum()
    illness_remaining = illness_remaining[~palindromic].copy()
    illness_remaining[["A0", "A1"]] = illness_remaining[["A1", "A0"]].values
    illness_remaining["Z"] = -illness_remaining["Z"]

    flipped = illness_remaining.merge(df_mri, on=["ID", "A0", "A1"], how="inner")
    n_rows_flipped = flipped.shape[0]

    # Step 3: concatenate
    aligned = pd.concat([direct, flipped], ignore_index=True)

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
    output_path_illness= Path(f"../data/pipeline/intermediate/aligned_{illness}.txt").expanduser().resolve()
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


def aligne_clumped_illness_mri(illness, verbose=True, chunk_size=10000, total_chunks=None, polars=False, mri_path=None):
    if verbose:
        print(f"Loading clumped data for illness {illness}")
    if polars:
        df_clumped = load_txt_polars(Path(f"../data/pipeline/output/clumped_{illness}.clumps"), chunk_size=chunk_size, total_chunks=total_chunks)
    else:
        df_clumped = load_txt(Path(f"../data/pipeline/output/clumped_{illness}.clumps"), chunk_size=chunk_size)
    df_clumped.rename(columns={"ID": "ID"}, inplace=True)
    n_rows_clumped = df_clumped.shape[0]
     # drop rows with missing values
    df_clumped.dropna(inplace=True)
    n_rows_clumped_after_drop = df_clumped.shape[0]
    if verbose:
        print(f"Number of rows in clumped data: {n_rows_clumped}")
        print(f"Number of rows in clumped data after dropping missing values: {n_rows_clumped_after_drop}")
    if verbose:
        print(f"Loading aligned MRI data for illness {illness}")
    
    if mri_path is None:
        mri_path = Path(f"../data/pipeline/input/gwas_mri/all_z_scores.txt")
    if polars:
        df_mri = load_txt_polars(Path(mri_path), chunk_size=chunk_size, total_chunks=total_chunks)
    else:
        df_mri = load_txt(Path(mri_path), chunk_size=chunk_size, total_chunks=total_chunks)
    n_rows_mri = df_mri.shape[0]
    
    if verbose:
        print(f"Aligning clumped data for illness {illness} with MRI data")
    #aligned_clumped = df_clumped.merge(df_mri, on="ID", how="inner")

    if verbose:
        print(f"Loading GWAS data for illness {illness}")
    if polars:
        df_illness = load_txt_polars(Path(f"../data/pipeline/input/gwas_illness/z_{illness}.txt"), chunk_size=chunk_size)
    else:
        df_illness = load_txt(Path(f"../data/pipeline/input/gwas_illness/z_{illness}.txt"), chunk_size=chunk_size)
    df_illness.rename(columns={"rsID": "ID"}, inplace=True)
    n_rows_illness = df_illness.shape[0]
    # drop rows with missing values    df_illness.dropna(inplace=True)
    n_rows_illness_after_drop = df_illness.shape[0]
    if verbose:
        print(f"Number of rows in illness data: {n_rows_illness}")
        print(f"Number of rows in illness data rows dropped: {n_rows_illness - n_rows_illness_after_drop}")
    if verbose:
        print(f"Aligning data for illness {illness} with MRI data")
    aligned_illness = df_illness.merge(df_mri, on="ID", how="inner")
    cols_to_keep = [col for col in aligned_illness.columns if col not in ["P"]]
    aligned_illness = aligned_illness[cols_to_keep]
    aligned = df_clumped.merge(aligned_illness, on="ID", how="inner")
    if verbose:
        print(f"Number of rows in aligned clumped data: {aligned.shape[0]}")

    
    # remove columns chrom	pos	A0	A1	N
    #CHROM	POS	P	TOTAL	NONSIG	S0.05	S0.01	S0.001	S0.0001	SP2	chrom	pos	A0	A1	N
    aligned = aligned.drop(columns=["#CHROM","POS","TOTAL","NONSIG","S0.05","S0.01","S0.001","S0.0001","SP2","chrom","pos","A0","A1","N"])
    output_path = Path(f"../data/pipeline/final/aligned_clumped_{illness}.txt").expanduser().resolve()
    aligned.to_csv(output_path, sep="\t", index=False)
    if verbose:
        print(f"Saved aligned clumped data for illness {illness} at {output_path}")
    output = {"n_rows_clumped": n_rows_clumped, 
              "n_rows_clumped_dropped": n_rows_clumped - n_rows_clumped_after_drop, 
              "n_rows_aligned_illness": aligned_illness.shape[0],
              "n_rows_mri": n_rows_mri,
              "n_rows_aligned": aligned.shape[0]}
    return output