from pathlib import Path
import shlex
import shutil
import subprocess
import sys


from dataloader.preprocess import load_txt, load_txt_polars

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
    aligned = df_illness.merge(df_mri, on="ID", how="inner")
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