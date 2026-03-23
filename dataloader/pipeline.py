from dataloader.preprocess import load_txt, preprocess
from pathlib import Path

def aligne_illness_mri(illness, verbose=True, chunk_size=10000):
    
    if verbose:
        print(f"Loading GWAS data for MRI")
    df_mri = load_txt(Path("../../data/pipeline/input/gwas_mri/donnees_MRI_et_diseases.txt"), chunk_size=chunk_size)
    if verbose:
        print(f"Loading GWAS data for illness {illness}")
    df_illness = load_txt(Path(f"../../data/pipeline/input/gwas_illness/z_PGC_{illness}.txt"), chunk_size=chunk_size)
    df_illness.rename(columns={"rsID": "ID"}, inplace=True)
    if verbose:
        print(f"Aligning data for illness {illness} with MRI data")
    aligned = df_illness.merge(df_mri, on="ID", how="inner")
    # save it as txt file
    output_path = Path(f"../../data/pipeline/intermediate/aligned_{illness}.txt").expanduser().resolve()
    aligned.to_csv(output_path, sep="\t", index=False)
    if verbose:
        print(f"Saved aligned data for illness {illness} at {output_path}")