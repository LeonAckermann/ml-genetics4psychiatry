"""
CLI wrapper around dataloader.pipeline.construct_gwas_phenotype.

Joins per-illness, per-chromosome GWAS summary statistic files
(e.g. z_BCAC_BREAST-CANCER_chr1.txt) on (chromosome, position),
concatenating chromosome files within each illness before harmonizing
A0/A1 allele coding and rsID across illnesses (per-row consensus, falling
back through illnesses in order) and carrying the Z score through the join.

For pipeline-based runs (as part of an experiment YAML config) use the
"gwas_phenotype_construction" section and run main.py instead of this
script directly — see experiments/pipeline/construct_gwas_phenotype.yaml.

Pass --join-key rs to join on rsID instead of (chrom, pos). Comparing the
row counts between the two modes is a useful diagnostic: a large gap (e.g.
millions via rs vs. a handful via chrom) points to a genome build mismatch
between input files, since rsIDs are build independent while chrom/pos
are not.

Usage:
    python join_gwas_zscores.py \
        --input-dir data/pipeline/input/gwas_illness \
        --output data/pipeline/output/joined_gwas_zscores.tsv \
        --how inner --join-key chrom
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dataloader.pipeline import construct_gwas_phenotype  # noqa: E402

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "pipeline" / "input" / "gwas_illness"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "pipeline" / "input" / "gwas_pheno" / "all_z_scores.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                         help="Directory containing z_<ILLNESS>_chr<CHROM>.txt GWAS files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                         help="Path to write the joined TSV")
    parser.add_argument("--info-csv", type=Path, default=None,
                         help="info.csv with include/Consortium/Outcome columns. When set, only "
                              "phenotypes with include=1 are read, each combining its "
                              "z_<Consortium>_<Outcome>_chr<N>.txt files")
    parser.add_argument("--how", choices=["inner", "outer"], default="inner",
                         help="Join type across illnesses (default: inner, i.e. SNP must be present in every file)")
    parser.add_argument("--join-key", choices=["chrom", "rs"], default="chrom",
                         help="Join on (chrom, pos) [default] or on rsID")
    args = parser.parse_args()

    stats = construct_gwas_phenotype(
        input_path=args.input_dir,
        output_path=args.output,
        how=args.how,
        join_key=args.join_key,
        info_csv_path=args.info_csv,
    )

    report_lines = ["GWAS Z-score join report", "=" * 40]
    report_lines.append(f"Input directory: {args.input_dir}")
    report_lines.append(f"Join type: {args.how}")
    report_lines.append(f"Join key: {args.join_key}")
    report_lines.append(f"Reference illness (allele orientation): {stats['reference_illness']}")
    report_lines.append("")
    report_lines.append("Lines per input file:")
    for illness in stats["illnesses"]:
        n_dupes = stats["duplicate_counts"][illness]
        note = f" ({n_dupes} duplicate {args.join_key} rows dropped)" if n_dupes else ""
        report_lines.append(f"  {illness}: {stats['line_counts'][illness]}{note}")
    report_lines.append("")
    report_lines.append("Allele swaps detected relative to reference (Z sign flipped):")
    for illness, count in stats["swap_counts"].items():
        report_lines.append(f"  {illness}: {count}")
    report_lines.append("")
    report_lines.append("Unresolvable allele mismatches (that illness's Z nulled, row kept):")
    for illness, count in stats["mismatch_counts"].items():
        report_lines.append(f"  {illness}: {count}")
    report_lines.append(f"  total values nulled for mismatch: {stats['n_values_nulled_mismatch']}")
    report_lines.append("")
    report_lines.append(f"Number of joined lines in the end: {stats['n_rows_joined']}")

    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = args.output.with_suffix(".report.txt")
    report_path.write_text(report_text + "\n")


if __name__ == "__main__":
    main()
