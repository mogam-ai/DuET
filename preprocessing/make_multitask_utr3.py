#!/usr/bin/env python3
# Program: DuET v1.2.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

# Attach a utr3 column to the multi-task TE table.
#
# multitask_TE.tsv holds [txID, utr5, cds, TE_*] and carries no 3' UTR, which the
# three-region model and the region ablation need. The 3' UTR is a transcript-level
# sequence, so it is taken from the per-cell-type tables (celltype_te/*_TE.tsv) and
# joined on txID. Rows whose txID has no 3' UTR are dropped and reported.

import glob
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default="datasets/multitask_TE.tsv",
        help="Multi-task TE table to extend. Default=%(default)s.")
parser.add_argument("-c", "--celltype-dir", type=str, default="datasets/celltype_te",
        help="Directory of per-cell-type *_TE.tsv tables. Default=%(default)s.")
parser.add_argument("-o", "--output", type=str,
        default="datasets/multitask_utr3_TE.tsv",
        help="Output file name. Default=%(default)s.")

args = parser.parse_args()

utr3_map = {}
conflicts = 0
for fname in sorted(glob.glob(f"{args.celltype_dir}/*_TE.tsv")):
    table = pd.read_csv(fname, sep="\t", usecols=["txID", "utr3"]).dropna(subset=["utr3"])
    for tx, utr3 in zip(table["txID"], table["utr3"]):
        if tx not in utr3_map:
            utr3_map[tx] = utr3
        elif utr3_map[tx] != utr3:
            conflicts += 1
if not utr3_map:
    raise SystemExit(f"No *_TE.tsv with a utr3 column under {args.celltype_dir}")
if conflicts:
    print(f"[warn] {conflicts} txID with inconsistent utr3 across cell types (kept first)")
print(f"txID -> utr3 map size: {len(utr3_map)}")

data = pd.read_csv(args.input, sep="\t")
print(f"input rows={len(data)} cols={len(data.columns)}")

data["utr3"] = data["txID"].map(utr3_map)
print(f"rows missing utr3: {data['utr3'].isna().sum()}/{len(data)}")
data = data.dropna(subset=["utr3"]).reset_index(drop=True)

te_cols = [c for c in data.columns if c.startswith("TE_")]
data = data[["txID", "utr5", "cds", "utr3"] + te_cols]
data.to_csv(args.output, sep="\t", index=False)
print(f"wrote {args.output} rows={len(data)} cols={len(data.columns)}")
