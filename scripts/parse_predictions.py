#!/usr/bin/env python
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

import sys
import os
import glob
import pandas as pd

prefix = sys.argv[1]
pattern = os.path.join("_logs", f"{prefix}-all-celltype-seed42-split*")
dirs = sorted(glob.glob(pattern))

frames = []
for d in dirs:
    split = os.path.basename(d).rsplit("split", 1)[1]
    idx_df = pd.read_csv(os.path.join(d, "test_dataset_idx.tsv"), sep="\t", index_col=0)
    pred_df = pd.read_csv(os.path.join(d, "all-celltype_TE_prediction.tsv"), sep="\t")
    df = pd.DataFrame({
        "split": split,
        "index": idx_df.index,
        "txID": idx_df["txID"].values,
        "y": pred_df["y"].values,
        "pred_y": pred_df["pred_y"].values,
    })
    frames.append(df)

out = pd.concat(frames, ignore_index=True)
out.to_csv(f"{prefix}_all-celltype_predictions.tsv", sep="\t", index=False)
print(f"Wrote {len(out)} rows to {prefix}_all-celltype_predictions.tsv")
