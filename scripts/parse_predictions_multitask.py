#!/usr/bin/env python
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""Parse multitask DuET predictions into a single long-format matrix.

Mirrors parse_predictions.py (single-task) but for the 76 cell-type multitask
model. Single-task output columns were: split, index, txID, y, pred_y.
Multitask adds a `celltype` column (the TE_ prefix stripped) since each gene has
one prediction per cell type. Rows where y is NaN (cell type unmeasured for that
gene) are dropped.

Source per split dir:
  <split>/multitask_TE_prediction_matrix.tsv  (kind in {y,yhat}, gene_idx, TE_* cols)
gene_idx is the test-set order, i.e. the i-th element of that split's KFold test
partition. We reproduce KFold(10, shuffle, seed) on the dataset to map
gene_idx -> global dataset index -> txID.

Usage:
  python parse_predictions_multitask.py duet
Output:
  <prefix>_multitask_predictions.tsv  with columns
  split, index, txID, celltype, y, pred_y
"""
import sys
import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

prefix = sys.argv[1]
KFOLD_K = 10
KFOLD_SEED = 42  # DataModuleConfig default (config.yaml has no override)
DATA_TSV = "datasets/extended_celltype_te/multitask_TE.tsv"

# dataset row order MUST match DuetDataset (read_csv, no reordering).
data = pd.read_csv(DATA_TSV, sep="\t")
txid = data["txID"].values

kf = KFold(n_splits=KFOLD_K, shuffle=True, random_state=KFOLD_SEED)
fold_test_idx = [te for _, te in kf.split(data)]  # global indices per split

pattern = os.path.join("_logs", f"{prefix}-seed42-split*")
dirs = sorted(d for d in glob.glob(pattern) if "inner" not in os.path.basename(d))

frames = []
for d in dirs:
    split = int(os.path.basename(d).rsplit("split", 1)[1])
    mpath = os.path.join(d, "multitask_TE_prediction_matrix.tsv")
    if not os.path.exists(mpath):
        print(f"  skip {split}: no prediction matrix")
        continue
    m = pd.read_csv(mpath, sep="\t")
    te_cols = [c for c in m.columns if c.startswith("TE_")]
    y_mat = m[m["kind"] == "y"].sort_values("gene_idx")
    yhat_mat = m[m["kind"] == "yhat"].sort_values("gene_idx")
    assert (y_mat["gene_idx"].values == yhat_mat["gene_idx"].values).all()
    gidx = y_mat["gene_idx"].values                 # 0..(n_test-1), test-set order
    global_idx = fold_test_idx[split][gidx]          # -> dataset row index
    assert len(global_idx) == len(gidx), (len(global_idx), len(gidx))

    y = y_mat[te_cols].to_numpy()                    # (n_test, 76)
    yhat = yhat_mat[te_cols].to_numpy()
    celltypes = np.array([c[3:] for c in te_cols])   # strip 'TE_'

    # long format; drop NaN y
    n_test, n_ct = y.shape
    rows_split = np.repeat(np.arange(n_test), n_ct)
    df = pd.DataFrame({
        "split": split,
        "index": np.repeat(global_idx, n_ct),
        "txID": np.repeat(txid[global_idx], n_ct),
        "celltype": np.tile(celltypes, n_test),
        "y": y.reshape(-1),
        "pred_y": yhat.reshape(-1),
    })
    df = df[~df["y"].isna()].reset_index(drop=True)
    frames.append(df)

out = pd.concat(frames, ignore_index=True)
out_path = f"{prefix}_multitask_predictions.tsv"
out.to_csv(out_path, sep="\t", index=False)
print(f"Wrote {len(out)} rows ({out['txID'].nunique()} txIDs x cell types, NaN-dropped) "
      f"to {out_path}")
