#!/usr/bin/env python3
"""Merge all celltype TE files into wide-format multi-target TSV for RiboNN.

Usage:
    python baselines/RiboNN/make_multitarget.py --config config.yaml
"""

import argparse
import pandas as pd
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "benchlib"))
from benchlib import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = Path(cfg["datasets"]["celltype_te"]["base_path"])

    # Find all celltype files (exclude all-celltype)
    files = sorted(base.glob("*_TE.tsv"))
    # Exclude aggregate files (all-celltype, multitask, multitarget) — they are outputs, not celltypes
    files = [f for f in files
             if not any(k in f.name for k in ("all-celltype", "multitask", "multitarget"))]

    seq_cols = ["utr5", "cds", "utr3", "full_seq"]
    merged = None

    for f in files:
        ct_name = f.stem.replace("_TE", "")  # e.g. "hek293t"
        df = pd.read_csv(f, sep="\t")
        te_col = f"TE_{ct_name}"
        if merged is None:
            merged = df[["txID"] + seq_cols + ["logratio_te"]].rename(columns={"logratio_te": te_col})
        else:
            merged = merged.merge(df[["txID", "logratio_te"]].rename(columns={"logratio_te": te_col}),
                                  on="txID", how="outer")

    # Fill sequence info for outer-joined rows
    for f in files:
        fill = pd.read_csv(f, sep="\t", usecols=["txID"] + seq_cols).set_index("txID")
        for col in seq_cols:
            mask = merged[col].isna()
            if mask.any():
                merged.loc[mask, col] = merged.loc[mask, "txID"].map(fill[col])

    # Drop rows missing ALL targets
    te_cols = [c for c in merged.columns if c.startswith("TE_")]
    merged = merged.dropna(subset=te_cols, how="all").reset_index(drop=True)
    merged = merged[["txID"] + seq_cols + te_cols]

    out_dir = Path(cfg["output_dir"]) / "celltype_te_multi"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / "multitarget_all_TE.tsv"
    merged.to_csv(out_path, sep="\t", index=False)
    print(f"Saved: {out_path}")
    print(f"Shape: {merged.shape} ({len(te_cols)} targets)")
    print(f"NaN summary: min={merged[te_cols].isna().sum().min()}, max={merged[te_cols].isna().sum().max()}")


if __name__ == "__main__":
    main()
