#!/usr/bin/env python3
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

import os
import sys
import argparse
from argparse import RawTextHelpFormatter


print("""

 ▄▄▄▄▄               ▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄ 
 ██▀▀▀██             ██▀▀▀▀▀▀  ▀▀▀██▀▀▀ 
 ██    ██  ██    ██  ██           ██    
 ██    ██  ██    ██  ███████      ██    
 ██    ██  ██    ██  ██           ██    
 ██▄▄▄██   ██▄▄▄███  ██▄▄▄▄▄▄     ██    
 ▀▀▀▀▀      ▀▀▀▀ ▀▀  ▀▀▀▀▀▀▀▀     ▀▀    
 
\x1B[3mA Unified Deep Learning Framework for 
Predicting mRNA Translation Efficiency 
       Across Human Cell Types\x1B[0m             

""", flush=True)


parser = argparse.ArgumentParser(
    formatter_class=RawTextHelpFormatter,
    description="Predict cell-type translation efficiency (TE) from 5' UTR + CDS sequences.\n"
                "DuET predicts TE for all human cell types at once (multi-target).")

sequence_args = parser.add_argument_group("Predict a single sequence from the command line",
                                          "Both 5' UTR and CDS must be specified.")
sequence_args.add_argument("-u", "--utr5", type=str, help="Input 5' UTR sequence")
sequence_args.add_argument("-c", "--cds", type=str, help="Input CDS sequence")

batch_args = parser.add_argument_group("Predict a batch of sequences from a .tsv file",
                                       "(The file must have utr5 and cds columns; txID optional.)")
batch_args.add_argument("-i", "--input", type=str,
                        help="Path to input .tsv file containing sequences")
batch_args.add_argument("-o", "--output", type=str,
                        help="Path to output .tsv file (if not specified, prints to stdout)")

model_args = parser.add_argument_group("Prediction options")
model_args.add_argument("-t", "--target", type=str, default=None,
                        help="Comma-separated cell type name(s) to report (e.g. 'hek293t,a549').\n"
                             "TE_ prefix optional. If omitted, all cell types are returned.")
model_args.add_argument("--device", type=str, default="cuda",
                        help="Inference device (default=%(default)s), e.g. cuda, cuda:0, cpu.")
model_args.add_argument("--batch_size", type=int, default=256,
                        help="Inference batch size (default=%(default)s).")

args = parser.parse_args()

if args.input is None and (args.utr5 is None or args.cds is None):
    print("\x1b[1mEither --input or both --utr5 and --cds must be specified\x1b[0m\n")
    parser.print_help()
    exit(1)

if args.input is not None:
    batch_mode = True
    if args.utr5 is not None or args.cds is not None:
        print("\x1b[1m--input specified, ignoring --utr5 and --cds arguments\x1b[0m")
else:
    batch_mode = False

if args.output is not None and os.path.exists(args.output):
    print(f"\x1b[1mOutput file {args.output} already exists, overwriting\x1b[0m")


print("Loading libraries...", end="\t", flush=True)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "src"))
import pandas as pd
import duet
print("Done.")


# ---- assemble query --------------------------------------------------------
if batch_mode:
    df = pd.read_csv(args.input, sep="\t")
    if not {"utr5", "cds"}.issubset(df.columns):
        print(f"\x1b[1mInput file must contain 'utr5' and 'cds' columns (found: {list(df.columns)})\x1b[0m")
        exit(1)
    query = df[["utr5", "cds"]].to_dict(orient="records")
    tx_ids = df["txID"].tolist() if "txID" in df.columns else list(range(len(df)))
else:
    query = [{"utr5": args.utr5, "cds": args.cds}]
    tx_ids = [0]

target = [t.strip() for t in args.target.split(",")] if args.target else None
if target is not None and len(target) == 1:
    target = target[0]


# ---- predict ---------------------------------------------------------------
print("\nPredicting...")
result = duet.predict(query, target=target, device=args.device,
                      batch_size=args.batch_size, use_tqdm=True)

# duet.predict returns:
#   target=None or list[str]  -> DataFrame (cols = cell types)
#   target=str                -> list[float] (single cell type)
if isinstance(result, list):
    out = pd.DataFrame({"txID": tx_ids, f"pred_TE_{target}": result})
else:
    out = result.copy()
    out.insert(0, "txID", tx_ids)


# ---- report ----------------------------------------------------------------
if batch_mode:
    if args.output is not None:
        out.to_csv(args.output, sep="\t", index=False)
        print(f"\nWrote {len(out)} prediction(s) -> {args.output}")
    else:
        print("\n")
        out.to_csv(sys.stdout, sep="\t", index=False)
else:
    print()
    row = out.drop(columns="txID").iloc[0]
    if len(row) == 1:
        print(f"Predicted TE ({row.index[0]}): \x1b[1m{row.iloc[0]:.4f}\x1b[0m")
    else:
        print("Predicted TE:")
        for name, val in row.items():
            print(f"  {name}\t\x1b[1m{val:.4f}\x1b[0m")
