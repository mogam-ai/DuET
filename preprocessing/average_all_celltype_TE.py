#!/usr/bin/env python3
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD


import sys
from os.path import dirname, basename
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

def avg(vector):
    return sum(vector) / len(vector)

parser = argparse.ArgumentParser()
parser.add_argument("queue", metavar="FPATH", nargs="+",
        help="Input TE.tsv datasets.")
parser.add_argument("-o", "--output", type=str, default="all-celltype_TE.tsv",
        help="Output file name. Default=%(default)s.)")
parser.add_argument("-c", "--cutoff", type=float, default=0.3,
        help="Prevalence cutoff of transcripts. Default=%(default)s.")
parser.add_argument("-t", "--threshold", type=int, default=-1,
        help="Prevalence threshold of transcripts. This will override --cutoff.")

args = parser.parse_args()
queue = [k for k in args.queue if basename(k) != args.output]

output_fname = args.output
output_dir = dirname(first_fname := queue[0])
header = open(first_fname).readline()

prevalence_cutoff = args.cutoff
if args.threshold != -1:
    threshold = args.threshold
    assert threshold >= 0, f"Wrong threshold value: {threshold}"
else:
    threshold = round(len(queue) * prevalence_cutoff)

print(f"Prevalence cutoff {prevalence_cutoff}; threshold {threshold}\n")

tx_to_logratio_te = {}
tx_to_residual_te = {}
tx_to_rpf = {}
tx_to_rna = {}
tx_to_detail = {}
for fname in (pbar := tqdm(queue)):
    #pbar.set_postfix_str(current_file := basename(fname))

    with open(fname) as f:
        next(f)
        for line in f:
            txid, utr5, cds, utr3, full_seq, logratio_te, residual_te, rna, rpf = line.strip().split("\t")
            tx_to_detail.setdefault(txid, (utr5, cds, utr3, full_seq))
            try:
                tx_to_logratio_te.setdefault(txid, []).append(float(logratio_te))
                tx_to_residual_te.setdefault(txid, []).append(float(residual_te))
                tx_to_rpf.setdefault(txid, []).append(float(rpf))
                tx_to_rna.setdefault(txid, []).append(float(rna))
            except ValueError:
                print(fname)
                print(line)
                exit()

print(f"Found {len(tx_to_logratio_te)} unique sequences.", end=" ")

with open(output_dir + "/" + output_fname, "w") as output:
    output.write(header)
    count = 0
    for txid, te_list in tx_to_logratio_te.items():
        if len(te_list) >= threshold:
            count += 1
            logratio_te_avg = avg(te_list)
            residual_te_avg = avg(tx_to_residual_te[txid])
            rpf_avg = avg(tx_to_rpf[txid])
            rna_avg = avg(tx_to_rna[txid])
            print(txid, *tx_to_detail[txid], logratio_te_avg, residual_te_avg, rpf_avg, rna_avg, sep="\t", file=output)

print(f"Total {count} sequences survived. ({count / len(tx_to_logratio_te):.2%})")
