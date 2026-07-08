#!/usr/bin/env python
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""Build motif-search datasets (XSTREME FASTA).

Conditions:
  TE cut : top/bottom percentile cutoff (te{pct}), configurable via --te-cutoff
  window : full_lenmatch (whole sequence, length-matched 1:1 within +-tol nt)
           5p{W} (5' end W nt), 3p{W} (3' end W nt); sequences shorter than W dropped

TE cutoff (--te-cutoff), window sizes (--windows), and length-match tolerance
(--tol) are configurable.

XSTREME FASTA (high = primary, low = control):
  te{pct}_full_lenmatch_{high,low}.fa
  te{pct}_5p{W}_{high,low}.fa
  te{pct}_3p{W}_{high,low}.fa
  header: >{txID}|TE={te:.4f}
"""
import os
import argparse
import numpy as np
import pandas as pd

PATH = "datasets/celltype_te/all-celltype_TE.tsv"
OUT = "motif_search/data"


def write_fasta(sub, col, path):
    with open(path, "w") as f:
        for _, r in sub.iterrows():
            f.write(f">{r['txID']}|TE={r['logratio_te']:.4f}\n{r[col]}\n")
    return len(sub)


def greedy_length_match(top, bot, tol):
    """1:1 length-match each top-TE seq to a bottom-TE seq within +-tol nt."""
    bot = bot.reset_index(drop=True)
    bl = bot["len"].values
    used = np.zeros(len(bot), dtype=bool)
    tk, bk = [], []
    for ti, L in top.sort_values("len").iterrows():
        av = np.where(~used)[0]
        if len(av) == 0:
            break
        j = np.argmin(np.abs(bl[av] - L["len"]))
        if abs(bl[av][j] - L["len"]) <= tol:
            used[av[j]] = True
            tk.append(ti); bk.append(av[j])
    return top.loc[tk], bot.iloc[bk]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--te-cutoff", type=int, nargs="+", default=[10],
                        help="TE top/bottom percentile cutoff(s) in %% (default 10, i.e. top/bottom 10%%)")
    parser.add_argument("--tol", type=int, default=10,
                        help="length-match tolerance in nt for full_lenmatch (default 10)")
    parser.add_argument("--windows", type=int, nargs="+", default=[100],
                        help="window sizes (nt) for 5'/3'-end subsets (default 100)")
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)

    df = pd.read_csv(PATH, sep="\t", usecols=["txID", "utr5", "logratio_te"]).dropna()
    df["utr5"] = df["utr5"].astype(str).str.upper().str.replace("U", "T")
    df["len"] = df["utr5"].str.len()

    summary = []
    for pct in args.te_cutoff:
        qfrac, qn = pct / 100.0, f"te{pct}"
        # full_lenmatch (whole sequence)
        d = df.copy()
        ql, qh = d["logratio_te"].quantile([qfrac, 1 - qfrac])
        top = d[d["logratio_te"] >= qh].copy()
        bot = d[d["logratio_te"] <= ql].copy()
        tm, bm = greedy_length_match(top, bot, args.tol)
        nh = write_fasta(tm, "utr5", f"{OUT}/{qn}_full_lenmatch_high.fa")
        nl = write_fasta(bm, "utr5", f"{OUT}/{qn}_full_lenmatch_low.fa")
        summary.append((f"{qn}_full_lenmatch", nh, nl))

        # windowed (W nt from each end; sequences shorter than W are dropped)
        for W in args.windows:
            dW = df[df["len"] >= W].copy()
            qlW, qhW = dW["logratio_te"].quantile([qfrac, 1 - qfrac])
            topW = dW[dW["logratio_te"] >= qhW].copy()
            botW = dW[dW["logratio_te"] <= qlW].copy()
            topW = topW.assign(w5=topW["utr5"].str[:W], w3=topW["utr5"].str[-W:])
            botW = botW.assign(w5=botW["utr5"].str[:W], w3=botW["utr5"].str[-W:])
            nh = write_fasta(topW, "w5", f"{OUT}/{qn}_5p{W}_high.fa")
            nl = write_fasta(botW, "w5", f"{OUT}/{qn}_5p{W}_low.fa")
            summary.append((f"{qn}_5p{W}", nh, nl))
            nh = write_fasta(topW, "w3", f"{OUT}/{qn}_3p{W}_high.fa")
            nl = write_fasta(botW, "w3", f"{OUT}/{qn}_3p{W}_low.fa")
            summary.append((f"{qn}_3p{W}", nh, nl))

    print(f"=== XSTREME FASTA ({len(summary)} conditions x high/low) | "
          f"te_cutoff={args.te_cutoff} tol={args.tol} windows={args.windows} ===")
    print(f"{'condition':<22} {'high':>6} {'low':>6}")
    for c, h, l in summary:
        print(f"{c:<22} {h:>6} {l:>6}")


if __name__ == "__main__":
    main()
