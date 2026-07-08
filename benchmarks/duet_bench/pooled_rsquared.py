#!/usr/bin/env python3
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""Pooled Pearson R / R2 (fraction of total variance explained) + celltype variance decomposition.

Beyond Spearman (rank), reports how much of the total (gene x cell type) TE variance
each model explains (R2) and its Pearson R, decomposed into between-celltype and
within-celltype variance to check whether celltype-specific signal is captured.

Targets (key celltypes, excluding the all-celltype pseudo-column):
  - single-model baselines: optimus5p, framepool, translatelstm, UTR-LM
      -> pool each celltype's fold_*/predictions.tsv (y_true, y_pred)
  - DuET / RiboNN: multi 10-fold, inner_0 (no ensemble) predictions
      -> the target celltype columns of the multi npz (N, 75)

Pooled R2 is against each model's own test observations. The transcript set may
differ between single vs multi model groups, but the celltype set is the same.

Run (repo root):
    python benchmarks/duet_bench/pooled_rsquared.py --config benchmarks/config.yaml
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from collect_all import _cfg, _mt_info, _load_npz_inners   # noqa: E402

BASELINES = {'optimus5p': 'optimus5p', 'framepool': 'framepool',
             'translatelstm': 'translatelstm', 'utr-lm': 'UTR-LM'}
MULTI = {'duet': 'DuET_multi', 'ribonn': 'RiboNN'}    # uses inner_0
EXCLUDE = {'all-celltype'}


def target_celltypes(cfg, ct_base, truth):
    """Celltypes excluding all-celltype, having baseline predictions and a multi-truth column."""
    out = []
    for name in cfg['datasets']['celltype_te']['names']:
        stem = name.replace('_TE', '')
        if stem in EXCLUDE:
            continue
        if ('TE_' + stem) not in truth:
            continue
        if not os.path.exists(os.path.join(ct_base, name, 'optimus5p', 'fold_0', 'predictions.tsv')):
            continue
        out.append(name)
    return out


def pool_baseline(ct_base, model_dir, celltypes, k=10):
    """baseline: pool each celltype/fold predictions.tsv -> DataFrame(celltype, y, yhat)."""
    recs = []
    for name in celltypes:
        for f in range(k):
            p = os.path.join(ct_base, name, model_dir, f'fold_{f}', 'predictions.tsv')
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p, sep='\t')
            recs.append(pd.DataFrame({'celltype': name.replace('_TE', ''),
                                      'y': d['y_true'].to_numpy(float),
                                      'yhat': d['y_pred'].to_numpy(float)}))
    return pd.concat(recs, ignore_index=True) if recs else None


def pool_multi_inner0(mt_base, model_dir, celltypes, truth, k=10):
    """DuET/RiboNN multi: pool inner_0 predictions per celltype column -> DataFrame(celltype, y, yhat)."""
    want = {'TE_' + n.replace('_TE', '') for n in celltypes}
    recs = []
    for f in range(k):
        items = _load_npz_inners(os.path.join(mt_base, model_dir, f'fold_{f}'))
        if not items:
            continue
        it0 = items[0]                    # inner_0 (sorted) = no ensemble
        pred = it0['pred']                # (N, T)
        txid = it0['txid']; te_cols = it0['te_cols']
        colidx = {c: i for i, c in enumerate(te_cols)}
        for c in te_cols:
            if c not in want or c not in truth:
                continue
            ci = colidx[c]; ct_truth = truth[c]
            ys, ps = [], []
            for j, tx in enumerate(txid):
                if tx in ct_truth and np.isfinite(ct_truth[tx]):
                    ys.append(ct_truth[tx]); ps.append(pred[j, ci])
            if ys:
                recs.append(pd.DataFrame({'celltype': c[3:], 'y': ys, 'yhat': ps}))
    return pd.concat(recs, ignore_index=True) if recs else None


def compute_metrics(D):
    """Pooled R/R2, between/within-celltype decomposition, and per-celltype R2."""
    D = D[np.isfinite(D['y']) & np.isfinite(D['yhat'])].copy()
    y = D['y'].to_numpy(); yhat = D['yhat'].to_numpy()
    grand = y.mean()
    pooled_R = np.corrcoef(y, yhat)[0, 1]
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - grand) ** 2)
    pooled_R2 = 1.0 - ss_res / ss_tot

    dfg = D.groupby('celltype')
    ct_mean_y = dfg['y'].transform('mean').to_numpy()
    ss_between = np.sum((ct_mean_y - grand) ** 2)
    ss_within = np.sum((y - ct_mean_y) ** 2)

    e = y - yhat
    ct_mean_e = D.assign(e=e).groupby('celltype')['e'].transform('mean').to_numpy()
    within_R2 = 1.0 - np.sum((e - ct_mean_e) ** 2) / ss_within
    between_R2 = 1.0 - np.sum((ct_mean_e - e.mean()) ** 2) / ss_between

    per_ct_R2 = dfg.apply(lambda g: 1 - np.sum((g['y'] - g['yhat']) ** 2)
                          / np.sum((g['y'] - g['y'].mean()) ** 2))
    per_ct_R = dfg.apply(lambda g: np.corrcoef(g['y'], g['yhat'])[0, 1])
    return {
        'n_obs': len(y), 'n_celltype': D['celltype'].nunique(),
        'pooled_pearson_R': pooled_R,
        'pooled_R2_pct': pooled_R2 * 100,
        'var_between_ct_pct': ss_between / ss_tot * 100,
        'var_within_ct_pct': ss_within / ss_tot * 100,
        'within_ct_R2': within_R2,
        'between_ct_R2': between_R2,
        'mean_perct_R2': per_ct_R2.mean(),
        'mean_perct_R': per_ct_R.mean(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='benchmarks/config.yaml')
    ap.add_argument('--celltypes', nargs='+', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cfg, output_root = _cfg(args.config)
    ct_base = os.path.join(output_root, 'celltype_te')
    _, mt_name, mt_data = _mt_info(cfg)
    mt_base = os.path.join(output_root, 'celltype_te_multi', mt_name)

    mt_df = pd.read_csv(mt_data, sep='\t')
    truth = {c: dict(zip(mt_df['txID'], mt_df[c]))
             for c in mt_df.columns if c.startswith('TE_')}

    celltypes = args.celltypes or target_celltypes(cfg, ct_base, truth)
    print(f'celltypes ({len(celltypes)}): {[c.replace("_TE","") for c in celltypes]}', flush=True)

    results = {}
    for tag, mdir in BASELINES.items():
        D = pool_baseline(ct_base, mdir, celltypes)
        if D is not None:
            results[tag] = compute_metrics(D)
    for tag, mdir in MULTI.items():
        D = pool_multi_inner0(mt_base, mdir, celltypes, truth)
        if D is not None:
            results[tag] = compute_metrics(D)

    table = pd.DataFrame(results).T
    order = ['optimus5p', 'framepool', 'translatelstm', 'utr-lm', 'ribonn', 'duet']
    table = table.reindex([m for m in order if m in table.index])
    cols = ['n_obs', 'n_celltype', 'pooled_pearson_R', 'pooled_R2_pct',
            'var_between_ct_pct', 'var_within_ct_pct', 'within_ct_R2',
            'between_ct_R2', 'mean_perct_R2', 'mean_perct_R']
    table = table[cols]
    print('\n' + table.round(4).to_string())

    out = args.out or os.path.join(output_root, 'collect_all', 'pooled_rsquared.tsv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    table.to_csv(out, sep='\t')
    print(f'\nsaved: {out}', flush=True)


if __name__ == '__main__':
    main()
