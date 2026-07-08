#!/usr/bin/env python3
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""Unified aggregation of DuET + RiboNN across single / multi / finetune tasks.

----------------------------------------------------------------------------
Why this script exists
----------------------------------------------------------------------------
The previous collect_single / collect_multitarget / collect_finetune scripts
A single entry point that distinguishes "inner fold 0 only" vs "top-5 inner
ensemble" consistently across tasks.

----------------------------------------------------------------------------
Terminology
----------------------------------------------------------------------------
- outer fold : The outer split of 10-fold cross validation; defines the test set.
- inner fold : Each outer fold's train+val further split into 9 parts; for model selection/ensemble.
- inner0     : Prediction using only inner fold 0 (baseline without ensemble effect).
- ensemble   : Mean of predictions from the top-5 inner folds ranked by val_r2 (standard reporting).
- mix        : Mean of DuET + RiboNN predictions after aligning by txID.

----------------------------------------------------------------------------
Aggregation only (compare-only)
----------------------------------------------------------------------------
Inner predictions are pre-saved by training scripts with --predict-only (this
script does not use a GPU; it only collects those results):

  - run_multitarget.py --predict-only  -> DuET multi  inner_<i>/test_pred.npz
  - run_finetune.py    --predict-only  -> DuET ft     inner_<i>/test_pred.npz
  - baselines/RiboNN/run.py --predict-only -> RiboNN  inner_<i>/test_pred.npz
  - run_single.py already saves per-fold predictions.tsv during training (for single)

Aggregation:

       python duet_bench/collect_all.py --task single
       python duet_bench/collect_all.py --task multi
       python duet_bench/collect_all.py --task ft

----------------------------------------------------------------------------
Methods compared per task
----------------------------------------------------------------------------
single : duet,                              (DuET single -- no inner)
         ribonn_inner0, ribonn_ensemble,     (RiboNN -- inner0 / top5)
         mix_duet_ribonn_inner0, mix_duet_ribonn_ensemble
multi  : duet, duet_ens,
         ribonn, ribonn_ens, mix, mix_ens    (per celltype -- per_target unrolled)
ft     : Same method set as multi (but each model is trained per celltype independently)

----------------------------------------------------------------------------
Output (long format, preserving per-fold rows -> facilitates downstream statistics)
----------------------------------------------------------------------------
  <output_root>/collect_all/<task>_long.csv
  columns: task, celltype, outer_fold, method, pearson, spearman, r2, rmse, n
  (n = number of samples actually used for metric computation; may vary by model/alignment)

----------------------------------------------------------------------------
npz locations (saved by training scripts into inner directories)
----------------------------------------------------------------------------
  multi: <mt_base>/<model_dir>/fold_<F>/inner_<I>/test_pred.npz   (all celltypes (N,75))
  ft   : <ct_base>/<ct>/<model_dir>/fold_<F>/inner_<I>/test_pred.npz  (single target)
"""
import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

_HERE = os.path.dirname(os.path.abspath(__file__))          # benchmarks/duet_bench
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))        # repo root

DUET_MODELS = ['duet']
TOP_K = 5                                                   # ensemble = top-5 inner by val_r2


# -----------------------------------------------------------------------------
# Config / path helpers
# -----------------------------------------------------------------------------
def _cfg(config):
    """Read benchmark config.yaml and return (full dict, output_dir)."""
    import yaml
    with open(config) as f:
        cfg = yaml.safe_load(f)
    return cfg, cfg['output_dir']


def _mt_info(cfg):
    """Extract multitask dataset info.

    Returns:
        mt_cfg  : celltype_te_multi dataset config block
        mt_name : multitask dataset name (e.g., multitask)
        mt_data : path to the actual multitask tsv file (contains TE_* columns for all celltypes)
    """
    mt_cfg = cfg['datasets']['celltype_te_multi']
    mt_name = mt_cfg['names'][0]
    mt_data = os.path.join(mt_cfg['base_path'], mt_cfg['data_pattern'].format(name=mt_name))
    return mt_cfg, mt_name, mt_data


def _ct_names(args, cfg):
    """Target celltype list. Can be restricted to a subset via --celltypes."""
    all_cts = cfg['datasets']['celltype_te']['names']
    return [c for c in all_cts if c in args.celltypes] if args.celltypes else all_cts


def _metrics(t, p):
    """Compute regression metrics between predictions and ground truth.

    Positions where either value is NaN are excluded. If fewer than 10 valid
    samples remain, all metrics are NaN.
    n is the number of samples actually used -- it may differ across models/alignments
    and is reported alongside the metrics.
    """
    t, p = np.asarray(t, float), np.asarray(p, float)
    mask = ~(np.isnan(t) | np.isnan(p))
    if mask.sum() < 10:
        return {'pearson': float('nan'), 'spearman': float('nan'),
                'r2': float('nan'), 'rmse': float('nan'), 'n': int(mask.sum())}
    t, p = t[mask], p[mask]
    return {
        'pearson':  float(stats.pearsonr(t, p)[0]),
        'spearman': float(stats.spearmanr(t, p)[0]),
        'r2':       float(r2_score(t, p)),
        'rmse':     float(np.sqrt(np.mean((t - p) ** 2))),
        'n':        int(mask.sum()),
    }



# =============================================================================
# COMPARE shared helpers
# =============================================================================
def _read_pred_tsv(path):
    """predictions.tsv -> (txid, y_true, y_pred). Returns (None, None, None) if missing."""
    if not os.path.exists(path):
        return None, None, None
    df = pd.read_csv(path, sep='\t')
    return (df['txID'].values if 'txID' in df.columns else None,
            df['y_true'].values if 'y_true' in df.columns else None,
            df['y_pred'].values if 'y_pred' in df.columns else None)


def _align(txid_a, pred_a, txid_b, pred_b, txid_true, y_true):
    """Align two predictions (pred_a, pred_b) and ground truth by txID intersection.

    If txIDs are unavailable, assumes the same order and truncates to the shortest.
    Returns: (aligned y_true, aligned pred_a, aligned pred_b)
    """
    if txid_a is None or txid_b is None:
        n = min(len(pred_a), len(pred_b), len(y_true))
        return y_true[:n], pred_a[:n], pred_b[:n]
    pos_b = {tx: i for i, tx in enumerate(txid_b)}
    pos_t = {tx: i for i, tx in enumerate(txid_true)}
    common = [tx for tx in txid_a if tx in pos_b and tx in pos_t]
    if not common:
        return np.array([]), np.array([]), np.array([])
    ia = {tx: i for i, tx in enumerate(txid_a)}
    return (y_true[[pos_t[tx] for tx in common]],
            pred_a[[ia[tx] for tx in common]],
            pred_b[[pos_b[tx] for tx in common]])


def _load_npz_inners(model_fold_dir):
    """Load inner_*/test_pred.npz from one outer fold directory -> list of dicts (in inner order).

    Reads the inner_<i>/test_pred.npz files saved by run_multitarget / run_finetune / RiboNN run.py
    during training/prediction. The 'true' field may be absent and is treated as optional.
    """
    items = []
    inner_dirs = sorted(glob(os.path.join(model_fold_dir, 'inner_*')),
                        key=lambda p: int(p.rsplit('_', 1)[1]))
    for idir in inner_dirs:
        npz = os.path.join(idir, 'test_pred.npz')
        if not os.path.exists(npz):
            continue
        d = np.load(npz, allow_pickle=True)
        items.append({'pred': d['pred'], 'txid': list(d['txid']),
                      'true': d['true'] if 'true' in d else None,
                      'val_r2': float(d['val_r2']),
                      'te_cols': list(d['te_cols']) if 'te_cols' in d else None})
    return items


def _topk_mean(items, k):
    """Mean prediction of the top-k inner folds ranked by val_r2 (= ensemble prediction)."""
    return np.mean([r['pred'] for r in sorted(items, key=lambda x: x['val_r2'], reverse=True)[:k]], axis=0)


# =============================================================================
# COMPARE -- single
# =============================================================================
def compare_single(args):
    """Single task aggregation: reads saved predictions.tsv files for comparison (no GPU needed).

    DuET uses a single predictions.tsv per fold.
    RiboNN has two variants: ensemble (fold-level predictions.tsv, top5 mean) and
    inner0 (inner_0/predictions.tsv).
    mix is the mean of DuET + RiboNN after txID alignment.
    """
    cfg, output_root = _cfg(args.config)
    ct_base = os.path.join(output_root, 'celltype_te')
    out_dir = os.path.join(output_root, 'collect_all'); os.makedirs(out_dir, exist_ok=True)
    k = cfg['datasets']['celltype_te'].get('k', 10)
    rows = []
    for ct in _ct_names(args, cfg):
        for fold_i in range(k):
            if args.folds and fold_i not in args.folds:
                continue
            # Pre-read (txid, y_true, y_pred) for each method
            preds = {}
            for tag, sub in [('duet', 'DuET')]:
                preds[tag] = _read_pred_tsv(os.path.join(ct_base, ct, sub, f'fold_{fold_i}', 'predictions.tsv'))
            preds['ribonn_ensemble'] = _read_pred_tsv(os.path.join(ct_base, ct, 'RiboNN', f'fold_{fold_i}', 'predictions.tsv'))
            preds['ribonn_inner0'] = _read_pred_tsv(os.path.join(ct_base, ct, 'RiboNN', f'fold_{fold_i}', 'inner_0', 'predictions.tsv'))

            # Simple methods evaluated directly against their own y_true
            # Naming convention: ribonn_ensemble -> ribonn_top5 (top5 inner ensemble)
            for tag, mname in [('duet', 'duet'),
                               ('ribonn_ensemble', 'ribonn_top5')]:
                tx, yt, yp = preds[tag]
                if yp is None:
                    continue
                rows.append({'task': 'single', 'celltype': ct, 'outer_fold': fold_i,
                             'method': mname, **_metrics(yt, yp)})

            # DuET _common: evaluate only on the txID intersection with RiboNN (fair comparison).
            # RiboNN txIDs are the same for ensemble/inner0, so use the ensemble file as reference.
            tx_rb = preds['ribonn_ensemble'][0]
            if tx_rb is not None:
                rb_set = set(tx_rb)
                for tag in ['duet']:
                    tx, yt, yp = preds[tag]
                    if yp is None or tx is None:
                        continue
                    keep = np.array([t in rb_set for t in tx])
                    if keep.sum() >= 10:
                        rows.append({'task': 'single', 'celltype': ct, 'outer_fold': fold_i,
                                     'method': f'{tag}_common',
                                     **_metrics(yt[keep], yp[keep])})

            # ribonn (inner0): evaluate inner_0 prediction aligned by txID with ensemble file's y_true
            tx_e, yt_e, _ = preds['ribonn_ensemble']
            tx0, _, yp0 = preds['ribonn_inner0']
            if yp0 is not None and yt_e is not None:
                yt_a, yp0_a, _ = _align(tx0, yp0, tx_e, yt_e, tx_e, yt_e)
                if len(yt_a) >= 10:
                    rows.append({'task': 'single', 'celltype': ct, 'outer_fold': fold_i,
                                 'method': 'ribonn', **_metrics(yt_a, yp0_a)})

            # mix: DuET + RiboNN (top5 / inner0) -- mean after txID alignment
            tx5, yt5, yp5 = preds['duet']
            for rib_method, (txr, ytr, ypr) in [('ribonn_ensemble', preds['ribonn_ensemble']),
                                                ('ribonn_inner0', preds['ribonn_inner0'])]:
                if yp5 is None or ypr is None:
                    continue
                yt_m, p5, pr = _align(tx5, yp5, txr, ypr, tx5, yt5)
                if len(yt_m) >= 10:
                    name = 'mix_duet_ribonn' if 'inner0' in rib_method else 'mix_duet_ribonn_top5'
                    rows.append({'task': 'single', 'celltype': ct, 'outer_fold': fold_i,
                                 'method': name, **_metrics(yt_m, (p5 + pr) / 2)})

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, 'single_long.csv')
    df.to_csv(path, index=False)
    print(f'Saved {path} ({len(df)} rows)')
    if len(df):
        print(df.groupby('method')[['pearson', 'spearman', 'r2', 'rmse', 'n']].mean().round(4).to_string())


# =============================================================================
# COMPARE -- multi (per_target unrolled)
# =============================================================================
def compare_multi(args):
    """Multi task aggregation: unroll (N,75) npz predictions per celltype (target) for comparison.

    For each model, compute both inner0 (first inner) and ens (top-5) variants,
    plus DuET + ribonn mix (inner0/ens). All predictions are aligned by txID to the
    reference (DuET) order so the same transcript set is compared.
    """
    cfg, output_root = _cfg(args.config)
    mt_cfg, mt_name, mt_data = _mt_info(cfg)
    mt_base = os.path.join(output_root, 'celltype_te_multi', mt_name)
    out_dir = os.path.join(output_root, 'collect_all'); os.makedirs(out_dir, exist_ok=True)

    # Ground truth is not in the npz; load txID -> per-celltype TE from the multitask tsv
    mt_df = pd.read_csv(mt_data, sep='\t')
    truth = {c: dict(zip(mt_df['txID'], mt_df[c]))
             for c in mt_df.columns if c.startswith('TE_')}

    duet_dirs = {'duet': 'DuET_multi'}
    rows = []
    # Collect outer fold numbers present (based on fold_* dirs across model directories)
    folds = sorted({int(os.path.basename(p).rsplit('_', 1)[1])
                    for d in list(duet_dirs.values()) + ['RiboNN']
                    for p in glob(os.path.join(mt_base, d, 'fold_*'))})
    for fold_i in folds:
        if args.folds and fold_i not in args.folds:
            continue
        # Load models available for this fold
        loaded = {}
        for tag, d in duet_dirs.items():
            it = _load_npz_inners(os.path.join(mt_base, d, f'fold_{fold_i}'))
            if it:
                loaded[tag] = it
        ribo = _load_npz_inners(os.path.join(mt_base, 'RiboNN', f'fold_{fold_i}'))
        if ribo:
            loaded['ribonn'] = ribo
        if not loaded:
            continue

        # Reference: celltype columns/order, txID reference (prefer DuET)
        ref = loaded.get('duet') or loaded.get('ribonn')
        if ref is None:
            continue
        te_cols = ref[0]['te_cols']; ref_txid = ref[0]['txid']

        # Precompute prediction matrices (inner0/top3/top5) and column/txID indices per model
        method_preds = {}
        for tag, items in loaded.items():
            cols = items[0]['te_cols']
            method_preds[tag] = {
                'cols': {c: i for i, c in enumerate(cols)},   # celltype column name -> column index
                'top5': _topk_mean(items, 5),                 # (N,T) top-5 inner ensemble
                'top3': _topk_mean(items, 3),                 # (N,T) top-3 inner ensemble
                'in0': items[0]['pred'],                      # (N,T) inner 0
                'tx': items[0]['txid'],                       # txID order for this model's predictions
            }

        # Pool selection is fold-level (shared across all celltypes): combine DuET inner + ribonn inner,
        # pick val_r2 top-5. Precompute (pred, column map, txID positions) for each selected inner.
        pool_top = None
        if 'duet' in loaded and 'ribonn' in loaded:
            pool = loaded['duet'] + loaded['ribonn']
            sel = sorted(pool, key=lambda x: x['val_r2'], reverse=True)[:TOP_K]
            pool_top = [{'pred': it['pred'],
                         'colmap': {c: i for i, c in enumerate(it['te_cols'])},
                         'txpos': {t: j for j, t in enumerate(it['txid'])},
                         'txid': it['txid']} for it in sel]

        for ci, ct in enumerate(te_cols):
            ct_name = ct.replace('TE_', '') + '_TE'           # TE_pc3 -> pc3_TE
            ct_truth = truth.get(ct, {})                       # txID -> ground truth TE
            # RiboNN txID set (used for _common evaluation if available)
            rb_txset = set(method_preds['ribonn']['tx']) if 'ribonn' in loaded else None

            def emit(method, pred_vec, tx_for_pred, restrict=None):
                """Align pred_vec with ground truth by txID and append a metric row.

                If restrict is given, additionally limit to that txID set (for fair _common comparison).
                """
                pos = {t: j for j, t in enumerate(tx_for_pred)}
                common = [t for t in tx_for_pred if t in ct_truth and (restrict is None or t in restrict)]
                yt = np.array([ct_truth[t] for t in common])
                yp = np.array([pred_vec[pos[t]] for t in common])
                rows.append({'task': args.task, 'celltype': ct_name, 'outer_fold': fold_i,
                             'method': method, **_metrics(yt, yp)})

            # DuET single models: inner0(full/common) + top5(full/common) = 4 variants
            for tag in ['duet']:
                if tag not in loaded:
                    continue
                mp = method_preds[tag]
                if ct not in mp['cols']:
                    continue
                rci = mp['cols'][ct]
                emit(tag, mp['in0'][:, rci], mp['tx'])
                emit(f'{tag}_top5', mp['top5'][:, rci], mp['tx'])
                if rb_txset is not None:                       # fair comparison on RiboNN txID intersection
                    emit(f'{tag}_common', mp['in0'][:, rci], mp['tx'], restrict=rb_txset)
                    emit(f'{tag}_top5_common', mp['top5'][:, rci], mp['tx'], restrict=rb_txset)
            # RiboNN (its own txID set is already the common set)
            if 'ribonn' in loaded:
                mp = method_preds['ribonn']
                if ct in mp['cols']:
                    rci = mp['cols'][ct]
                    emit('ribonn', mp['in0'][:, rci], mp['tx'])
                    emit('ribonn_top5', mp['top5'][:, rci], mp['tx'])

            # mix: DuET + ribonn, combos inner0 / top5+top5 / top3+top3 (txID intersection of both).
            if 'duet' in loaded and 'ribonn' in loaded:
                mv = method_preds['duet']; mr = method_preds['ribonn']
                if ct in mv['cols'] and ct in mr['cols']:
                    vci = mv['cols'][ct]; rci = mr['cols'][ct]
                    posv = {t: j for j, t in enumerate(mv['tx'])}
                    posr = {t: j for j, t in enumerate(mr['tx'])}
                    common = [t for t in mv['tx'] if t in posr and t in ct_truth]
                    yt = np.array([ct_truth[t] for t in common])
                    for name, vkey, rkey in [('mix_duet_ribonn', 'in0', 'in0'),
                                             ('mix_duet_top5_ribonn_top5', 'top5', 'top5'),
                                             ('mix_duet_top3_ribonn_top3', 'top3', 'top3')]:
                        pv = np.array([mv[vkey][posv[t], vci] for t in common])
                        pr = np.array([mr[rkey][posr[t], rci] for t in common])
                        rows.append({'task': args.task, 'celltype': ct_name, 'outer_fold': fold_i,
                                     'method': name, **_metrics(yt, (pv + pr) / 2)})

            # pool: from the 20-inner DuET+ribonn pool, average the top-5 by val_r2 per celltype column
            if pool_top is not None and all(ct in it['colmap'] for it in pool_top):
                common = [t for t in pool_top[0]['txid']
                          if t in ct_truth and all(t in it['txpos'] for it in pool_top)]
                if len(common) >= 10:
                    yt = np.array([ct_truth[t] for t in common])
                    aligned_preds = [np.array([it['pred'][it['txpos'][t], it['colmap'][ct]]
                                               for t in common]) for it in pool_top]
                    rows.append({'task': args.task, 'celltype': ct_name, 'outer_fold': fold_i,
                                 'method': 'mix_duet_ribonn_pool_top5',
                                 **_metrics(yt, np.mean(aligned_preds, axis=0))})

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, f'{args.task}_long.csv')
    df.to_csv(path, index=False)
    print(f'Saved {path} ({len(df)} rows)')
    if len(df):
        print(df.groupby('method')[['pearson', 'spearman', 'r2', 'rmse', 'n']].mean().round(4).to_string())


# ============================================================================
# COMPARE -- ft (per-celltype npz)
# ============================================================================
def compare_ft(args):
    """ft task aggregation: compare per-celltype single-target npz.

    Same method set as multi (per celltype), but each celltype model is independent,
    so npz lives under <ct>/<model_dir>/fold_<F>/inner_<I>/test_pred.npz.
    The inner npz is saved by the training script's --predict-only (includes true).
    """
    cfg, output_root = _cfg(args.config)
    ct_base = os.path.join(output_root, 'celltype_te')
    out_dir = os.path.join(output_root, 'collect_all'); os.makedirs(out_dir, exist_ok=True)

    # truth may be absent from npz (DuET ft), so load per-celltype TE by txID from the multitask tsv
    _, _, mt_data = _mt_info(cfg)
    mt_df = pd.read_csv(mt_data, sep='\t')
    truth = {c: dict(zip(mt_df['txID'], mt_df[c]))
             for c in mt_df.columns if c.startswith('TE_')}

    duet_dirs = {'duet': 'DuET_ft'}
    k = cfg['datasets']['celltype_te'].get('k', 10)
    rows = []
    for ct in _ct_names(args, cfg):
        ct_truth = truth.get('TE_' + ct.replace('_TE', ''), {})   # txID -> true TE
        if not ct_truth:
            continue
        for fold_i in range(k):
            if args.folds and fold_i not in args.folds:
                continue
            # load models present in this fold (<ct>/<model_dir>/fold_<F>)
            loaded = {}
            for tag, d in duet_dirs.items():
                it = _load_npz_inners(os.path.join(ct_base, ct, d, f'fold_{fold_i}'))
                if it:
                    loaded[tag] = it
            ribo = _load_npz_inners(os.path.join(ct_base, ct, 'RiboNN_ft', f'fold_{fold_i}'))
            if ribo:
                loaded['ribonn'] = ribo
            if not loaded:
                continue

            def aligned(items, mode, restrict=None):
                """Align items' predictions (mode: in0/top3/top5) to truth by txID.

                If restrict is given, further limit to that txID set (_common fair comparison).
                """
                tx = items[0]['txid']
                pos = {t: j for j, t in enumerate(tx)}
                if mode == 'top5':
                    pred = _topk_mean(items, 5)
                elif mode == 'top3':
                    pred = _topk_mean(items, 3)
                else:  # in0
                    pred = items[0]['pred']
                common = [t for t in tx if t in ct_truth and (restrict is None or t in restrict)]
                yt = np.array([ct_truth[t] for t in common])
                yp = np.array([pred[pos[t]] for t in common])
                return common, yt, yp

            # RiboNN txID set (if present, used for DuET _common evaluation)
            rb_txset = set(loaded['ribonn'][0]['txid']) if 'ribonn' in loaded else None

            # DuET single models: inner0(full/common) + top5(full/common) = 4 variants
            for tag in ['duet']:
                if tag not in loaded:
                    continue
                for mode, suf in [('in0', ''), ('top5', '_top5')]:
                    _, yt, yp = aligned(loaded[tag], mode)
                    if len(yt) >= 10:
                        rows.append({'task': 'ft', 'celltype': ct, 'outer_fold': fold_i,
                                     'method': f'{tag}{suf}', **_metrics(yt, yp)})
                    if rb_txset is not None:                   # fair comparison on RiboNN txID intersection
                        _, ytc, ypc = aligned(loaded[tag], mode, restrict=rb_txset)
                        if len(ytc) >= 10:
                            rows.append({'task': 'ft', 'celltype': ct, 'outer_fold': fold_i,
                                         'method': f'{tag}{suf}_common', **_metrics(ytc, ypc)})
            # RiboNN (its own txID set is already the common set)
            if 'ribonn' in loaded:
                for mode, name in [('in0', 'ribonn'), ('top5', 'ribonn_top5')]:
                    _, yt, yp = aligned(loaded['ribonn'], mode)
                    if len(yt) >= 10:
                        rows.append({'task': 'ft', 'celltype': ct, 'outer_fold': fold_i,
                                     'method': name, **_metrics(yt, yp)})
            # mix: DuET + ribonn. inner0 / top5+top5 / top3+top3
            if 'duet' in loaded and 'ribonn' in loaded:
                for mode, name in [('in0', 'mix_duet_ribonn'),
                                   ('top5', 'mix_duet_top5_ribonn_top5'),
                                   ('top3', 'mix_duet_top3_ribonn_top3')]:
                    txv = loaded['duet'][0]['txid']; posv = {t: j for j, t in enumerate(txv)}
                    txr = loaded['ribonn'][0]['txid']; posr = {t: j for j, t in enumerate(txr)}
                    if mode == 'top5':
                        predv, predr = _topk_mean(loaded['duet'], 5), _topk_mean(loaded['ribonn'], 5)
                    elif mode == 'top3':
                        predv, predr = _topk_mean(loaded['duet'], 3), _topk_mean(loaded['ribonn'], 3)
                    else:
                        predv, predr = loaded['duet'][0]['pred'], loaded['ribonn'][0]['pred']
                    common = [t for t in txv if t in posr and t in ct_truth]
                    yt = np.array([ct_truth[t] for t in common])
                    pv = np.array([predv[posv[t]] for t in common])
                    pr = np.array([predr[posr[t]] for t in common])
                    if len(yt) >= 10:
                        rows.append({'task': 'ft', 'celltype': ct, 'outer_fold': fold_i,
                                     'method': name, **_metrics(yt, (pv + pr) / 2)})

            # pool: merge DuET inner + ribonn inner into one pool, pick top-5 by val_r2, then average.
            # (Unlike mix, which averages each model's top5 first, here the top 5 are chosen
            #  from all 20 inners -- favoring whichever of DuET/RiboNN is better.)
            if 'duet' in loaded and 'ribonn' in loaded:
                pool = loaded['duet'] + loaded['ribonn']
                top = sorted(pool, key=lambda x: x['val_r2'], reverse=True)[:TOP_K]
                txsets = [set(it['txid']) for it in top]
                common = [t for t in top[0]['txid']
                          if t in ct_truth and all(t in s for s in txsets)]
                if len(common) >= 10:
                    yt = np.array([ct_truth[t] for t in common])
                    aligned_preds = []
                    for it in top:
                        pos = {t: j for j, t in enumerate(it['txid'])}
                        aligned_preds.append(np.array([it['pred'][pos[t]] for t in common]))
                    yp = np.mean(aligned_preds, axis=0)
                    rows.append({'task': 'ft', 'celltype': ct, 'outer_fold': fold_i,
                                 'method': 'mix_duet_ribonn_pool_top5', **_metrics(yt, yp)})

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, 'ft_long.csv')
    df.to_csv(path, index=False)
    print(f'Saved {path} ({len(df)} rows)')
    if len(df):
        print(df.groupby('method')[['pearson', 'spearman', 'r2', 'rmse', 'n']].mean().round(4).to_string())


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--task', choices=['single', 'multi', 'ft'], required=True)
    ap.add_argument('--folds', nargs='+', type=int, default=None, help='only these outer folds')
    ap.add_argument('--celltypes', nargs='+', default=None, help='only these celltypes')
    args = ap.parse_args()

    # Aggregation only. Inner prediction npz files are pre-saved by
    # run_multitarget/run_finetune/RiboNN run.py via --predict-only into inner_<i>/test_pred.npz.
    {'single': compare_single, 'multi': compare_multi, 'ft': compare_ft}[args.task](args)


if __name__ == '__main__':
    main()
