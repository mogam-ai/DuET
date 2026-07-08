#!/usr/bin/env python3
"""DuET multitarget benchmark - plain 10-fold CV (no inner ensemble).

Like run_single.py (one model per outer fold, saves predictions.tsv), but trains
the multi-target model on the multitask TE table (76 celltypes at once), mirroring
the plain multitask training path (kfold test, no inner CV). Contrast with
run_multitarget.py, which adds an inner 9-fold top-5 ensemble on top of each outer fold.

Uses benchlib splits so predictions align with the other benchmark models.
No WandB. AdamW + ReduceLROnPlateau + early stopping.

Usage:
    conda activate benchmark_torch && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    python benchmarks/duet_bench/run_multi_cv.py --config benchmarks/config.yaml --gpus 0,1
    python benchmarks/duet_bench/run_multi_cv.py --config benchmarks/config.yaml --gpu 0 --folds 0
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from duet.configs.config import LossConfig

from benchlib import check_cached, Timer, load_config, generate_splits, load_indices, compute_metrics
from benchlib.config import get_data_path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # benchmarks/
from duet_bench.run_multitarget import MultiTargetDataset
from duet_bench.model_config import load_model_config, get_nn_classes, amp_enabled

torch.set_float32_matmul_precision('high')

MODEL_CLS = None       # MultiModel class, set by main() from model_config
MODEL_PARAMS = {}      # constructor kwargs
USE_AMP = False        # 16-mixed precision flag
DEFAULTS = {}          # train params (yaml 'train' section)


def run_fold(train_df, val_df, test_df, te_cols, output_dir, device):
    """Train one multi-target model on this outer fold; save predictions.tsv + metrics."""
    os.makedirs(output_dir, exist_ok=True)
    n_targets = len(te_cols)

    train_ds = MultiTargetDataset(train_df, te_cols,
                                  DEFAULTS['utr_seq_size'], DEFAULTS['cds_seq_size'])
    val_ds   = MultiTargetDataset(val_df, te_cols,
                                  DEFAULTS['utr_seq_size'], DEFAULTS['cds_seq_size'])
    test_ds  = MultiTargetDataset(test_df, te_cols,
                                  DEFAULTS['utr_seq_size'], DEFAULTS['cds_seq_size'])

    train_loader = DataLoader(train_ds, batch_size=DEFAULTS['batch_size'],
                              shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=4)

    loss_cfg = LossConfig(name='huber')
    model = MODEL_CLS(n_targets=n_targets, dropout=DEFAULTS['dropout'],
                      loss_cfg=loss_cfg, **MODEL_PARAMS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULTS['lr'],
                                  weight_decay=DEFAULTS['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=DEFAULTS['lr_factor'],
        patience=DEFAULTS['lr_patience'], min_lr=1e-7)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_val_loss, patience_count, best_state = float('inf'), 0, None
    for epoch in range(DEFAULTS['max_epochs']):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast('cuda', dtype=torch.float16, enabled=USE_AMP):
                _, loss = model(batch)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast('cuda', dtype=torch.float16, enabled=USE_AMP):
                    _, loss = model(batch)
                if not torch.isnan(loss):
                    val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses)) if val_losses else float('inf')
        scheduler.step(val_loss)

        marker = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
            marker = ' *'
        else:
            patience_count += 1
            if patience_count >= DEFAULTS['patience']:
                print(f"    early stop at epoch {epoch}", flush=True)
                break
        if epoch % 10 == 0 or marker:
            print(f"    epoch {epoch:3d} | val_loss={val_loss:.4f}{marker}", flush=True)

    # Test with best weights
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            yhat, _ = model.predict(batch)
            preds.append(yhat.cpu().numpy())
    pred = np.concatenate(preds)                       # (N, n_targets)
    true = np.stack([test_ds[i]['y'].numpy() for i in range(len(test_ds))])  # (N, n_targets)

    # Per-target metrics (skip celltypes with <10 measured transcripts in test).
    per_target = {}
    pearsons, spearmans, r2s = [], [], []
    for i, col in enumerate(te_cols):
        mask = ~np.isnan(true[:, i])
        if mask.sum() < 10:
            continue
        m = compute_metrics(true[mask, i], pred[mask, i])
        per_target[col] = m
        pearsons.append(m['pearson']); spearmans.append(m['spearman']); r2s.append(m['r2'])

    # Long-format predictions.tsv (txID, celltype, y_true, y_pred), NaN rows dropped.
    txid = test_df['txID'].values if 'txID' in test_df.columns else np.arange(len(test_df))
    rows = []
    for i, col in enumerate(te_cols):
        mask = ~np.isnan(true[:, i])
        ct = col[3:] if col.startswith('TE_') else col
        rows.append(pd.DataFrame({'txID': txid[mask], 'celltype': ct,
                                  'y_true': true[mask, i], 'y_pred': pred[mask, i]}))
    pd.concat(rows, ignore_index=True).to_csv(
        os.path.join(output_dir, 'predictions.tsv'), sep='\t', index=False)

    metrics = {
        'pearson':  float(np.mean(pearsons)),
        'spearman': float(np.mean(spearmans)),
        'r2':       float(np.mean(r2s)),
        'rmse':     0.0,
        'per_target': per_target,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--jobs-per-gpu', type=int, default=1)
    parser.add_argument('--folds', nargs='+', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--worker-id', type=int, default=None)
    parser.add_argument('--names', nargs='+', default=None,
                        help='Ignored; accepted for run_parallel worker compatibility')
    args = parser.parse_args()

    global MODEL_CLS, MODEL_PARAMS, USE_AMP, DEFAULTS
    mcfg = load_model_config('duet')
    MODEL_CLS, _ = get_nn_classes(mcfg['model_class'])   # MultiModel
    MODEL_PARAMS = mcfg.get('model_params', {})
    USE_AMP      = amp_enabled(mcfg['precision'])
    DEFAULTS     = mcfg['train']
    model_dirname = mcfg['output_names'].get('multi_cv', 'DuET_multi_cv')
    print(f"[model=duet] class={mcfg['model_class']} params={MODEL_PARAMS} "
          f"precision={mcfg['precision']} out={model_dirname}", flush=True)

    cfg = load_config(args.config)
    output_root = cfg['output_dir']

    mt_cfg     = cfg['datasets']['celltype_te_multi']
    mt_name    = mt_cfg['names'][0]                              # e.g. 'multitask'
    ds_output  = os.path.join(output_root, 'celltype_te_multi', mt_name)
    splits_dir = os.path.join(ds_output, 'splits')              # shared with RiboNN/multi
    mt_data    = get_data_path(mt_cfg, mt_name)                 # .../multitask_TE.tsv

    if args.gpus is not None:
        from benchlib import run_parallel
        gpus = [int(g) for g in args.gpus.split(',')]
        all_jobs = [{'name': f'celltype_te_multi/{mt_name}', 'config': args.config}]
        extra = []
        if args.folds:
            extra += ['--folds'] + [str(f) for f in args.folds]
        if args.force:
            extra += ['--force']
        run_parallel(__file__, all_jobs, gpus, args.jobs_per_gpu, extra_args=extra)
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading {mt_data}", flush=True)
    df = pd.read_csv(mt_data, sep='\t')
    te_cols = [c for c in df.columns if c.startswith('TE_')]
    print(f"n_targets={len(te_cols)}, n_rows={len(df)}", flush=True)

    # Same KFold splits as RiboNN/run_multitarget (seed=42), so predictions align.
    n_folds = generate_splits(mt_data, splits_dir, mt_cfg['k'], mt_cfg['seed'],
                              scaling=mt_cfg.get('scaling', 'none'),
                              skip_if_exists=True)

    all_metrics = []
    for fold_i in range(n_folds):
        if args.folds and fold_i not in args.folds:
            continue

        fold_dir = os.path.join(splits_dir, f'fold_{fold_i}')
        out      = os.path.join(ds_output, model_dirname, f'fold_{fold_i}')

        if not args.force:
            cached = check_cached(out)
            if cached:
                all_metrics.append(cached)
                print(f"  fold {fold_i}: cached  pearson={cached['pearson']:.4f}")
                continue

        indices  = load_indices(fold_dir)
        train_df = df.iloc[indices['train']].reset_index(drop=True)
        val_df   = df.iloc[indices['val']].reset_index(drop=True)
        test_df  = df.iloc[indices['test']].reset_index(drop=True)

        print(f"\n{'='*50}\nfold {fold_i}  train={len(train_df)} val={len(val_df)} test={len(test_df)}\n{'='*50}")
        with Timer() as t:
            metrics = run_fold(train_df, val_df, test_df, te_cols, out, device)
        metrics['elapsed_sec'] = t.elapsed
        with open(os.path.join(out, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        all_metrics.append(metrics)
        print(f"  fold {fold_i}: pearson={metrics['pearson']:.4f} "
              f"spearman={metrics['spearman']:.4f} [{t.elapsed:.0f}s]")

    if all_metrics:
        avg = {k: float(np.mean([m[k] for m in all_metrics]))
               for k in ('pearson', 'spearman', 'r2', 'rmse')}
        print(f"  AVG: pearson={avg['pearson']:.4f} spearman={avg['spearman']:.4f} "
              f"r2={avg['r2']:.4f}")


if __name__ == '__main__':
    main()
