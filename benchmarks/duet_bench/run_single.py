#!/usr/bin/env python3
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""DuET benchmark - 10-fold CV using benchlib splits.

Uses the same splits, scalers, and metrics as other baselines for fair comparison.
No WandB. Trains with AdamW + ReduceLROnPlateau + early stopping.

Usage:
    conda activate benchmark_torch
    cd benchmarks
    python duet/run.py --config config.yaml --gpus 0,1 --jobs-per-gpu 1
    python duet/run.py --config config.yaml --gpu 0 --names celltype_te/hek293t_TE
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Add src to path for DuetSingletaskDataset and the DuET model classes

from duet.data.duet_singletask import DuetSingletaskDataset
from duet.models.loss import build_loss
from duet.configs.config import LossConfig

from benchlib import check_cached, Timer, load_config, generate_splits, generate_rank_split, \
    load_indices, load_scaler, compute_metrics
from benchlib.config import get_data_path, get_rank_paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # benchmarks/
from duet_bench.model_config import load_model_config, get_nn_classes, amp_enabled

torch.set_float32_matmul_precision('high')

# Data field defaults (kept here; not in model_config yaml). Train/model params
# (utr_seq_size, lr, dropout, max_epochs, ...) are overridden by the yaml 'train' section in main().
DEFAULTS = dict(
    utr_seq_size=500, cds_seq_size=3000,
    utr_channel_size=4, cds_channel_size=64,
    use_codon_encoding=True, use_sequence_feature=False,
    log_label=False, scale_label=False,
    utr_col='utr5', cds_col='cds', label_col='logratio_te', join_col='txID',
    lr=1e-3, weight_decay=0.02, batch_size=64,
    max_epochs=70, patience=10, lr_patience=5, lr_factor=0.5,
    dropout=0.4,
)

MODEL_CLS = None       # SingleModel class, set by main() from model_config
MODEL_PARAMS = {}      # constructor kwargs
USE_AMP = False        # 16-mixed precision flag


def run_fold(train_df, val_df, test_df, ds_cfg, output_dir, fold_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    label_col = ds_cfg['label_col']
    scale_fn, inverse_fn = load_scaler(fold_dir)

    # Write temp TSVs (DuetSingletaskDataset is file-based)
    with tempfile.TemporaryDirectory() as tmp:
        train_path = os.path.join(tmp, 'train.tsv')
        val_path   = os.path.join(tmp, 'val.tsv')
        test_path  = os.path.join(tmp, 'test.tsv')

        # Apply scaler to label before writing (DuetSingletaskDataset reads raw values)
        def write_scaled(df, path):
            d = df.copy()
            d[label_col] = scale_fn(d[label_col].values)
            d.to_csv(path, sep='\t', index=False)

        write_scaled(train_df, train_path)
        write_scaled(val_df,   val_path)
        test_df.to_csv(test_path, sep='\t', index=False)  # test: keep raw, evaluate after inverse

        dataset_kwargs = dict(
            utr_seq_size=DEFAULTS['utr_seq_size'],
            cds_seq_size=DEFAULTS['cds_seq_size'],
            utr_channel_size=DEFAULTS['utr_channel_size'],
            cds_channel_size=DEFAULTS['cds_channel_size'],
            use_codon_encoding=DEFAULTS['use_codon_encoding'],
            use_sequence_feature=False,
            utr_col=ds_cfg.get('utr_col', DEFAULTS['utr_col']),
            cds_col=ds_cfg.get('cds_col', DEFAULTS['cds_col']),
            label_col=label_col,
            join_col=DEFAULTS['join_col'],
            log_label=False,
            scale_label=False,
        )

        train_ds = DuetSingletaskDataset(data_path=train_path, **dataset_kwargs)
        val_ds   = DuetSingletaskDataset(data_path=val_path,   **dataset_kwargs)

        # Test: write scaled for model, keep raw for metrics
        test_scaled_path = os.path.join(tmp, 'test_scaled.tsv')
        write_scaled(test_df, test_scaled_path)
        test_ds = DuetSingletaskDataset(data_path=test_scaled_path, **dataset_kwargs)
        # DuetSingletaskDataset loads all data into memory in __init__, so tempdir can safely close after this
        test_y_raw = test_df[label_col].values

    train_loader = DataLoader(train_ds, batch_size=DEFAULTS['batch_size'],
                              shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2)

    loss_cfg = LossConfig(name='huber')
    model = MODEL_CLS(dropout=DEFAULTS['dropout'], loss_cfg=loss_cfg, **MODEL_PARAMS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULTS['lr'],
                                  weight_decay=DEFAULTS['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=DEFAULTS['lr_factor'],
        patience=DEFAULTS['lr_patience'], min_lr=1e-7)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_val_loss, patience_count, best_state = float('inf'), 0, None
    for epoch in range(DEFAULTS['max_epochs']):
        # Train
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

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast('cuda', dtype=torch.float16, enabled=USE_AMP):
                    _, loss = model(batch)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
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
                print(f"    early stop at epoch {epoch}{marker}", flush=True)
                break

        if epoch % 10 == 0 or marker:
            print(f"    epoch {epoch:3d} | val_loss={val_loss:.4f}{marker}", flush=True)

    # Test
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            yhat, _ = model.predict(batch)
            preds.append(yhat.cpu().numpy())
    pred_scaled = np.concatenate(preds)
    pred = inverse_fn(pred_scaled)

    out_df = {'y_true': test_y_raw, 'y_pred': pred}
    if DEFAULTS['join_col'] in test_df.columns:
        out_df = {DEFAULTS['join_col']: test_df[DEFAULTS['join_col']].values, **out_df}
    pd.DataFrame(out_df).to_csv(
        os.path.join(output_dir, 'predictions.tsv'), sep='\t', index=False)
    return compute_metrics(test_y_raw, pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--jobs-per-gpu', type=int, default=1)
    parser.add_argument('--names', nargs='+', default=None)
    parser.add_argument('--folds', nargs='+', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--worker-id', type=int, default=None)
    args = parser.parse_args()

    global MODEL_CLS, MODEL_PARAMS, USE_AMP
    mcfg = load_model_config('duet')
    _, MODEL_CLS = get_nn_classes(mcfg['model_class'])   # SingleModel
    MODEL_PARAMS = mcfg.get('model_params', {})
    USE_AMP      = amp_enabled(mcfg['precision'])
    DEFAULTS.update(mcfg['train'])                       # override train params, keep data fields
    single_dirname = mcfg['output_names']['single']
    print(f"[model=duet] class={mcfg['model_class']} params={MODEL_PARAMS} "
          f"precision={mcfg['precision']} out={single_dirname}", flush=True)

    cfg = load_config(args.config)
    output_root = cfg['output_dir']

    if args.gpus is not None:
        from benchlib import run_parallel
        gpus = [int(g) for g in args.gpus.split(',')]
        all_jobs = []
        for ds_name, ds_cfg in cfg['datasets'].items():
            allowed = ds_cfg.get('models')
            if allowed and 'DuET' not in allowed:
                continue
            for name in ds_cfg['names']:
                all_jobs.append({'name': f'{ds_name}/{name}', 'config': args.config})
        if args.names:
            all_jobs = [j for j in all_jobs if j['name'] in args.names]
        extra = []
        if args.folds:
            extra += ['--folds'] + [str(f) for f in args.folds]
        if args.force:
            extra += ['--force']
        run_parallel(__file__, all_jobs, gpus, args.jobs_per_gpu, extra_args=extra)
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for ds_name, ds_cfg in cfg['datasets'].items():
        allowed = ds_cfg.get('models')
        if allowed and 'DuET' not in allowed:
            continue

        for name in ds_cfg['names']:
            job_id = f'{ds_name}/{name}'
            if args.names and job_id not in args.names:
                continue

            print(f"\n{'='*50}")
            print(f"Dataset: {job_id}")
            print(f"{'='*50}")

            ds_output  = os.path.join(output_root, ds_name, name)
            splits_dir = os.path.join(ds_output, 'splits')

            if ds_cfg['split'] == 'rank':
                train_path, test_path = get_rank_paths(ds_cfg, name)
                n_folds = generate_rank_split(train_path, test_path, splits_dir,
                                              label_col=ds_cfg['label_col'],
                                              scaling=ds_cfg.get('scaling', 'none'))
            else:
                data_path = get_data_path(ds_cfg, name)
                n_folds = generate_splits(data_path, splits_dir, ds_cfg['k'], ds_cfg['seed'],
                                          label_col=ds_cfg['label_col'],
                                          scaling=ds_cfg.get('scaling', 'none'))

            all_metrics = []
            for fold_i in range(n_folds):
                if args.folds and fold_i not in args.folds:
                    continue
                fold_dir = os.path.join(splits_dir, f'fold_{fold_i}')
                indices  = load_indices(fold_dir)
                out      = os.path.join(ds_output, single_dirname, f'fold_{fold_i}')

                if not args.force:
                    cached = check_cached(out)
                    if cached:
                        all_metrics.append(cached)
                        print(f"  fold {fold_i}: cached")
                        continue

                if ds_cfg['split'] == 'rank':
                    with open(os.path.join(fold_dir, 'meta.json')) as f:
                        meta = json.load(f)
                    train_full = pd.read_csv(meta['train_file'], sep='\t')
                    train_df = train_full.iloc[indices['train']]
                    val_df   = train_full.iloc[indices['val']]
                    test_df  = pd.read_csv(meta['test_file'], sep='\t')
                else:
                    df       = pd.read_csv(data_path, sep='\t')
                    train_df = df.iloc[indices['train']]
                    val_df   = df.iloc[indices['val']]
                    test_df  = df.iloc[indices['test']]

                with Timer() as t:
                    metrics = run_fold(train_df, val_df, test_df, ds_cfg, out, fold_dir, device)
                metrics['elapsed_sec'] = t.elapsed
                with open(os.path.join(out, 'metrics.json'), 'w') as f:
                    json.dump(metrics, f, indent=2)
                all_metrics.append(metrics)
                print(f"  fold {fold_i}: pearson={metrics['pearson']:.4f} "
                      f"spearman={metrics['spearman']:.4f} [{t.elapsed:.0f}s]")

            if all_metrics:
                avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
                print(f"  AVG: pearson={avg['pearson']:.4f} spearman={avg['spearman']:.4f} "
                      f"r2={avg['r2']:.4f} rmse={avg['rmse']:.4f}")


if __name__ == '__main__':
    main()
