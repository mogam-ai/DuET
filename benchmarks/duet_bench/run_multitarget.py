#!/usr/bin/env python3
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""DuET multitarget pretrain benchmark runner.

Trains the DuET multi-target model on multitask_TE.tsv (76 celltypes simultaneously).
Uses the same outer fold splits as celltype_te baselines for fair finetune comparison.
Saves best checkpoint per fold for downstream finetuning.

Usage:
    conda activate benchmark_torch
    cd benchmarks
    python duet/run_multitarget.py --config config.yaml --gpu 0
    python duet/run_multitarget.py --config config.yaml --gpus 0,1 --jobs-per-gpu 1
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


from duet.models.loss import build_loss
from duet.configs.config import LossConfig
from duet.data.utils import tensorize, CODON_CODES
from functools import partial

from benchlib import check_cached, Timer, load_config, generate_splits, load_indices, compute_metrics
from benchlib.config import get_data_path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # benchmarks/
from duet_bench.model_config import load_model_config, get_nn_classes, amp_enabled

# Set in main() from model_config yaml.
MODEL_CLS = None            # MultiModel class
MODEL_PARAMS = {}           # constructor kwargs (cnn_filters, gru_hidden_dim, ...)
USE_AMP = False             # 16-mixed precision flag
DEFAULTS = {}               # train params loaded from yaml 'train' section

torch.set_float32_matmul_precision('high')


class MultiTargetDataset(Dataset):
    """In-memory multitarget dataset for the DuET multi-target model."""

    _tensorize_utr = staticmethod(partial(tensorize, channel_size=4))
    _tensorize_cds = staticmethod(partial(tensorize, channel_size=65,
                                          code=CODON_CODES, default=64,
                                          quantizer=lambda x: [x[k:k+3] for k in range(0, len(x), 3)]))

    def __init__(self, df: pd.DataFrame, te_cols: list,
                 utr_seq_size: int, cds_seq_size: int, pretensorize: bool = False,
                 tag: str = ''):
        self.te_cols = te_cols
        self.pretensorize = pretensorize

        # Pre-process sequences (padding same as DuetDataset)
        utrs = df['utr5'].str.upper().str[-utr_seq_size:].str.pad(utr_seq_size, 'left', 'N')
        cdss = df['cds'].str.upper().str[:cds_seq_size].str.pad(cds_seq_size, 'right', 'N')

        self.utrs = utrs.tolist()
        self.cdss = cdss.tolist()
        self.labels = df[te_cols].values.astype(np.float32)  # (N, n_targets)

        # Optionally one-hot encode all sequences up front. This makes __getitem__
        # a pure indexing op so DataLoader can use num_workers=0 without GPU starvation
        # (and without fork+CUDA worker deadlocks when many processes run concurrently).
        if pretensorize:
            import time
            n = len(self.utrs)
            t0 = time.time()
            print(f"    [pretensorize] {tag} encoding {n} samples...", flush=True)
            self.utr_tensors = [self._tensorize_utr(seq=u) for u in self.utrs]
            self.cds_tensors = [self._tensorize_cds(seq=c) for c in self.cdss]
            self.utrs = self.cdss = None  # free raw strings
            print(f"    [pretensorize] {tag} done: {n} samples in {time.time()-t0:.1f}s", flush=True)

    def __len__(self):
        return len(self.utr_tensors) if self.pretensorize else len(self.utrs)

    def __getitem__(self, i):
        if self.pretensorize:
            utr = self.utr_tensors[i]
            cds = self.cds_tensors[i]
        else:
            utr = self._tensorize_utr(seq=self.utrs[i])
            cds = self._tensorize_cds(seq=self.cdss[i])
        y   = torch.tensor(self.labels[i], dtype=torch.float32)
        return {
            'utr': utr, 'cds': cds,
            'start': torch.tensor([], dtype=torch.float32),
            'sequence_feature': torch.tensor([], dtype=torch.float32),
            'y': y,
        }


INNER_FOLDS = 9
INNER_SPLIT_FILE = 'duet_multi_inner_indices.json'
TOP_K = 5
INNER_SPLIT_SEED = 42


def _build_inner_splits(n, n_splits, seed):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [{'train': tr.tolist(), 'val': va.tolist()}
            for tr, va in kf.split(np.arange(n))]


def _train_one_inner(train_val_ds, inner_train_idx, inner_val_idx,
                     test_ds, n_targets, inner_dir, device,
                     test_txid=None, test_te_cols=None):
    """Train one inner model, return (val_loss, test_pred_array).

    If test_txid/test_te_cols are given, always save per-inner test predictions to
    inner_dir/test_pred.npz so downstream analysis needs no GPU re-prediction.
    """
    os.makedirs(inner_dir, exist_ok=True)
    ckpt_path = os.path.join(inner_dir, 'best.pt')
    npz_path = os.path.join(inner_dir, 'test_pred.npz')

    # If both ckpt and npz exist, return the cached prediction without retraining.
    if os.path.exists(ckpt_path) and os.path.exists(npz_path) and not getattr(_train_one_inner, '_force', False):
        d = np.load(npz_path, allow_pickle=True)
        return float(d['val_loss']), d['pred']

    # Resume if checkpoint exists
    if os.path.exists(ckpt_path):
        print(f"      [cached] {inner_dir}", flush=True)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        val_loss = ckpt['val_loss']
    else:
        from torch.utils.data import Subset
        tr_ds  = Subset(train_val_ds, inner_train_idx)
        va_ds  = Subset(train_val_ds, inner_val_idx)

        tr_loader = DataLoader(tr_ds, batch_size=DEFAULTS['batch_size'],
                               shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=256, shuffle=False, num_workers=4)

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
            for batch in tr_loader:
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
                for batch in va_loader:
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
                patience_count = 0; marker = ' *'
            else:
                patience_count += 1
                if patience_count >= DEFAULTS['patience']:
                    print(f"      early stop at epoch {epoch}", flush=True); break
            if epoch % 10 == 0 or marker:
                print(f"      epoch {epoch:3d} | val_loss={val_loss:.4f}{marker}", flush=True)

        val_loss = best_val_loss
        torch.save({'state_dict': best_state, 'n_targets': n_targets,
                    'te_cols': None, 'val_loss': val_loss}, ckpt_path)

    # Predict on test set
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    loss_cfg = LossConfig(name='huber')
    model = MODEL_CLS(n_targets=n_targets, dropout=DEFAULTS['dropout'],
                              loss_cfg=loss_cfg, **MODEL_PARAMS).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    te_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    preds = []
    with torch.no_grad():
        for batch in te_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            yhat, _ = model.predict(batch)
            preds.append(yhat.cpu().numpy())
    test_pred = np.concatenate(preds)  # (N, n_targets)

    # Always save per-inner test predictions so downstream collect/compare needs no re-prediction.
    # val_r2: mean per-target R2 on the inner val split. txID/te_cols included.
    if test_txid is not None:
        from sklearn.metrics import r2_score
        from torch.utils.data import Subset
        va_ds = Subset(train_val_ds, inner_val_idx)
        va_loader = DataLoader(va_ds, batch_size=256, shuffle=False, num_workers=4)
        vp, vt = [], []
        with torch.no_grad():
            for batch in va_loader:
                y = batch['y'].numpy()
                batch = {k: v.to(device) for k, v in batch.items()}
                yhat, _ = model.predict(batch)
                vp.append(yhat.cpu().numpy()); vt.append(y)
        vp = np.concatenate(vp); vt = np.concatenate(vt)
        if vt.ndim == 1:
            vt, vp = vt[:, None], vp[:, None]
        r2s = [r2_score(vt[~np.isnan(vt[:, c]), c], vp[~np.isnan(vt[:, c]), c])
               for c in range(vt.shape[1]) if (~np.isnan(vt[:, c])).sum() >= 10]
        val_r2 = float(np.mean(r2s)) if r2s else float('nan')
        np.savez(os.path.join(inner_dir, 'test_pred.npz'),
                 pred=test_pred, txid=np.array(test_txid),
                 te_cols=np.array(test_te_cols if test_te_cols is not None else []),
                 val_r2=val_r2, val_loss=val_loss)

    return val_loss, test_pred  # (N, n_targets)


def run_fold(train_df, val_df, test_df, te_cols, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    n_targets = len(te_cols)

    # Combine train+val for inner CV (same as RiboNN)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    train_val_ds = MultiTargetDataset(train_val_df, te_cols,
                                      DEFAULTS['utr_seq_size'], DEFAULTS['cds_seq_size'])
    test_ds = MultiTargetDataset(test_df, te_cols,
                                 DEFAULTS['utr_seq_size'], DEFAULTS['cds_seq_size'])

    # Build / load inner splits
    split_path = os.path.join(output_dir, INNER_SPLIT_FILE)
    if os.path.exists(split_path):
        with open(split_path) as f:
            inner_splits = json.load(f)['folds']
        print(f"    inner splits: loaded {split_path}", flush=True)
    else:
        inner_splits = _build_inner_splits(len(train_val_ds), INNER_FOLDS, INNER_SPLIT_SEED)
        with open(split_path, 'w') as f:
            json.dump({'inner_folds': INNER_FOLDS, 'seed': INNER_SPLIT_SEED,
                       'n': len(train_val_ds), 'folds': inner_splits}, f, indent=2)
        print(f"    inner splits: saved {split_path}", flush=True)

    # Train inner models
    all_preds = []  # [(val_loss, inner_idx, pred_array)]
    for i, split in enumerate(inner_splits):
        print(f"    inner fold {i}/{INNER_FOLDS}", flush=True)
        inner_dir = os.path.join(output_dir, f'inner_{i}')
        val_loss, pred = _train_one_inner(
            train_val_ds, split['train'], split['val'],
            test_ds, n_targets, inner_dir, device,
            test_txid=test_df['txID'].values if 'txID' in test_df.columns else None,
            test_te_cols=te_cols)
        all_preds.append((val_loss, i, pred))
        print(f"      val_loss={val_loss:.4f}", flush=True)

    # Top-K ensemble by val_loss (lower = better)
    all_preds.sort(key=lambda x: x[0])
    top_preds = [p for _, _, p in all_preds[:TOP_K]]
    print(f"    ensemble: top {len(top_preds)}/{len(all_preds)}", flush=True)
    pred = np.mean(top_preds, axis=0)  # (N, n_targets)

    # Get true labels from test_ds
    true = np.stack([test_ds[i]['y'].numpy() for i in range(len(test_ds))])  # (N, n_targets)

    per_target = {}
    pearsons, spearmans, r2s = [], [], []
    for i, col in enumerate(te_cols):
        mask = ~np.isnan(true[:, i])
        if mask.sum() < 10:
            continue
        m = compute_metrics(true[mask, i], pred[mask, i])
        per_target[col] = m
        pearsons.append(m['pearson']); spearmans.append(m['spearman']); r2s.append(m['r2'])

    metrics = {
        'pearson':    float(np.mean(pearsons)),
        'spearman':   float(np.mean(spearmans)),
        'r2':         float(np.mean(r2s)),
        'rmse':       0.0,
        'per_target': per_target,
    }

    # Save best checkpoint (lowest val_loss inner model) for downstream finetune
    best_inner = all_preds[0][1]  # all_preds sorted by val_loss asc; [0] = best
    best_ckpt_src = os.path.join(output_dir, f'inner_{best_inner}', 'best.pt')
    best_ckpt_dst = os.path.join(output_dir, 'best.pt')
    if os.path.exists(best_ckpt_src):
        import shutil
        shutil.copy2(best_ckpt_src, best_ckpt_dst)
        # Patch te_cols into best.pt
        ckpt = torch.load(best_ckpt_dst, map_location='cpu', weights_only=False)
        ckpt['te_cols'] = te_cols
        torch.save(ckpt, best_ckpt_dst)

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--jobs-per-gpu', type=int, default=1)
    parser.add_argument('--folds', nargs='+', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--predict-only', action='store_true',
                        help='Skip training (reuse ckpt); only regenerate inner test_pred.npz')
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
    model_dirname = mcfg['output_names']['multi']
    print(f"[model=duet] class={mcfg['model_class']} params={MODEL_PARAMS} "
          f"precision={mcfg['precision']} out={model_dirname}", flush=True)

    cfg = load_config(args.config)
    output_root = cfg['output_dir']

    # Use celltype_te_multi config - SAME data and splits as RiboNN multi/finetune
    mt_cfg     = cfg['datasets']['celltype_te_multi']
    mt_name    = mt_cfg['names'][0]                               # e.g. 'multitask'
    ds_output  = os.path.join(output_root, 'celltype_te_multi', mt_name)
    splits_dir = os.path.join(ds_output, 'splits')               # multitask/splits (shared w/ RiboNN)
    mt_data    = get_data_path(mt_cfg, mt_name)                  # .../multitask_TE.tsv

    if args.gpus is not None:
        from benchlib import run_parallel
        gpus = [int(g) for g in args.gpus.split(',')]
        all_jobs = [{'name': f'celltype_te_multi/{mt_name}', 'config': args.config}]
        extra = []
        if args.folds:
            extra += ['--folds'] + [str(f) for f in args.folds]
        if args.force:
            extra += ['--force']
        if args.predict_only:
            extra += ['--predict-only']
        run_parallel(__file__, all_jobs, gpus, args.jobs_per_gpu, extra_args=extra)
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load multitask data
    print(f"Loading {mt_data}", flush=True)
    df = pd.read_csv(mt_data, sep='\t')
    te_cols = [c for c in df.columns if c.startswith('TE_')]
    print(f"n_targets={len(te_cols)}, n_rows={len(df)}", flush=True)

    # Generate splits on multitask_TE.tsv itself (KFold seed=42 => identical to RiboNN's splits).
    # generate_splits is idempotent: if RiboNN already created them, indices match exactly.
    n_folds = generate_splits(mt_data, splits_dir, mt_cfg['k'], mt_cfg['seed'],
                              scaling=mt_cfg.get('scaling', 'none'),
                              skip_if_exists=True)

    all_metrics = []
    for fold_i in range(n_folds):
        if args.folds and fold_i not in args.folds:
            continue

        fold_dir = os.path.join(splits_dir, f'fold_{fold_i}')
        out      = os.path.join(ds_output, model_dirname, f'fold_{fold_i}')

        # --predict-only: run run_fold even if metrics.json exists
        # (training is skipped via the ckpt cache; only inner test_pred.npz is regenerated).
        if not args.predict_only and not args.force and os.path.exists(os.path.join(out, 'metrics.json')):
            with open(os.path.join(out, 'metrics.json')) as f:
                cached = json.load(f)
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
        all_metrics.append(metrics)
        print(f"  fold {fold_i}: pearson={metrics['pearson']:.4f} "
              f"spearman={metrics['spearman']:.4f} [{t.elapsed:.0f}s]")

    if all_metrics:
        avg_p = np.mean([m['pearson'] for m in all_metrics])
        avg_s = np.mean([m['spearman'] for m in all_metrics])
        print(f"\nAVG: pearson={avg_p:.4f} spearman={avg_s:.4f}")


if __name__ == '__main__':
    main()
