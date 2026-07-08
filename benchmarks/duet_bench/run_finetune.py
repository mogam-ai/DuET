#!/usr/bin/env python3
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""DuET finetune benchmark runner.

Matches RiboNN finetune data flow EXACTLY so predictions can be ensembled:
  - Uses the SAME multitask data + multitask outer split as RiboNN
  - For each celltype, slices train_val/test by the multitask fold indices,
    then keeps only rows where TE_<celltype> is non-NaN
  - Fits StandardScaler on the filtered train_val labels
  - Inner 9-fold ensemble (top-5 by val_loss), same as RiboNN

Two-phase training per inner fold:
  Phase 1: freeze utr_branch + cds_branch, train head + BatchNorm
  Phase 2: unfreeze all, lower lr

Usage:
    conda activate benchmark_torch && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    cd benchmarks
    python duet/run_finetune.py --config config.yaml --gpu 0 \\
        --names celltype_te/hek293t_TE \\
        --pretrain-dir <output_root>/celltype_te_multi/multitask/DuET_multi
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from duet.configs.config import LossConfig
from benchlib import Timer, load_config, generate_splits, load_indices, compute_metrics
from benchlib.config import get_data_path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # benchmarks/
from duet_bench.run_multitarget import MultiTargetDataset
from duet_bench.model_config import load_model_config, get_nn_classes, amp_enabled

# Set in main() from model_config yaml.
MULTI_CLS, SINGLE_CLS = None, None
MODEL_PARAMS = {}
USE_AMP = False
DEFAULTS = {}               # train params (yaml 'train' section)
FT = {}                     # finetune params (yaml 'finetune' section)

torch.set_float32_matmul_precision('high')

# Limit per-process CPU threads. Many finetune workers run concurrently; without this
# each process spawns intra-op threads up to the core count (96), oversubscribing the
# host (load avg in the hundreds) and starving other users' jobs. Override via
# DUET_NUM_THREADS env if needed.
_NUM_THREADS = int(os.environ.get('DUET_NUM_THREADS', '4'))
torch.set_num_threads(_NUM_THREADS)

INNER_SPLIT_SEED = 42                     # SAME as RiboNN INNER_SPLIT_RANDOM_STATE
INNER_SPLIT_FILE = 'duet_ft_inner_indices.json'


def _build_inner_splits(n, n_splits, seed):
    """KFold(shuffle=True, random_state=seed) - identical scheme to RiboNN."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [{'train': tr.tolist(), 'val': va.tolist()}
            for tr, va in kf.split(np.arange(n))]


def build_finetune_model(pretrain_dir, fold_i, inner_fold, device):
    """Load inner-fold pretrained multi-target model, transfer encoder to single-target."""
    inner_ckpt = os.path.join(pretrain_dir, f'fold_{fold_i}', f'inner_{inner_fold}', 'best.pt')
    if not os.path.exists(inner_ckpt):
        inner_ckpt = os.path.join(pretrain_dir, f'fold_{fold_i}', 'best.pt')  # fallback
    ckpt = torch.load(inner_ckpt, map_location='cpu', weights_only=False)
    n_targets = ckpt['n_targets']

    loss_cfg = LossConfig(name='huber')
    multi = MULTI_CLS(n_targets=n_targets, dropout=DEFAULTS['dropout'], loss_cfg=loss_cfg, **MODEL_PARAMS)
    multi.load_state_dict(ckpt['state_dict'])

    single = SINGLE_CLS(dropout=DEFAULTS['dropout'], loss_cfg=loss_cfg, **MODEL_PARAMS)
    single.utr_branch.load_state_dict(multi.utr_branch.state_dict())
    single.cds_branch.load_state_dict(multi.cds_branch.state_dict())
    return single.to(device)


def _train_phases(model, train_loader, val_loader, device):
    """Two-phase finetune (head-only -> full). Returns (model, best_val_loss)."""
    def run_phase(model, optimizer, n_epochs, patience, phase_name):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=DEFAULTS['lr_factor'],
            patience=DEFAULTS['lr_patience'], min_lr=1e-7)
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
        best_val_loss, p_count, best_state = float('inf'), 0, None
        for epoch in range(n_epochs):
            model.train()
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                batch['y'] = batch['y'].squeeze(-1)  # (B,1) -> (B,)
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
                    batch['y'] = batch['y'].squeeze(-1)
                    with torch.autocast('cuda', dtype=torch.float16, enabled=USE_AMP):
                        _, loss = model(batch)
                    val_losses.append(loss.item())
            val_loss = float(np.mean(val_losses))
            scheduler.step(val_loss)

            marker = ''
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                p_count = 0; marker = ' *'
            else:
                p_count += 1
                if p_count >= patience:
                    print(f"        [{phase_name}] early stop at epoch {epoch}", flush=True)
                    break
            if epoch % 5 == 0 or marker:
                print(f"        [{phase_name}] epoch {epoch:3d} | val_loss={val_loss:.4f}{marker}", flush=True)
        model.load_state_dict(best_state)
        return model, best_val_loss

    model.utr_branch.requires_grad_(False)
    model.cds_branch.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.requires_grad_(True)
    opt1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=FT['lr_p1'], weight_decay=DEFAULTS['weight_decay'])
    model, _ = run_phase(model, opt1, FT['p1_epochs'], FT['patience'], 'P1')

    model.requires_grad_(True)
    opt2 = torch.optim.AdamW(model.parameters(), lr=FT['lr_p2'], weight_decay=DEFAULTS['weight_decay'])
    model, best_val = run_phase(model, opt2, FT['p2_epochs'], FT['patience'], 'P2')
    return model, best_val


def run_fold(mt_df, mt_indices, te_col, output_dir, fold_dir, pretrain_dir, fold_i, device,
             args_force=False):
    """Finetune one outer fold with inner 9-fold ensemble. Mirrors RiboNN run_finetune_fold."""
    os.makedirs(output_dir, exist_ok=True)

    # Slice multitask data by outer split, filter to non-NaN target (SAME as RiboNN)
    mt_train_val = mt_df.iloc[np.concatenate([mt_indices['train'], mt_indices['val']])]
    mt_test      = mt_df.iloc[mt_indices['test']]
    mt_train_val = mt_train_val[mt_train_val[te_col].notna()].reset_index(drop=True)
    mt_test      = mt_test[mt_test[te_col].notna()].reset_index(drop=True)

    # Leakage check
    trainval_txids = set(mt_train_val['txID'])
    test_txids     = set(mt_test['txID'])
    leakage = trainval_txids & test_txids
    print(f"    [{te_col}] train_val={len(mt_train_val)} test={len(mt_test)} | "
          f"leakage={len(leakage)}/{len(test_txids)}", flush=True)
    if leakage:
        print(f"    [WARNING] LEAKAGE DETECTED!", flush=True)

    test_y_raw = mt_test[te_col].values

    # Scale labels on filtered train_val (SAME as RiboNN)
    scaler = StandardScaler()
    scaler.fit(mt_train_val[te_col].values.reshape(-1, 1))
    scale_fn   = lambda x: scaler.transform(np.asarray(x).reshape(-1, 1)).flatten()
    inverse_fn = lambda x: scaler.inverse_transform(np.asarray(x).reshape(-1, 1)).flatten()

    def to_dataset(df, tag):
        d = df[['utr5', 'cds']].copy()
        d['__label__'] = scale_fn(df[te_col].values)
        return MultiTargetDataset(d, ['__label__'],
                                  DEFAULTS['utr_seq_size'], DEFAULTS['cds_seq_size'],
                                  pretensorize=True, tag=tag)

    train_val_ds = to_dataset(mt_train_val, f"{te_col} fold{fold_i} train_val")
    test_ds      = to_dataset(mt_test, f"{te_col} fold{fold_i} test")

    # Inner splits (KFold seed=42, same scheme as RiboNN)
    split_path = os.path.join(output_dir, INNER_SPLIT_FILE)
    if os.path.exists(split_path):
        with open(split_path) as f:
            inner_splits = json.load(f)['folds']
    else:
        inner_splits = _build_inner_splits(len(train_val_ds), FT['inner_folds'], INNER_SPLIT_SEED)
        with open(split_path, 'w') as f:
            json.dump({'inner_folds': FT['inner_folds'], 'seed': INNER_SPLIT_SEED,
                       'n': len(train_val_ds), 'folds': inner_splits}, f, indent=2)

    all_preds = []  # [(val_r2, val_loss, pred_scaled)]
    inner_timings = []
    for i, split in enumerate(inner_splits):
        print(f"    inner fold {i}/{FT['inner_folds']}", flush=True)
        inner_dir = os.path.join(output_dir, f'inner_{i}')
        os.makedirs(inner_dir, exist_ok=True)
        ckpt_path = os.path.join(inner_dir, 'best.pt')
        npz_path = os.path.join(inner_dir, 'test_pred.npz')

        # If the npz cache exists, reuse it without retraining/re-predicting (for downstream collect).
        if os.path.exists(npz_path) and not args_force:
            d = np.load(npz_path, allow_pickle=True)
            all_preds.append((float(d['val_r2']), float(d['val_loss']), d['pred_scaled']))
            print(f"      [npz cached] val_r2={float(d['val_r2']):.4f}", flush=True)
            continue

        tr_loader = DataLoader(Subset(train_val_ds, split['train']),
                               batch_size=DEFAULTS['batch_size'], shuffle=True,
                               drop_last=True, num_workers=0, pin_memory=True)
        va_loader = DataLoader(Subset(train_val_ds, split['val']),
                               batch_size=256, shuffle=False, num_workers=0)
        te_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

        # If a ckpt exists, skip training and load it (predict-only re-prediction).
        if os.path.exists(ckpt_path):
            model = build_finetune_model(pretrain_dir, fold_i, i, device)
            sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(sd['state_dict']); model.to(device)
            val_loss = sd.get('val_loss', float('nan'))
            t_train_elapsed = 0.0
        else:
            model = build_finetune_model(pretrain_dir, fold_i, i, device)
            with Timer() as t_train:
                model, val_loss = _train_phases(model, tr_loader, va_loader, device)
            t_train_elapsed = t_train.elapsed

        model.eval()
        preds = []
        with Timer() as t_pred:
            with torch.no_grad():
                for batch in te_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    yhat, _ = model.predict(batch)
                    preds.append(yhat.cpu().numpy())
        test_pred = np.concatenate(preds).flatten()

        # Save inner checkpoint for later re-evaluation
        if not os.path.exists(ckpt_path):
            torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                        'val_loss': val_loss}, ckpt_path)

        # Val R2 on inner validation split (NaN-masked)
        val_preds = []
        with torch.no_grad():
            for batch in va_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                yhat, _ = model.predict(batch)
                val_preds.append(yhat.cpu().numpy())
        val_pred_arr = np.concatenate(val_preds).flatten()
        val_true_arr = np.array([train_val_ds[j]['y'].item() for j in split['val']])
        vmask = ~(np.isnan(val_true_arr) | np.isnan(val_pred_arr))
        if vmask.sum() > 1:
            val_r2 = float(r2_score(val_true_arr[vmask], val_pred_arr[vmask]))
        else:
            val_r2 = float('nan')

        # Save per-inner test predictions (raw-scale + txID) so downstream collect needs no re-prediction.
        np.savez(npz_path,
                 pred=inverse_fn(test_pred), pred_scaled=test_pred,
                 txid=mt_test['txID'].values, val_r2=val_r2, val_loss=val_loss)

        all_preds.append((val_r2, val_loss, test_pred))
        inner_timings.append({'inner': i, 'train_sec': t_train_elapsed,
                              'pred_sec': t_pred.elapsed, 'val_loss': val_loss,
                              'val_r2': val_r2})
        print(f"      val_loss={val_loss:.4f} val_r2={val_r2:.4f} train={t_train_elapsed:.0f}s pred={t_pred.elapsed:.1f}s",
              flush=True)

    # Top-K ensemble by val_r2 (higher = better)
    all_preds.sort(key=lambda x: x[0], reverse=True)
    top_preds = [p for _, _, p in all_preds[:FT['top_k']]]
    print(f"    ensemble: top {len(top_preds)}/{len(all_preds)} by val_r2", flush=True)
    pred = inverse_fn(np.mean(top_preds, axis=0))

    out_df = {'txID': mt_test['txID'].values, 'y_true': test_y_raw, 'y_pred': pred}
    pd.DataFrame(out_df).to_csv(
        os.path.join(output_dir, 'predictions.tsv'), sep='\t', index=False)
    metrics = compute_metrics(test_y_raw, pred)
    metrics['inner_timings'] = inner_timings
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--jobs-per-gpu', type=int, default=1)
    parser.add_argument('--names', nargs='+', default=None)
    parser.add_argument('--folds', nargs='+', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--predict-only', action='store_true',
                        help='Skip training (reuse ckpt); only regenerate inner test_pred.npz')
    parser.add_argument('--worker-id', type=int, default=None)
    parser.add_argument('--pretrain-dir', required=True,
                        help='Dir with fold_N/inner_M/best.pt from run_multitarget.py')
    args = parser.parse_args()

    global MULTI_CLS, SINGLE_CLS, MODEL_PARAMS, USE_AMP, DEFAULTS, FT
    mcfg = load_model_config('duet')
    MULTI_CLS, SINGLE_CLS = get_nn_classes(mcfg['model_class'])
    MODEL_PARAMS = mcfg.get('model_params', {})
    USE_AMP      = amp_enabled(mcfg['precision'])
    DEFAULTS     = mcfg['train']
    FT           = mcfg['finetune']
    ft_dirname   = mcfg['output_names']['finetune']
    print(f"[model=duet] class={mcfg['model_class']} params={MODEL_PARAMS} "
          f"precision={mcfg['precision']} out={ft_dirname}", flush=True)

    cfg = load_config(args.config)
    output_root = cfg['output_root'] if 'output_root' in cfg else cfg['output_dir']

    ct_cfg = cfg['datasets']['celltype_te']
    mt_cfg = cfg['datasets']['celltype_te_multi']
    mt_name = mt_cfg['names'][0]
    mt_data = get_data_path(mt_cfg, mt_name)
    mt_splits_dir = os.path.join(output_root, 'celltype_te_multi', mt_name, 'splits')

    if args.gpus is not None:
        from benchlib import run_parallel
        gpus = [int(g) for g in args.gpus.split(',')]
        all_jobs = [{'name': f'celltype_te/{n}', 'config': args.config}
                    for n in ct_cfg['names']]
        if args.names:
            all_jobs = [j for j in all_jobs if j['name'] in args.names]
        extra = ['--pretrain-dir', args.pretrain_dir]
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

    # Load multitask data + ensure multitask outer splits exist (shared w/ RiboNN)
    mt_df = pd.read_csv(mt_data, sep='\t')
    n_folds = generate_splits(mt_data, mt_splits_dir, mt_cfg['k'], mt_cfg['seed'],
                              scaling=mt_cfg.get('scaling', 'none'),
                              skip_if_exists=True)

    for name in ct_cfg['names']:
        job_id = f'celltype_te/{name}'
        if args.names and job_id not in args.names:
            continue

        te_col = 'TE_' + name.replace('_TE', '')
        if te_col not in mt_df.columns:
            print(f"  {name}: {te_col} not in multitask data, skip")
            continue

        print(f"\n{'='*50}\nDataset: {job_id}  (target={te_col})\n{'='*50}")
        ds_output = os.path.join(output_root, 'celltype_te', name)

        all_metrics = []
        for fold_i in range(n_folds):
            if args.folds and fold_i not in args.folds:
                continue

            out = os.path.join(ds_output, ft_dirname, f'fold_{fold_i}')
            if not args.predict_only and not args.force and os.path.exists(os.path.join(out, 'metrics.json')):
                with open(os.path.join(out, 'metrics.json')) as f:
                    cached = json.load(f)
                all_metrics.append(cached)
                print(f"  fold {fold_i}: cached  pearson={cached['pearson']:.4f}")
                continue

            mt_fold_dir = os.path.join(mt_splits_dir, f'fold_{fold_i}')
            mt_indices  = load_indices(mt_fold_dir)

            print(f"  --- fold {fold_i} ---")
            with Timer() as t:
                metrics = run_fold(mt_df, mt_indices, te_col, out, mt_fold_dir,
                                   args.pretrain_dir, fold_i, device, args_force=args.force)
            metrics['elapsed_sec'] = t.elapsed
            # patch metrics.json to include outer elapsed
            with open(os.path.join(out, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            all_metrics.append(metrics)
            print(f"  fold {fold_i}: pearson={metrics['pearson']:.4f} "
                  f"spearman={metrics['spearman']:.4f} [{t.elapsed:.0f}s]")

        if all_metrics:
            avg = {k: np.mean([m[k] for m in all_metrics])
                   for k in ['pearson', 'spearman', 'r2', 'rmse']}
            print(f"  AVG: pearson={avg['pearson']:.4f} spearman={avg['spearman']:.4f} "
                  f"r2={avg['r2']:.4f} rmse={avg['rmse']:.4f}")


if __name__ == '__main__':
    main()
