#!/usr/bin/env python3
"""GEMORNA 5'UTR prediction model benchmark (Zhang et al., Science 2025).

Re-trains the GEMORNA MRL prediction architecture (2-layer GRU + Linear)
on our benchmark splits for fair comparison.

Usage:
    conda activate benchmark_torch
    python baselines/GEMORNA/run.py [--config config.yaml] --gpus 0,1,2,3
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from model_pred5UTR import Model

from benchlib import (check_cached, Timer, load_config, generate_splits, generate_rank_split,
                      load_indices, load_scaler, compute_metrics)
from benchlib.config import get_data_path, get_rank_paths


# Nucleotide vocab matching GEMORNA (from src/shared/helper.py)
VOCAB = {'A': 5, 'U': 6, 'T': 6, 'G': 7, 'C': 8, 'N': 9}
PAD = 0  # [PAD]


class Args:
    """Model hyperparameters matching GEMORNA 5utr.pt checkpoint."""
    embed_num = 10
    embed_dim = 64
    kernel_num = 128
    dropout = 0.1


def encode_sequences(df, col, max_len):
    """Integer-encode sequences, truncate from left, pad right."""
    results = []
    for s in df[col].astype(str).values:
        s = s[-max_len:]  # keep rightmost (5'UTR)
        ids = [VOCAB.get(c.upper(), PAD) for c in s]
        ids = ids + [PAD] * (max_len - len(ids))  # pad right
        results.append(ids)
    return torch.tensor(results, dtype=torch.long)


def run_fold(train_df, val_df, test_df, ds_cfg, output_dir, fold_dir, device):
    """Train and evaluate one fold."""
    os.makedirs(output_dir, exist_ok=True)
    label_col = ds_cfg["label_col"]
    utr_col = ds_cfg.get("utr_col", "utr5")
    inp_len = ds_cfg.get("inp_len", 50)

    train_x = encode_sequences(train_df, utr_col, inp_len)
    val_x = encode_sequences(val_df, utr_col, inp_len)
    test_x = encode_sequences(test_df, utr_col, inp_len)

    scale_fn, inverse_fn = load_scaler(fold_dir)
    train_y = torch.tensor(scale_fn(train_df[label_col].values), dtype=torch.float32)
    val_y = torch.tensor(scale_fn(val_df[label_col].values), dtype=torch.float32)
    test_y_raw = test_df[label_col].values

    model = Model(Args()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    loss_fn = nn.MSELoss()

    batch_size = 512
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)

    best_val_loss, patience, max_patience, best_state = float('inf'), 0, 10, None
    for epoch in range(50):
        model.train()
        train_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            loss = loss_fn(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                loss = loss_fn(pred, by)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        marker = '*' if val_loss < best_val_loss else ''
        from scipy.stats import pearsonr, spearmanr
        with torch.no_grad():
            vp = np.concatenate([model(bx.to(device)).cpu().numpy() for bx, _ in val_loader])
        vy = val_y.numpy()
        pr, sr = pearsonr(vy, vp)[0], spearmanr(vy, vp)[0]
        print(f"    epoch {epoch:3d} | train_loss={np.mean(train_losses):.4f} val_loss={val_loss:.4f} pearson={pr:.4f} spearman={sr:.4f} {marker}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    test_loader = DataLoader(TensorDataset(test_x), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (bx,) in test_loader:
            preds.append(model(bx.to(device)).cpu().numpy())

    pred = inverse_fn(np.concatenate(preds))
    pd.DataFrame({"y_true": test_y_raw, "y_pred": pred}).to_csv(
        os.path.join(output_dir, "predictions.tsv"), sep='\t', index=False)
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

    cfg = load_config(args.config)
    output_root = cfg["output_dir"]

    # Dispatcher mode
    if args.gpus is not None:
        from benchlib import run_parallel
        gpus = [int(g) for g in args.gpus.split(',')]
        all_jobs = []
        for ds_name, ds_cfg in cfg["datasets"].items():
            allowed_models = ds_cfg.get("models")
            if allowed_models and "GEMORNA" not in allowed_models:
                continue
            # GEMORNA only supports MRL (5'UTR only, no CDS)
            if ds_name != "mrl_rank":
                continue
            for name in ds_cfg["names"]:
                all_jobs.append({"name": f"{ds_name}/{name}", "config": args.config})
        if args.names:
            all_jobs = [j for j in all_jobs if j["name"] in args.names]
        extra = []
        if args.folds:
            extra += ["--folds"] + [str(f) for f in args.folds]
        if args.force:
            extra += ["--force"]
        run_parallel(__file__, all_jobs, gpus, args.jobs_per_gpu, extra_args=extra)
        return

    # Worker mode
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for ds_name, ds_cfg in cfg["datasets"].items():
        # GEMORNA 5'UTR prediction: MRL only
        if ds_name != "mrl_rank":
            continue

        for name in ds_cfg["names"]:
            job_id = f"{ds_name}/{name}"
            if args.names and job_id not in args.names:
                continue

            print(f"\n{'='*50}")
            print(f"Dataset: {job_id}")
            print(f"{'='*50}")

            ds_output = os.path.join(output_root, ds_name, name)
            splits_dir = os.path.join(ds_output, "splits")

            if ds_cfg["split"] == "rank":
                train_path, test_path = get_rank_paths(ds_cfg, name)
                n_folds = generate_rank_split(train_path, test_path, splits_dir,
                                             label_col=ds_cfg["label_col"],
                                             scaling=ds_cfg.get("scaling", "none"))
            else:
                data_path = get_data_path(ds_cfg, name)
                n_folds = generate_splits(data_path, splits_dir, ds_cfg["k"], ds_cfg["seed"],
                                         label_col=ds_cfg["label_col"],
                                         scaling=ds_cfg.get("scaling", "none"))

            all_metrics = []
            for fold_i in range(n_folds):
                if args.folds and fold_i not in args.folds:
                    continue
                fold_dir = os.path.join(splits_dir, f"fold_{fold_i}")
                indices = load_indices(fold_dir)
                out = os.path.join(ds_output, "GEMORNA", f"fold_{fold_i}")

                if not args.force:
                    cached = check_cached(out)
                    if cached:
                        all_metrics.append(cached)
                        print(f"  fold {fold_i}: cached")
                        continue

                if ds_cfg["split"] == "rank":
                    with open(os.path.join(fold_dir, "meta.json")) as f:
                        meta = json.load(f)
                    train_full = pd.read_csv(meta["train_file"], sep='\t')
                    train_df = train_full.iloc[indices["train"]]
                    val_df = train_full.iloc[indices["val"]]
                    test_df = pd.read_csv(meta["test_file"], sep='\t')
                else:
                    df = pd.read_csv(data_path, sep='\t')
                    train_df = df.iloc[indices["train"]]
                    val_df = df.iloc[indices["val"]]
                    test_df = df.iloc[indices["test"]]

                with Timer() as t:
                    metrics = run_fold(train_df, val_df, test_df, ds_cfg, out, fold_dir, device)
                metrics["elapsed_sec"] = t.elapsed
                with open(os.path.join(out, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
                all_metrics.append(metrics)
                print(f"  fold {fold_i}: pearson={metrics['pearson']:.4f} spearman={metrics['spearman']:.4f} [{t.elapsed}s]")

            if all_metrics:
                avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
                print(f"  AVG: pearson={avg['pearson']:.4f} spearman={avg['spearman']:.4f} "
                      f"r2={avg['r2']:.4f} rmse={avg['rmse']:.4f}")


if __name__ == "__main__":
    main()
