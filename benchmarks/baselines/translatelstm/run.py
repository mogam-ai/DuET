#!/usr/bin/env python3
"""TranslateLSTM benchmark — runs all folds for configured datasets.

Supports:
  - LstmModel (single, for MRL tasks without sequence features)
  - LstmDualModel (dual UTR+CDS, for TE tasks with sequence features)

Usage:
    conda activate benchmark_torch
    python baselines/translatelstm/run.py [--config config.yaml] --gpus 0,1,2,3
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
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from model import LstmModel, LstmDualModel

from benchlib import (check_cached, Timer, load_config, generate_splits, generate_rank_split,
                      load_indices, load_scaler, compute_metrics)
from benchlib.config import get_data_path, get_rank_paths


# Vocabulary
VOCAB = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4, 'U': 4}
PAD = 0


def seq_to_ids(seq: str, max_len: int) -> torch.Tensor:
    """Integer-encode and pad/truncate sequence."""
    ids = [VOCAB.get(c.upper(), PAD) for c in seq]
    if len(ids) > max_len:
        ids = ids[-max_len:]
    seq_len = len(ids)
    ids = ids + [PAD] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long), seq_len


def encode_col(df, col, max_len):
    """Encode a DataFrame column of sequences."""
    x_list, len_list = [], []
    for s in df[col].astype(str).values:
        x, sl = seq_to_ids(s, max_len)
        x_list.append(x)
        len_list.append(sl)
    return torch.stack(x_list), torch.tensor(len_list, dtype=torch.long)


def load_sequence_features(df, feature_path, feature_cols, join_col='txID'):
    """Load and merge sequence features, return normalized feature tensor."""
    feat_df = pd.read_csv(feature_path, sep='\t', usecols=[join_col] + feature_cols)
    merged = df[[join_col]].merge(feat_df, on=join_col, how='left')
    values = merged[feature_cols].fillna(0).values.astype(np.float32)
    return values


def run_fold_single(train_df, val_df, test_df, ds_cfg, output_dir, fold_dir, device):
    """Train single-encoder LSTM (MRL tasks, no sequence features)."""
    os.makedirs(output_dir, exist_ok=True)
    label_col = ds_cfg["label_col"]
    utr_col = ds_cfg.get("utr_col", "utr5")
    inp_len = ds_cfg.get("inp_len", 50)

    train_x, train_len = encode_col(train_df, utr_col, inp_len)
    val_x, val_len = encode_col(val_df, utr_col, inp_len)
    test_x, test_len = encode_col(test_df, utr_col, inp_len)

    scale_fn, inverse_fn = load_scaler(fold_dir)
    train_y = torch.tensor(scale_fn(train_df[label_col].values), dtype=torch.float32)
    val_y = torch.tensor(scale_fn(val_df[label_col].values), dtype=torch.float32)
    test_y_raw = test_df[label_col].values

    model = LstmModel(feature_size=5, sequence_feature_size=0, hidden_dim=64, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    batch_size = 512
    train_loader = DataLoader(TensorDataset(train_x, train_len, train_y),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_x, val_len, val_y),
                            batch_size=batch_size, shuffle=False)

    best_val_loss, patience, max_patience, best_state = float('inf'), 0, 10, None
    for epoch in range(50):
        model.train()
        train_losses = []
        for bx, blen, by in train_loader:
            bx, blen, by = bx.to(device), blen.to(device), by.to(device)
            batch = {'x': bx, 'seq_len': blen, 'y': by,
                     'sequence_feature': torch.zeros(bx.size(0), 0, device=device)}
            _, loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for bx, blen, by in val_loader:
                bx, blen, by = bx.to(device), blen.to(device), by.to(device)
                batch = {'x': bx, 'seq_len': blen, 'y': by,
                         'sequence_feature': torch.zeros(bx.size(0), 0, device=device)}
                _, loss = model(batch)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        marker = '*' if val_loss < best_val_loss else ''
        from scipy.stats import pearsonr, spearmanr
        vp = np.concatenate([model.predict({'x': bx.to(device), 'seq_len': blen.to(device),
             'sequence_feature': torch.zeros(bx.size(0), 0, device=device)})[0].cpu().numpy()
             for bx, blen, _ in val_loader])
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
    preds = []
    test_loader = DataLoader(TensorDataset(test_x, test_len), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for bx, blen in test_loader:
            bx, blen = bx.to(device), blen.to(device)
            batch = {'x': bx, 'seq_len': blen,
                     'sequence_feature': torch.zeros(bx.size(0), 0, device=device)}
            yhat, _ = model.predict(batch)
            preds.append(yhat.cpu().numpy())

    pred = inverse_fn(np.concatenate(preds))
    pd.DataFrame({"y_true": test_y_raw, "y_pred": pred}).to_csv(
        os.path.join(output_dir, "predictions.tsv"), sep='\t', index=False)
    return compute_metrics(test_y_raw, pred)


def onehot_encode_col(df, col, max_len, n_channels=4, side='left'):
    """One-hot encode sequences. N maps to [0,0,0,0].
    side='left': truncate from left, pad right (UTR)
    side='right': truncate from right, pad right (CDS)
    """
    BASE_MAP = {'A': 0, 'G': 1, 'C': 2, 'T': 3, 'U': 3}
    results = []
    lengths = []
    for s in df[col].astype(str).values:
        if side == 'left':
            s = s[-max_len:]  # keep rightmost (UTR)
        else:
            s = s[:max_len]  # keep leftmost (CDS)
        seq_len = len(s)
        s = s.ljust(max_len, 'N')  # always pad right
        tensor = torch.zeros(max_len, n_channels)
        for i, c in enumerate(s):
            idx = BASE_MAP.get(c.upper())
            if idx is not None:
                tensor[i, idx] = 1.0
        results.append(tensor)
        lengths.append(min(seq_len, max_len))
    return torch.stack(results), torch.tensor(lengths, dtype=torch.long)


def run_fold_dual(train_df, val_df, test_df, ds_cfg, model_cfg, output_dir, fold_dir, device):
    """Train dual-encoder LSTM (TE tasks, with sequence features)."""
    os.makedirs(output_dir, exist_ok=True)
    label_col = ds_cfg["label_col"]
    utr_col = ds_cfg.get("utr_col", "utr5")
    cds_col = model_cfg.get("cds_col", "cds")
    utr_len_max = ds_cfg.get("inp_len", 100)
    cds_len_max = model_cfg.get("cds_inp_len", 100)
    feature_path = model_cfg.get("sequence_feature_path")
    feature_cols = model_cfg.get("sequence_feature_cols", [])
    join_col = model_cfg.get("join_col", "txID")
    log_label = model_cfg.get("log_label", True)

    # Encode sequences (one-hot, N=[0,0,0,0])
    train_utr, train_utr_len = onehot_encode_col(train_df, utr_col, utr_len_max, side='left')
    val_utr, val_utr_len = onehot_encode_col(val_df, utr_col, utr_len_max, side='left')
    test_utr, test_utr_len = onehot_encode_col(test_df, utr_col, utr_len_max, side='left')

    train_cds, train_cds_len = onehot_encode_col(train_df, cds_col, cds_len_max, side='right')
    val_cds, val_cds_len = onehot_encode_col(val_df, cds_col, cds_len_max, side='right')
    test_cds, test_cds_len = onehot_encode_col(test_df, cds_col, cds_len_max, side='right')

    # Sequence features
    n_features = len(feature_cols)
    if n_features > 0 and feature_path:
        train_feat = load_sequence_features(train_df, feature_path, feature_cols, join_col)
        val_feat = load_sequence_features(val_df, feature_path, feature_cols, join_col)
        test_feat = load_sequence_features(test_df, feature_path, feature_cols, join_col)
        # Normalize features (fit on train)
        feat_scaler = StandardScaler()
        train_feat = feat_scaler.fit_transform(train_feat)
        val_feat = feat_scaler.transform(val_feat)
        test_feat = feat_scaler.transform(test_feat)
    else:
        n_features = 0
        train_feat = np.zeros((len(train_df), 0), dtype=np.float32)
        val_feat = np.zeros((len(val_df), 0), dtype=np.float32)
        test_feat = np.zeros((len(test_df), 0), dtype=np.float32)

    train_feat_t = torch.tensor(train_feat, dtype=torch.float32)
    val_feat_t = torch.tensor(val_feat, dtype=torch.float32)
    test_feat_t = torch.tensor(test_feat, dtype=torch.float32)

    # Labels
    scale_fn, inverse_fn = load_scaler(fold_dir)
    raw_train_y = train_df[label_col].values.astype(np.float64)
    raw_val_y = val_df[label_col].values.astype(np.float64)
    test_y_raw = test_df[label_col].values.astype(np.float64)

    if log_label:
        raw_train_y = np.log1p(raw_train_y)
        raw_val_y = np.log1p(raw_val_y)
        test_y_raw_for_eval = np.log1p(test_y_raw)
    else:
        test_y_raw_for_eval = test_y_raw

    train_y = torch.tensor(scale_fn(raw_train_y), dtype=torch.float32)
    val_y = torch.tensor(scale_fn(raw_val_y), dtype=torch.float32)

    # Model
    model = LstmDualModel(
        feature_size=4, sequence_feature_size=n_features,
        hidden_dim=64, utr_dropout=0.8, cds_dropout=0.2
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    batch_size = 256
    train_loader = DataLoader(
        TensorDataset(train_utr, train_utr_len, train_cds, train_cds_len, train_feat_t, train_y),
        batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        TensorDataset(val_utr, val_utr_len, val_cds, val_cds_len, val_feat_t, val_y),
        batch_size=batch_size, shuffle=False)

    best_val_loss, patience, max_patience, best_state = float('inf'), 0, 20, None
    for epoch in range(250):
        model.train()
        train_losses = []
        for b_utr, b_utr_len, b_cds, b_cds_len, b_feat, b_y in train_loader:
            batch = {
                'utr': b_utr.to(device), 'utr_len': b_utr_len.to(device),
                'cds': b_cds.to(device), 'cds_len': b_cds_len.to(device),
                'sequence_feature': b_feat.to(device), 'y': b_y.to(device),
            }
            _, loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for b_utr, b_utr_len, b_cds, b_cds_len, b_feat, b_y in val_loader:
                batch = {
                    'utr': b_utr.to(device), 'utr_len': b_utr_len.to(device),
                    'cds': b_cds.to(device), 'cds_len': b_cds_len.to(device),
                    'sequence_feature': b_feat.to(device), 'y': b_y.to(device),
                }
                _, loss = model(batch)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        marker = '*' if val_loss < best_val_loss else ''
        if epoch % 10 == 0 or val_loss < best_val_loss:
            from scipy.stats import pearsonr, spearmanr
            vp = np.concatenate([model.predict({
                'utr': b_utr.to(device), 'utr_len': b_utr_len.to(device),
                'cds': b_cds.to(device), 'cds_len': b_cds_len.to(device),
                'sequence_feature': b_feat.to(device)})[0].cpu().numpy()
                for b_utr, b_utr_len, b_cds, b_cds_len, b_feat, b_y in val_loader])
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
    test_loader = DataLoader(
        TensorDataset(test_utr, test_utr_len, test_cds, test_cds_len, test_feat_t),
        batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for b_utr, b_utr_len, b_cds, b_cds_len, b_feat in test_loader:
            batch = {
                'utr': b_utr.to(device), 'utr_len': b_utr_len.to(device),
                'cds': b_cds.to(device), 'cds_len': b_cds_len.to(device),
                'sequence_feature': b_feat.to(device),
            }
            yhat, _ = model.predict(batch)
            preds.append(yhat.cpu().numpy())

    pred_scaled = np.concatenate(preds)
    pred = inverse_fn(pred_scaled)
    if log_label:
        pred = np.expm1(pred)

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
            if allowed_models and "translatelstm" not in allowed_models:
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

    model_cfg = cfg.get("models", {}).get("translatelstm", {})

    for ds_name, ds_cfg in cfg["datasets"].items():
        # Skip datasets restricted to other models
        allowed_models = ds_cfg.get("models")
        if allowed_models and "translatelstm" not in allowed_models:
            continue
        is_dual = model_cfg.get("cds_col") is not None and ds_name == "celltype_te"

        for name in ds_cfg["names"]:
            job_id = f"{ds_name}/{name}"
            if args.names and job_id not in args.names:
                continue

            print(f"\n{'='*50}")
            print(f"Dataset: {job_id} ({'dual' if is_dual else 'single'})")
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
                out = os.path.join(ds_output, "translatelstm", f"fold_{fold_i}")

                if not args.force:
                    cached = check_cached(out)
                    if cached:
                        all_metrics.append(cached)
                        print(f"  fold {fold_i}: cached")
                        continue

                # Load and slice data
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
                    if is_dual:
                        metrics = run_fold_dual(train_df, val_df, test_df, ds_cfg, model_cfg, out, fold_dir, device)
                    else:
                        metrics = run_fold_single(train_df, val_df, test_df, ds_cfg, out, fold_dir, device)
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
