#!/usr/bin/env python3
"""Optimus5p benchmark — runs all folds for configured datasets.

Usage:
    conda activate benchmark_tf
    python baselines/optimus5p/run.py [--config config.yaml]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from benchlib import (check_cached, Timer, load_config, generate_splits, generate_rank_split,
                      load_indices, load_scaler, compute_metrics)
from benchlib.config import get_data_path, get_rank_paths


def _setup_tf():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def run_fold(train_df, val_df, test_df, ds_cfg, output_dir, fold_dir):
    """Train and evaluate one fold. Returns metrics dict."""
    from model import build_model, one_hot_encode
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    import keras

    os.makedirs(output_dir, exist_ok=True)
    label_col = ds_cfg["label_col"]
    utr_col = ds_cfg.get("utr_col", "utr5")
    inp_len = ds_cfg.get("inp_len", 50)

    # Encode
    train_x = one_hot_encode(train_df, col=utr_col, seq_len=inp_len)
    val_x = one_hot_encode(val_df, col=utr_col, seq_len=inp_len)
    test_x = one_hot_encode(test_df, col=utr_col, seq_len=inp_len)

    # Scale (loaded from shared scaler)
    scale_fn, inverse_fn = load_scaler(fold_dir)
    train_y = scale_fn(train_df[label_col].values).reshape(-1, 1)
    val_y = scale_fn(val_df[label_col].values).reshape(-1, 1)
    test_y_raw = test_df[label_col].values

    # Train
    model = build_model(inp_len=inp_len)
    ckpt_path = os.path.join(output_dir, "best_model.h5")
    model.fit(train_x, train_y, batch_size=128,
              validation_data=(val_x, val_y),
              epochs=50, verbose=0,
              callbacks=[
                  ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
                  EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
              ])

    # Predict
    model = keras.models.load_model(ckpt_path)
    pred = inverse_fn(model.predict(test_x).flatten())

    # Save & compute metrics
    pd.DataFrame({"y_true": test_y_raw, "y_pred": pred}).to_csv(
        os.path.join(output_dir, "predictions.tsv"), sep='\t', index=False)
    metrics = compute_metrics(test_y_raw, pred)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpus', type=str, default=None, help='Comma-separated GPU IDs for parallel dispatch')
    parser.add_argument('--jobs-per-gpu', type=int, default=1)
    parser.add_argument('--names', nargs='+', default=None, help='Run only these dataset names')
    parser.add_argument('--folds', nargs='+', type=int, default=None, help='Run only these fold indices')
    parser.add_argument('--force', action='store_true', help='Ignore cache')
    parser.add_argument('--worker-id', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = cfg["output_dir"]

    # Dispatcher mode: split work across GPUs
    if args.gpus is not None:
        from benchlib import run_parallel
        gpus = [int(g) for g in args.gpus.split(',')]
        # Collect all (ds_name, name) pairs
        all_jobs = []
        for ds_name, ds_cfg in cfg["datasets"].items():
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
    _setup_tf()
    import keras  # import after CUDA_VISIBLE_DEVICES is set

    for ds_name, ds_cfg in cfg["datasets"].items():
        if "models" in ds_cfg and "optimus5p" not in ds_cfg["models"]:
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

            # Generate splits (with scaler)
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
                out = os.path.join(ds_output, "optimus5p", f"fold_{fold_i}")

                # Check cache
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
                    metrics = run_fold(train_df, val_df, test_df, ds_cfg, out, fold_dir)
                metrics["elapsed_sec"] = t.elapsed
                with open(os.path.join(out, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
                all_metrics.append(metrics)
                print(f"  fold {fold_i}: pearson={metrics['pearson']:.4f} spearman={metrics['spearman']:.4f}")

            # Summary
            if all_metrics:
                avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
                print(f"  AVG: pearson={avg['pearson']:.4f} spearman={avg['spearman']:.4f} "
                      f"r2={avg['r2']:.4f} rmse={avg['rmse']:.4f}")


if __name__ == "__main__":
    main()
