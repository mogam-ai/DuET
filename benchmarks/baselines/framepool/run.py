#!/usr/bin/env python3
"""Framepool benchmark — runs all folds for configured datasets.

Usage:
    conda activate benchmark_tf
    python baselines/framepool/run.py [--config config.yaml] [--gpu 0]
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
    import tensorflow as tf
    import keras
    from keras.models import load_model
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    import model as framepool_model
    import utils_data

    os.makedirs(output_dir, exist_ok=True)
    label_col = ds_cfg["label_col"]
    utr_col = ds_cfg.get("utr_col", "utr5")

    # Encoding functions
    one_hot_fn = utils_data.OneHotEncoder(utr_col, min_len=ds_cfg.get("inp_len", 100))
    out_encoding_fn = utils_data.DataFrameExtractor(label_col, method="direct")

    # Scale labels
    scale_fn, inverse_fn = load_scaler(fold_dir)
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df[label_col] = scale_fn(train_df[label_col].values)
    val_df[label_col] = scale_fn(val_df[label_col].values)
    test_y_raw = test_df[label_col].values

    # Data generators
    train_gen = utils_data.DataSequence(train_df,
                                        encoding_functions=[one_hot_fn],
                                        output_encoding_fn=out_encoding_fn,
                                        shuffle=True)
    val_gen = utils_data.DataSequence(val_df,
                                      encoding_functions=[one_hot_fn],
                                      output_encoding_fn=out_encoding_fn,
                                      shuffle=False)
    test_gen = utils_data.DataSequence(test_df,
                                       encoding_functions=[one_hot_fn],
                                       output_encoding_fn=out_encoding_fn,
                                       shuffle=False)

    # Build model
    utr_model = framepool_model.create_frame_slice_model(
        kernel_size=[7, 7, 7],
        only_max_pool=False,
        padding="same",
        skip_connections="residual",
        use_scaling_regression=False)

    # Train with tf.data wrapper
    sample_input, sample_label = train_gen[0]

    def generator_wrapper():
        while True:
            train_gen.on_epoch_end()
            try:
                for inputs, labels in train_gen:
                    yield inputs, labels
            except IndexError:
                continue

    output_signature = (
        tf.TensorSpec(shape=(None, None, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
    train_dataset = tf.data.Dataset.from_generator(
        generator_wrapper, output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

    ckpt_path = os.path.join(output_dir, "best_model.h5")
    utr_model.fit(train_dataset,
                  epochs=15,
                  steps_per_epoch=len(train_gen),
                  validation_data=val_gen,
                  verbose=0,
                  callbacks=[
                      ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True),
                      EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                  ])

    # Load best and predict
    utr_model = load_model(ckpt_path, custom_objects={
        'FrameSliceLayer': framepool_model.FrameSliceLayer,
        'compute_pad_mask': framepool_model.compute_pad_mask,
        'apply_pad_mask': framepool_model.apply_pad_mask,
        'global_avg_pool_masked': framepool_model.global_avg_pool_masked,
        'PrintShapeLayer': framepool_model.PrintShapeLayer,
        'interaction_term': framepool_model.interaction_term,
    })

    pred_scaled = utr_model.predict(test_gen, verbose=0).flatten()
    pred = inverse_fn(pred_scaled)

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
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--jobs-per-gpu', type=int, default=1)
    parser.add_argument('--names', nargs='+', default=None)
    parser.add_argument('--folds', nargs='+', type=int, default=None, help='Run only these fold indices')
    parser.add_argument('--force', action='store_true', help='Ignore cache')
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

    for ds_name, ds_cfg in cfg["datasets"].items():
        if "models" in ds_cfg and "framepool" not in ds_cfg["models"]:
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
                out = os.path.join(ds_output, "framepool", f"fold_{fold_i}")

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
