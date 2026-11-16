#!/usr/bin/env python3
"""UTR-LM benchmark — dispatches model_te.py as subprocess per fold.

Usage:
    conda activate benchmark_torch
    python baselines/UTR-LM/run.py --config config.yaml --gpus 0,1
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from benchlib import (check_cached, Timer, load_config, generate_splits, generate_rank_split,
                      load_indices, load_scaler, compute_metrics)
from benchlib.config import get_data_path, get_rank_paths

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SCRIPT = os.path.join(_THIS_DIR, 'model_te.py')
MODEL_MRL_SCRIPT = os.path.join(_THIS_DIR, 'model_mrl.py')


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
            if "models" in ds_cfg and "UTR-LM" not in ds_cfg["models"]:
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

    for ds_name, ds_cfg in cfg["datasets"].items():
        if "models" in ds_cfg and "UTR-LM" not in ds_cfg["models"]:
            continue
        for name in ds_cfg["names"]:
            job_id = f"{ds_name}/{name}"
            if args.names and job_id not in args.names:
                continue

            print(f"\n{'='*50}\nDataset: {job_id}\n{'='*50}")

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
                out = os.path.join(ds_output, "UTR-LM", f"fold_{fold_i}")

                if not args.force:
                    cached = check_cached(out)
                    if cached:
                        all_metrics.append(cached)
                        print(f"  fold {fold_i}: cached")
                        continue

                # Save indices as .npy for subprocess
                indices = load_indices(fold_dir)
                train_npy = os.path.join(fold_dir, "train_idx.npy")
                val_npy = os.path.join(fold_dir, "val_idx.npy")
                test_npy = os.path.join(fold_dir, "test_idx.npy")
                np.save(train_npy, indices["train"])
                np.save(val_npy, indices["val"])

                # Scaler path
                scaler_path = os.path.join(fold_dir, "scaler.joblib")

                # Data file and test indices
                if ds_cfg["split"] == "rank":
                    with open(os.path.join(fold_dir, "meta.json")) as f:
                        meta = json.load(f)
                    data_file = meta["train_file"]
                    # test is the entire test_file — save all indices
                    test_df = pd.read_csv(meta["test_file"], sep='\t')
                    np.save(test_npy, np.arange(len(test_df)))
                    test_data_file = meta["test_file"]
                else:
                    data_file = data_path
                    np.save(test_npy, indices["test"])
                    test_data_file = data_file

                # Run model_te.py as subprocess
                cmd = [
                    sys.executable,
                    MODEL_MRL_SCRIPT if ds_name == "mrl_rank" else MODEL_SCRIPT,
                    '--data_file', data_file,
                    '--utr_col', ds_cfg.get("utr_col", "utr5"),
                    '--label_col', ds_cfg["label_col"],
                    '--lr', '0.01',
                    '--epochs', '300' if ds_name != "mrl_rank" else '200',
                    '--inp_len', str(ds_cfg.get("inp_len", 100)),
                    '--train_idx', train_npy,
                    '--val_idx', val_npy,
                    '--test_idx', test_npy,
                    '--scaler_path', scaler_path,
                    '--output_dir', out,
                ]
                if ds_cfg["split"] == "rank":
                    cmd += ['--test_data_file', meta["test_file"]]

                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
                if 'CONDA_PREFIX' in env:
                    env['LD_LIBRARY_PATH'] = f"{env['CONDA_PREFIX']}/lib:{env.get('LD_LIBRARY_PATH', '')}"

                with Timer() as t:
                    result = subprocess.run(cmd, env=env)

                if result.returncode != 0:
                    print(f"  fold {fold_i}: FAILED (exit code {result.returncode})")
                    continue

                # Read predictions and compute metrics
                pred_path = os.path.join(out, "predictions.tsv")
                if os.path.exists(pred_path):
                    pred_df = pd.read_csv(pred_path, sep='\t')
                    metrics = compute_metrics(pred_df['y_true'].values, pred_df['y_pred'].values)
                    metrics["elapsed_sec"] = t.elapsed
                    with open(os.path.join(out, "metrics.json"), "w") as f:
                        json.dump(metrics, f, indent=2)
                    all_metrics.append(metrics)
                    print(f"  fold {fold_i}: pearson={metrics['pearson']:.4f} spearman={metrics['spearman']:.4f} ({t.elapsed:.0f}s)")
                else:
                    print(f"  fold {fold_i}: no predictions file")

            if all_metrics:
                avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
                print(f"  AVG: pearson={avg['pearson']:.4f} spearman={avg['spearman']:.4f}")


if __name__ == "__main__":
    main()
