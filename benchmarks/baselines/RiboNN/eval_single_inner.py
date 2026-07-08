#!/usr/bin/env python3
"""Evaluate RiboNN (single/multi/finetune) using one inner fold checkpoint (no ensemble).

Saves results to {fold_dir}/single_inner/metrics.json and predictions.tsv.
Runs all three model variants by default.

Usage:
    conda activate benchmark_torch && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
    python benchmarks/baselines/RiboNN/eval_single_inner.py --config benchmarks/config.yaml
    python benchmarks/baselines/RiboNN/eval_single_inner.py --config benchmarks/config.yaml \\
        --names celltype_te/hek293t_TE --folds 0 --inner 2
"""

import argparse
import json
import os
import sys
import time
from glob import glob

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import RiboNN
from src.data import RiboNNDataModule
from run import DEFAULT_CONFIG

from benchlib import load_config, load_indices, load_scaler, compute_metrics
from benchlib.config import get_data_path

torch.set_float32_matmul_precision('high')


def _prepare_ribonn_df(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    d = df.copy()
    d = d.rename(columns={"utr5": "utr5_sequence", "cds": "cds_sequence", "utr3": "utr3_sequence"})
    for col in ["utr5_sequence", "cds_sequence", "utr3_sequence"]:
        if col in d.columns:
            d[col] = d[col].str.strip().str.upper().str.replace("U", "T")
    if "utr5_sequence" in d.columns and "cds_sequence" in d.columns:
        utr3 = d.get("utr3_sequence", "")
        d["tx_sequence"] = d["utr5_sequence"] + d["cds_sequence"] + utr3
        d["utr5_size"] = d["utr5_sequence"].str.len()
        d["cds_size"] = d["cds_sequence"].str.len()
        d["tx_size"] = d["tx_sequence"].str.len()
        d["utr3_size"] = d["tx_size"] - d["utr5_size"] - d["cds_size"]
    if label_col in d.columns and not label_col.startswith("TE_"):
        d["TE_target"] = d[label_col]
    elif label_col in d.columns and label_col.startswith("TE_"):
        d["TE_target"] = d[label_col]
        other_te = [c for c in d.columns if c.startswith("TE_") and c != "TE_target"]
        d = d.drop(columns=other_te)
    return d


def _predict(model, dm, config, gpu_id: int) -> np.ndarray:
    test_only_df = dm.df.query("split == 'test'").reset_index(drop=True)
    dm_pred = RiboNNDataModule(config)
    dm_pred.df = test_only_df
    dm_pred.max_utr5_len = dm.max_utr5_len
    dm_pred.max_cds_utr3_len = dm.max_cds_utr3_len
    dm_pred.max_tx_len = dm.max_tx_len
    pred_dl = dm_pred.make_dataloader("predict", dm.test_batch_size)
    trainer = pl.Trainer(accelerator='gpu', devices=1, logger=False, enable_progress_bar=False)
    results = trainer.predict(model, dataloaders=pred_dl)
    return torch.cat(results).numpy(), test_only_df


def eval_fold(fold_output_dir: str, inner: int, gpu_id: int,
              ds_cfg: dict, data_path: str,
              model_dir_name: str,
              mt_data_path: str = None, mt_splits_base: str = None) -> dict | None:
    """Load inner_{inner} checkpoint and predict on test set."""
    inner_dir = os.path.join(fold_output_dir, f"inner_{inner}")
    ckpts = glob(os.path.join(inner_dir, "*.ckpt"))
    if not ckpts:
        print(f"    [skip] no ckpt in inner_{inner}")
        return None
    if not os.path.exists(os.path.join(fold_output_dir, "predictions.tsv")):
        print(f"    [skip] no predictions.tsv (fold not complete)")
        return None

    out_dir = os.path.join(fold_output_dir, "single_inner")
    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    os.makedirs(out_dir, exist_ok=True)

    best_ckpt = ckpts[0]
    fold_name = os.path.basename(fold_output_dir)  # fold_0

    # --- Resolve splits dir ---
    # fold_output_dir: .../celltype_te/hek293t_TE/{model_dir_name}/fold_0
    if model_dir_name == "RiboNN_ft":
        # finetune uses multi-target splits
        splits_dir = os.path.join(mt_splits_base, fold_name)
    else:
        splits_dir = os.path.join(
            os.path.dirname(os.path.dirname(fold_output_dir)), "splits", fold_name)

    if not os.path.exists(splits_dir):
        print(f"    [skip] splits not found: {splits_dir}")
        return None

    indices = load_indices(splits_dir)

    # --- Load data ---
    if model_dir_name == "RiboNN_ft":
        df = pd.read_csv(mt_data_path, sep='\t')
        # extract celltype name: hek293t_TE -> TE_hek293t
        ct_dir = os.path.basename(os.path.dirname(os.path.dirname(fold_output_dir)))  # hek293t_TE
        te_col = "TE_" + ct_dir.replace("_TE", "")
        if te_col not in df.columns:
            print(f"    [skip] {te_col} not in multi-target data")
            return None
        train_val_df = df.iloc[np.concatenate([indices["train"], indices["val"]])].copy()
        test_df = df.iloc[indices["test"]].copy()
        train_val_df = train_val_df[train_val_df[te_col].notna()].reset_index(drop=True)
        test_df = test_df[test_df[te_col].notna()].reset_index(drop=True)
        label_col = te_col
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(train_val_df[te_col].values.reshape(-1, 1))
        scale_fn = lambda x: scaler.transform(np.asarray(x).reshape(-1, 1)).flatten()
        inverse_fn = lambda x: scaler.inverse_transform(np.asarray(x).reshape(-1, 1)).flatten()
    else:
        df = pd.read_csv(data_path, sep='\t')
        train_val_df = df.iloc[np.concatenate([indices["train"], indices["val"]])].copy()
        test_df = df.iloc[indices["test"]].copy()
        scale_fn, inverse_fn = load_scaler(splits_dir)
        label_col = ds_cfg["label_col"]
        if label_col.startswith("^"):
            label_col = None  # multi-target: keep all TE_ cols as-is

    multi_target = model_dir_name == "RiboNN" and ds_cfg["label_col"].startswith("^")

    # --- Prepare ---
    test_prep = _prepare_ribonn_df(test_df, label_col or "")
    train_prep = _prepare_ribonn_df(train_val_df, label_col or "")

    if not multi_target:
        test_prep["TE_target"] = scale_fn(test_prep["TE_target"].values)
        train_prep["TE_target"] = scale_fn(train_prep["TE_target"].values)
        te_cols = None
    else:
        te_cols = [c for c in test_prep.columns if c.startswith("TE_")]

    test_prep["split"] = "test"
    train_prep["split"] = "train"
    full_df = pd.concat([train_prep, test_prep], ignore_index=True)

    config = DEFAULT_CONFIG.copy()
    if multi_target:
        config["with_NAs"] = True
    config["tx_info_path"] = None
    config["test_fold"] = 0
    config["valid_fold"] = 0

    dm = RiboNNDataModule(config)
    dm.setup_from_df(full_df, config)
    config["num_targets"] = dm.num_targets
    config["len_after_conv"] = dm.get_sequence_length_after_ConvBlocks()

    model = RiboNN.load_from_checkpoint(best_ckpt)
    model.eval()

    t0 = time.time()
    preds, test_only_df = _predict(model, dm, config, gpu_id)
    elapsed = time.time() - t0

    if multi_target:
        preds = preds  # shape (N, num_targets)
        y_true_raw = test_only_df[te_cols].values
        all_target_metrics = {}
        for i, col in enumerate(te_cols):
            mask = ~np.isnan(y_true_raw[:, i])
            if mask.sum() > 10:
                all_target_metrics[col] = compute_metrics(y_true_raw[mask, i], preds[mask, i])
        metrics = {k: float(np.mean([m[k] for m in all_target_metrics.values()]))
                   for k in ["pearson", "spearman", "r2", "rmse"]}
        metrics["per_target"] = all_target_metrics
        results_df = pd.DataFrame(y_true_raw, columns=te_cols)
        for i, col in enumerate(te_cols):
            results_df[f"pred_{col}"] = preds[:, i]
    else:
        pred = inverse_fn(preds.flatten())
        y_true_raw = inverse_fn(test_only_df["TE_target"].values)
        metrics = compute_metrics(y_true_raw, pred)
        results_df = pd.DataFrame({"y_true": y_true_raw, "y_pred": pred})

    results_df.to_csv(os.path.join(out_dir, "predictions.tsv"), sep='\t', index=False)
    metrics["elapsed_sec"] = elapsed
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"    pearson={metrics['pearson']:.4f} spearman={metrics['spearman']:.4f} "
          f"r2={metrics['r2']:.4f} rmse={metrics['rmse']:.4f} elapsed={metrics['elapsed_sec']:.1f}s")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/config.yaml")
    parser.add_argument("--names", nargs="+", default=None, help="e.g. celltype_te/hek293t_TE")
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--inner", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", nargs="+", default=["RiboNN", "RiboNN_ft"],
                        choices=["RiboNN", "RiboNN_ft"], help="Which model dirs to eval")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = load_config(args.config)
    output_root = cfg["output_dir"]

    # multi-target data/splits for finetune
    mt_ds_cfg = cfg["datasets"].get("celltype_te_multi", {})
    mt_data_path = get_data_path(mt_ds_cfg, "multitarget_all") if mt_ds_cfg else None
    mt_splits_base = os.path.join(output_root, "celltype_te_multi", "multitarget_all", "splits") if mt_ds_cfg else None

    MODEL_DIRS = args.model

    for ds_name, ds_cfg in cfg["datasets"].items():
        if "models" in ds_cfg and "RiboNN" not in ds_cfg["models"]:
            continue
        for name in ds_cfg["names"]:
            job_id = f"{ds_name}/{name}"
            if args.names and job_id not in args.names:
                continue

            data_path = get_data_path(ds_cfg, name)

            for model_dir in MODEL_DIRS:
                ds_output = os.path.join(output_root, ds_name, name, model_dir)
                if not os.path.exists(ds_output):
                    continue

                print(f"\n{job_id} [{model_dir}]")
                fold_dirs = sorted(glob(os.path.join(ds_output, "fold_*")))
                all_metrics = []
                for fold_dir in fold_dirs:
                    fold_i = int(os.path.basename(fold_dir).replace("fold_", ""))
                    if args.folds and fold_i not in args.folds:
                        continue
                    print(f"  fold {fold_i}")
                    m = eval_fold(fold_dir, args.inner, args.gpu, ds_cfg, data_path,
                                  model_dir, mt_data_path, mt_splits_base)
                    if m:
                        all_metrics.append(m)

                if all_metrics:
                    avg_p = np.mean([m["pearson"] for m in all_metrics])
                    avg_s = np.mean([m["spearman"] for m in all_metrics])
                    print(f"  AVG pearson={avg_p:.4f} spearman={avg_s:.4f}")


if __name__ == "__main__":
    main()
