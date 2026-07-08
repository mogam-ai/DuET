#!/usr/bin/env python3
"""RiboNN benchmark — nested CV with inner fold ensemble.

Based on src/train.py (train_model_nested_cv).
Changes: MLflow removed, data loading via benchlib, DDP removed.
Inner fold ensemble preserved (original behavior).

Usage:
    conda activate benchmark_torch
    python baselines/RiboNN/run.py [--config config.yaml] [--gpu 0]
"""

import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import RiboNN
from src.data import RiboNNDataModule

from benchlib import (check_cached, Timer, load_config, generate_splits, generate_rank_split,
                      load_indices, load_scaler, compute_metrics, write_json_atomic)
from benchlib.config import get_data_path, get_rank_paths

torch.set_float32_matmul_precision('high')

INNER_SPLIT_RANDOM_STATE = 42
INNER_SPLIT_FILE = "ribonn_inner_indices.json"
INNER_FINETUNE_SPLIT_FILE = "ribonn_ft_inner_indices.json"

# Default RiboNN config (from conf.yml)
DEFAULT_CONFIG = {
    "num_workers": 2,
    "remove_extreme_txs": True,
    "train_batch_size": 64,
    "val_batch_size": 128,
    "test_batch_size": 128,
    "max_seq_len": 12288,
    "max_utr5_len": 1381,
    "max_cds_utr3_len": 11937,
    "num_targets": 1,
    "target_column_pattern": "^TE_",
    "pad_5_prime": True,
    "split_utr5_cds_utr3_channels": False,
    "label_codons": True,
    "label_3rd_nt_of_codons": False,
    "label_utr5": False,
    "label_utr3": False,
    "label_splice_sites": False,
    "label_up_probs": False,
    "with_NAs": False,
    "max_shift": 0,
    "symmetric_shift": True,
    "augmentation_shifts": [-3, -2, -1, 0, 1, 2, 3],
    "residual": False,
    "go_backwards": True,
    "rnn_type": "gru",
    "filters": 64,
    "kernel_size": 5,
    "conv_stride": 1,
    "conv_dilation": 1,
    "conv_padding": 0,
    "num_conv_layers": 10,
    "dropout": 0.3,
    "ln_epsilon": 0.007,
    "bn_momentum": 0.9,
    "optimizer": "AdamW",
    "lr": 0.0001,
    "l2_scale": 0.001,
    "min_lr": 0.0000001,
    "adam_beta1": 0.90,
    "adam_beta2": 0.998,
}


class _PrintMetrics(pl.Callback):
    def __init__(self, prefix=""):
        super().__init__()
        self.prefix = prefix

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        epoch = trainer.current_epoch
        head = f"[{self.prefix}epoch: {epoch:3d}]" if self.prefix else f"[epoch: {epoch:3d}]"
        parts = [head]
        for k in ("train_loss", "val_loss", "val_pearson", "val_r2"):
            if k in m:
                parts.append(f"{k}: {m[k]:.4f}")
        if len(parts) > 1:
            print(" | ".join(parts), flush=True)


def _build_inner_splits(train_val_size, inner_cv_folds):
    all_idx = np.arange(train_val_size)
    kf = KFold(
        n_splits=inner_cv_folds,
        shuffle=True,
        random_state=INNER_SPLIT_RANDOM_STATE,
    )
    return [
        {
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
        }
        for train_idx, val_idx in kf.split(all_idx)
    ]


def _load_or_create_inner_splits(fold_dir, train_val_size, inner_cv_folds, split_file):
    split_path = os.path.join(fold_dir, split_file)

    folds = _build_inner_splits(train_val_size, inner_cv_folds)
    payload = {
        "inner_cv_folds": inner_cv_folds,
        "random_state": INNER_SPLIT_RANDOM_STATE,
        "train_val_size": train_val_size,
        "folds": folds,
    }
    with open(split_path, "w") as f:
        json.dump(payload, f, indent=2)
    return folds, split_path, True


def run_fold(train_df, val_df, test_df, ds_cfg, output_dir, fold_dir, gpu_id,
             inner_cv_folds=9, max_epochs=200, patience=20, verbose=False,
             inner_folds_limit=None):
    """Train RiboNN with inner CV ensemble on one outer fold."""
    os.makedirs(output_dir, exist_ok=True)
    label_col = ds_cfg["label_col"]
    multi_target = label_col.startswith("^")  # regex pattern means multi-target

    # Scale
    scale_fn, inverse_fn = load_scaler(fold_dir)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    if multi_target:
        te_cols = [c for c in test_df.columns if c.startswith("TE_")]
        test_y_raw = test_df[te_cols].values  # (N, num_targets)
    else:
        test_y_raw = test_df[label_col].values

    # Prepare data for RiboNN format
    def prepare_ribonn_df(df):
        """Rename columns and preprocess for RiboNN DataModule."""
        df = df.copy()
        col_map = {"utr5": "utr5_sequence", "cds": "cds_sequence", "utr3": "utr3_sequence"}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        for col in ["utr5_sequence", "cds_sequence", "utr3_sequence"]:
            if col in df.columns:
                df[col] = df[col].str.strip().str.upper().str.replace("U", "T")
        if "utr5_sequence" in df.columns and "cds_sequence" in df.columns:
            utr3 = df["utr3_sequence"] if "utr3_sequence" in df.columns else ""
            df["tx_sequence"] = df["utr5_sequence"] + df["cds_sequence"] + utr3
            df["utr5_size"] = df["utr5_sequence"].str.len()
            df["cds_size"] = df["cds_sequence"].str.len()
            df["tx_size"] = df["tx_sequence"].str.len()
            df["utr3_size"] = df["tx_size"] - df["utr5_size"] - df["cds_size"]
        if not multi_target and label_col in df.columns and not label_col.startswith("TE_"):
            df["TE_target"] = df[label_col]
        return df

    train_val_df = prepare_ribonn_df(train_val_df)
    test_df_scaled = prepare_ribonn_df(test_df)

    # Scale labels (single-target only; multi-target uses raw values with masked loss)
    if not multi_target:
        train_val_df["TE_target"] = scale_fn(train_val_df["TE_target"].values)
        test_df_scaled["TE_target"] = scale_fn(test_df_scaled["TE_target"].values)

    # Inner CV ensemble predictions
    all_test_preds = []
    inner_splits, inner_split_path, inner_split_created = _load_or_create_inner_splits(
        fold_dir=fold_dir,
        train_val_size=len(train_val_df),
        inner_cv_folds=inner_cv_folds,
        split_file=INNER_SPLIT_FILE,
    )
    print(
        f"      inner splits: {'saved' if inner_split_created else 'loaded'} "
        f"{os.path.basename(inner_split_path)}"
    )

    for inner_fold, split in enumerate(inner_splits):
        if inner_folds_limit is not None and inner_fold >= inner_folds_limit:
            break
        inner_train_idx = np.asarray(split["train"])
        inner_val_idx = np.asarray(split["val"])
        print(f"    inner fold {inner_fold}/{inner_cv_folds}")

        combined = train_val_df.copy()
        combined["split"] = "train"
        combined.iloc[inner_val_idx, combined.columns.get_loc("split")] = "valid"
        test_part = test_df_scaled.copy()
        test_part["split"] = "test"
        full_df = pd.concat([combined, test_part], ignore_index=True)

        config = DEFAULT_CONFIG.copy()
        if multi_target:
            config["with_NAs"] = True
        config["tx_info_path"] = None
        config["test_fold"] = 0
        config["valid_fold"] = inner_fold

        dm = RiboNNDataModule(config)
        dm.setup_from_df(full_df, config)
        config["num_targets"] = dm.num_targets
        config["len_after_conv"] = dm.get_sequence_length_after_ConvBlocks()
        config["outer_cv_folds"] = 1
        config["inner_cv_folds"] = inner_cv_folds

        # Debug: verify split sizes
        split_counts = dm.df['split'].value_counts().to_dict()
        print(f"      splits: {split_counts} | max_tx_len={dm.max_tx_len} | num_targets={config['num_targets']}")

        # Check if inner fold checkpoint exists (resume support)
        inner_dir = os.path.join(output_dir, f"inner_{inner_fold}")
        existing_ckpts = glob(os.path.join(inner_dir, "*.ckpt")) if os.path.exists(inner_dir) else []
        if existing_ckpts:
            best_ckpt = existing_ckpts[0]
            print(f"      [cached] loading {os.path.basename(best_ckpt)}")
        else:
            monitor, mon_mode = ("val_r2", "max")
            checkpoint = ModelCheckpoint(
                dirpath=inner_dir,
                save_top_k=1, monitor=monitor, mode=mon_mode)
            early_stop = EarlyStopping(monitor=monitor, mode=mon_mode, patience=patience)
            cbs = [checkpoint, early_stop]
            if verbose:
                ct_name = os.path.basename(os.path.dirname(os.path.dirname(output_dir))).replace("_TE", "")
                outer_fold = os.path.basename(output_dir).replace("fold_", "")
                ct_prefix = f"celltype: {ct_name} | fold: {outer_fold} | inner: {inner_fold} | "
                cbs.append(_PrintMetrics(prefix=ct_prefix))

            trainer = pl.Trainer(
                accelerator='gpu', devices=1,
                max_epochs=max_epochs,
                gradient_clip_val=0.5,
                callbacks=cbs,
                logger=False,
                enable_progress_bar=False,
            )

            model = RiboNN(**config)
            trainer.fit(model, datamodule=dm)
            best_ckpt = checkpoint.best_model_path

        model = RiboNN.load_from_checkpoint(best_ckpt)
        model.eval()

        # Predict on test set
        test_only_df = dm.df.query("split == 'test'").reset_index(drop=True)
        from src.data import RiboNNDataModule as _DM
        dm_pred = _DM(config)
        dm_pred.df = test_only_df
        dm_pred.max_utr5_len = dm.max_utr5_len
        dm_pred.max_cds_utr3_len = dm.max_cds_utr3_len
        dm_pred.max_tx_len = dm.max_tx_len
        pred_dl = dm_pred.make_dataloader("predict", dm.test_batch_size)
        print(f"      predict on {len(test_only_df)} test samples")
        trainer_test = pl.Trainer(accelerator='gpu', devices=1, logger=False, enable_progress_bar=False)
        results = trainer_test.predict(model, dataloaders=pred_dl)
        if results:
            test_pred = torch.cat([r for r in results]).numpy()
            # Get val_r2 from checkpoint for top-k selection
            ckpt_data = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            val_r2 = float("-inf")
            for k, v in ckpt_data.get("callbacks", {}).items():
                if k.startswith("ModelCheckpoint") and "best_model_score" in v:
                    val_r2 = v["best_model_score"]
                    break
            if isinstance(val_r2, torch.Tensor):
                val_r2 = val_r2.item()
            all_test_preds.append((val_r2, test_pred))
            print(f"      pred shape: {test_pred.shape} | val_r2={val_r2:.4f}")
            # Save per-inner predictions for collect / ensemble analysis.
            txid = test_only_df["txID"].values if "txID" in test_only_df.columns else np.arange(len(test_only_df))
            if multi_target:
                # (N, n_targets) → npz 로 celltype(te_cols) 보존하여 저장
                np.savez(os.path.join(inner_dir, "test_pred.npz"),
                         pred=test_pred, txid=txid, te_cols=np.array(te_cols), val_r2=val_r2)
            else:
                inner_pred_df = {"y_pred": test_pred.flatten()}
                if "txID" in test_only_df.columns:
                    inner_pred_df = {"txID": txid, **inner_pred_df}
                pd.DataFrame(inner_pred_df).to_csv(
                    os.path.join(inner_dir, "predictions.tsv"), sep='\t', index=False)

    # Ensemble: top-5 by val_r2
    TOP_K = 5
    if all_test_preds:
        all_test_preds.sort(key=lambda x: x[0], reverse=True)
        top_preds = [p for _, p in all_test_preds[:TOP_K]]
        print(f"    ensemble: top {min(TOP_K, len(top_preds))}/{len(all_test_preds)} models")
        ensemble_pred_scaled = np.mean(top_preds, axis=0)
        if multi_target:
            pred = ensemble_pred_scaled
            test_y_raw = test_only_df[te_cols].values  # use filtered test df
        else:
            pred = inverse_fn(ensemble_pred_scaled.flatten())
            test_y_raw = test_only_df[["TE_target"]].values.flatten() if "TE_target" in test_only_df.columns else test_y_raw
    else:
        pred = np.zeros_like(test_y_raw)

    # Save & compute metrics
    if multi_target:
        # Per-target metrics
        results_df = pd.DataFrame(test_y_raw, columns=te_cols)
        for i, col in enumerate(te_cols):
            results_df[f"pred_{col}"] = pred[:, i]
        results_df.to_csv(os.path.join(output_dir, "predictions.tsv"), sep='\t', index=False)

        all_target_metrics = {}
        for i, col in enumerate(te_cols):
            mask = ~np.isnan(test_y_raw[:, i])
            if mask.sum() > 10:
                m = compute_metrics(test_y_raw[mask, i], pred[mask, i])
                all_target_metrics[col] = m
        # Average across targets
        metrics = {k: np.mean([m[k] for m in all_target_metrics.values()])
                   for k in ["pearson", "spearman", "r2", "rmse"]}
        metrics["per_target"] = all_target_metrics
    else:
        out_df = {"y_true": test_y_raw, "y_pred": pred}
        if "txID" in test_only_df.columns:
            out_df = {"txID": test_only_df["txID"].values, **out_df}
        pd.DataFrame(out_df).to_csv(
            os.path.join(output_dir, "predictions.tsv"), sep='\t', index=False)
        metrics = compute_metrics(test_y_raw, pred)

    if inner_folds_limit is None:
        write_json_atomic(os.path.join(output_dir, "metrics.json"), metrics)
    return metrics


def create_finetune_model(pretrain_ckpt: str, config: dict):
    """Load multi-target pretrained model and replace head for single-target."""
    # Load pretrained multi-target model
    pretrain_config = config.copy()
    pretrain_config["num_targets"] = config["pretrain_num_targets"]
    loaded_model = RiboNN(**pretrain_config)
    state_dict = torch.load(pretrain_ckpt, map_location="cpu", weights_only=False)["state_dict"]
    loaded_model.load_state_dict(state_dict, strict=True)

    # Create single-target model and copy conv layers
    model = RiboNN(**config)
    model.initial_conv = loaded_model.initial_conv
    model.middle_convs = loaded_model.middle_convs

    # Freeze conv, unfreeze batchnorm
    model.initial_conv.requires_grad_(False)
    model.middle_convs.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.LayerNorm)):
            m.requires_grad_(True)

    return model


def run_finetune_fold(train_df, val_df, test_df, ds_cfg, output_dir, fold_dir, gpu_id,
                      pretrain_dir, mt_data_path, inner_cv_folds=9, phase1_epochs=50,
                      max_epochs=200, patience=50, verbose=False, inner_folds_limit=None):
    """Fine-tune multi-target pretrained RiboNN on single celltype.

    Uses multi-target split to avoid data leakage: loads multi-target data,
    splits by pretrain_dir's fold indices, then filters to target celltype.
    """
    os.makedirs(output_dir, exist_ok=True)
    label_col = ds_cfg["label_col"]
    celltype_name = label_col  # e.g. "logratio_te"

    # Load multi-target data and use its split (same as pretrain)
    pretrain_splits_dir = os.path.dirname(pretrain_dir)  # .../RiboNN -> go up to find splits
    # pretrain_dir is like .../RiboNN/fold_0, so splits are at .../splits/fold_0
    mt_base = os.path.dirname(os.path.dirname(pretrain_dir))  # multitarget run base dir
    mt_splits_fold = os.path.join(mt_base, "splits", os.path.basename(pretrain_dir))
    # mt_data_path passed in from config (was hardcoded multitarget_all_TE.tsv)

    mt_indices = load_indices(mt_splits_fold)
    mt_df = pd.read_csv(mt_data_path, sep='\t')

    # Determine target column name from celltype dataset name
    # ds_cfg has label_col="logratio_te", but we need the TE_ column in multi-target
    # Extract celltype from the data file name (e.g. hek293t from hek293t_TE.tsv)
    ct_name = os.path.basename(fold_dir).replace("fold_", "")  # this won't work
    # Better: infer from output_dir which contains the celltype name
    # output_dir is like .../celltype_te/hek293t_TE/RiboNN_ft/fold_0
    ct_from_path = os.path.basename(os.path.dirname(os.path.dirname(output_dir)))  # hek293t_TE
    te_col = "TE_" + ct_from_path.replace("_TE", "")  # TE_hek293t

    if te_col not in mt_df.columns:
        print(f"      [error] {te_col} not found in multi-target data")
        return {"pearson": 0, "spearman": 0, "r2": 0, "rmse": 999}

    # Split multi-target data, keep only rows with valid target for this celltype
    mt_train_val = mt_df.iloc[np.concatenate([mt_indices["train"], mt_indices["val"]])]
    mt_test = mt_df.iloc[mt_indices["test"]]

    # Filter to rows with non-NaN target
    mt_train_val = mt_train_val[mt_train_val[te_col].notna()].reset_index(drop=True)
    mt_test = mt_test[mt_test[te_col].notna()].reset_index(drop=True)

    # Verify no leakage: pretrain train/val txIDs should not overlap with finetune test txIDs
    pretrain_trainval_txids = set(mt_df.iloc[np.concatenate([mt_indices["train"], mt_indices["val"]])]["txID"])
    ft_test_txids = set(mt_test["txID"])
    leakage = pretrain_trainval_txids & ft_test_txids
    print(f"      [{te_col}] train_val={len(mt_train_val)} test={len(mt_test)} | "
          f"pretrain_seen_in_test={len(leakage)}/{len(ft_test_txids)}", flush=True)
    if leakage:
        print(f"      [WARNING] LEAKAGE DETECTED!", flush=True)

    test_y_raw = mt_test[te_col].values

    # Prepare for RiboNN format
    def prepare_ribonn_df(df):
        df = df.copy()
        col_map = {"utr5": "utr5_sequence", "cds": "cds_sequence", "utr3": "utr3_sequence"}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        for col in ["utr5_sequence", "cds_sequence", "utr3_sequence"]:
            if col in df.columns:
                df[col] = df[col].str.strip().str.upper().str.replace("U", "T")
        if "utr5_sequence" in df.columns and "cds_sequence" in df.columns:
            utr3 = df["utr3_sequence"] if "utr3_sequence" in df.columns else ""
            df["tx_sequence"] = df["utr5_sequence"] + df["cds_sequence"] + utr3
            df["utr5_size"] = df["utr5_sequence"].str.len()
            df["cds_size"] = df["cds_sequence"].str.len()
            df["tx_size"] = df["tx_sequence"].str.len()
            df["utr3_size"] = df["tx_size"] - df["utr5_size"] - df["cds_size"]
        df["TE_target"] = df[te_col]
        # Drop all other TE_ columns to keep single target
        other_te_cols = [c for c in df.columns if c.startswith("TE_") and c != "TE_target"]
        df = df.drop(columns=other_te_cols)
        return df

    train_val_df = prepare_ribonn_df(mt_train_val)
    test_df_scaled = prepare_ribonn_df(mt_test)

    # Scale using train stats
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_val_df["TE_target"].values.reshape(-1, 1))
    scale_fn = lambda x: scaler.transform(np.asarray(x).reshape(-1, 1)).flatten()
    inverse_fn = lambda x: scaler.inverse_transform(np.asarray(x).reshape(-1, 1)).flatten()

    train_val_df["TE_target"] = scale_fn(train_val_df["TE_target"].values)
    test_df_scaled["TE_target"] = scale_fn(test_df_scaled["TE_target"].values)

    print(f"      celltype={te_col} | train_val={len(train_val_df)} | test={len(test_df_scaled)}")

    all_test_preds = []
    inner_splits, inner_split_path, inner_split_created = _load_or_create_inner_splits(
        fold_dir=fold_dir,
        train_val_size=len(train_val_df),
        inner_cv_folds=inner_cv_folds,
        split_file=INNER_FINETUNE_SPLIT_FILE,
    )
    print(
        f"      inner splits: {'saved' if inner_split_created else 'loaded'} "
        f"{os.path.basename(inner_split_path)}"
    )

    for inner_fold, split in enumerate(inner_splits):
        if inner_folds_limit is not None and inner_fold >= inner_folds_limit:
            break
        inner_train_idx = np.asarray(split["train"])
        inner_val_idx = np.asarray(split["val"])
        print(f"    inner fold {inner_fold}/{inner_cv_folds}")

        combined = train_val_df.copy()
        combined["split"] = "train"
        combined.iloc[inner_val_idx, combined.columns.get_loc("split")] = "valid"
        test_part = test_df_scaled.copy()
        test_part["split"] = "test"
        full_df = pd.concat([combined, test_part], ignore_index=True)

        config = DEFAULT_CONFIG.copy()
        config["lr"] = 0.0001
        config["remove_extreme_txs"] = False  # original uses False for fine-tune
        config["tx_info_path"] = None
        config["test_fold"] = 0
        config["valid_fold"] = inner_fold

        dm = RiboNNDataModule(config)
        dm.setup_from_df(full_df, config)
        config["num_targets"] = dm.num_targets  # 1
        config["len_after_conv"] = dm.get_sequence_length_after_ConvBlocks()

        # Find pretrained checkpoint for this inner fold
        pretrain_ckpts = glob(os.path.join(pretrain_dir, f"inner_{inner_fold}", "*.ckpt"))
        if not pretrain_ckpts:
            print(f"      [skip] no pretrain checkpoint for inner_{inner_fold}")
            continue
        pretrain_ckpt = pretrain_ckpts[0]

        inner_dir = os.path.join(output_dir, f"inner_{inner_fold}")
        existing_ckpts = glob(os.path.join(inner_dir, "*.ckpt")) if os.path.exists(inner_dir) else []
        if existing_ckpts:
            best_ckpt = existing_ckpts[0]
            print(f"      [cached] {os.path.basename(best_ckpt)}")
        else:
            # Determine pretrain num_targets from checkpoint
            ckpt_state = torch.load(pretrain_ckpt, map_location="cpu", weights_only=False)["state_dict"]
            pretrain_num_targets = ckpt_state["head.7.weight"].shape[0]
            config["pretrain_num_targets"] = pretrain_num_targets

            model = create_finetune_model(pretrain_ckpt, config)

            # Sanity check: forward pass on a single batch
            with torch.no_grad():
                model.eval()
                tmp_dl = dm.train_dataloader()
                sample = next(iter(tmp_dl))
                x_sample = sample[0] if isinstance(sample, (list, tuple)) else sample
                y_sample = sample[1] if isinstance(sample, (list, tuple)) else None
                out_sample = model(x_sample)
                print(f"      [sanity] x.shape={x_sample.shape} | "
                      f"x.range=[{x_sample.min():.3f},{x_sample.max():.3f}] | "
                      f"out.shape={out_sample.shape} | "
                      f"out.range=[{out_sample.min():.3f},{out_sample.max():.3f}] | "
                      f"out.has_nan={out_sample.isnan().any().item()}", flush=True)
                if y_sample is not None:
                    print(f"      [sanity] y.range=[{y_sample.min():.3f},{y_sample.max():.3f}] | "
                          f"y.has_nan={y_sample.isnan().any().item()}", flush=True)
                model.train()

            # Phase 1: train head only
            p1_cbs = []
            if verbose:
                ct_name = os.path.basename(os.path.dirname(os.path.dirname(output_dir))).replace("_TE", "")
                outer_fold = os.path.basename(output_dir).replace("fold_", "")
                ct_prefix = f"celltype: {ct_name} | fold: {outer_fold} | inner: {inner_fold} | P1 "
                p1_cbs.append(_PrintMetrics(prefix=ct_prefix))
            trainer = pl.Trainer(
                accelerator='gpu', devices=1,
                max_epochs=phase1_epochs,
                gradient_clip_val=0.5,
                callbacks=p1_cbs,
                logger=False,
                enable_progress_bar=False,
                enable_checkpointing=False,
            )
            trainer.fit(model, datamodule=dm)

            # Phase 2: unfreeze all, lower lr
            model.unfreeze()
            model.lr = 0.00001
            n_params_total = sum(p.numel() for p in model.parameters())
            n_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"      P1→P2 | unfreeze all | lr={model.lr} | params: {n_params_train}/{n_params_total} trainable", flush=True)
            checkpoint = ModelCheckpoint(
                dirpath=inner_dir,
                save_top_k=1, monitor="val_r2", mode="max")
            early_stop = EarlyStopping(monitor="val_r2", mode="max", patience=patience)
            cbs = [checkpoint, early_stop]
            if verbose:
                ct_name = os.path.basename(os.path.dirname(os.path.dirname(output_dir))).replace("_TE", "")
                outer_fold = os.path.basename(output_dir).replace("fold_", "")
                ct_prefix = f"celltype: {ct_name} | fold: {outer_fold} | inner: {inner_fold} | P2 "
                cbs.append(_PrintMetrics(prefix=ct_prefix))

            trainer = pl.Trainer(
                accelerator='gpu', devices=1,
                max_epochs=max_epochs - phase1_epochs,
                gradient_clip_val=0.5,
                callbacks=cbs,
                logger=False,
                enable_progress_bar=False,
            )
            trainer.fit(model, datamodule=dm)
            best_ckpt = checkpoint.best_model_path

        model = RiboNN.load_from_checkpoint(best_ckpt)
        model.eval()

        # Predict
        test_only_df = dm.df.query("split == 'test'").reset_index(drop=True)
        from src.data import RiboNNDataModule as _DM
        dm_pred = _DM(config)
        dm_pred.df = test_only_df
        dm_pred.max_utr5_len = dm.max_utr5_len
        dm_pred.max_cds_utr3_len = dm.max_cds_utr3_len
        dm_pred.max_tx_len = dm.max_tx_len
        pred_dl = dm_pred.make_dataloader("predict", dm.test_batch_size)
        trainer_test = pl.Trainer(accelerator='gpu', devices=1, logger=False, enable_progress_bar=False)
        results = trainer_test.predict(model, dataloaders=pred_dl)
        if results:
            test_pred = torch.cat([r for r in results]).numpy().flatten()
            ckpt_data = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            val_r2 = float("-inf")
            for k, v in ckpt_data.get("callbacks", {}).items():
                if k.startswith("ModelCheckpoint") and "best_model_score" in v:
                    val_r2 = v["best_model_score"]
                    break
            if isinstance(val_r2, torch.Tensor):
                val_r2 = val_r2.item()
            all_test_preds.append((val_r2, test_pred))
            print(f"      val_r2={val_r2:.4f}")
            # inner 별 test 예측 저장 (원스케일 복원 + txID). downstream collect 용.
            txid = test_only_df["txID"].values if "txID" in test_only_df.columns else np.arange(len(test_only_df))
            np.savez(os.path.join(inner_dir, "test_pred.npz"),
                     pred=inverse_fn(test_pred), pred_scaled=test_pred,
                     true=inverse_fn(test_only_df["TE_target"].values),
                     txid=txid, val_r2=val_r2)

    # Ensemble: top-5 by val_r2
    TOP_K = 5
    if all_test_preds:
        all_test_preds.sort(key=lambda x: x[0], reverse=True)
        top_preds = [p for _, p in all_test_preds[:TOP_K]]
        print(f"    ensemble: top {min(TOP_K, len(top_preds))}/{len(all_test_preds)} models")
        pred = inverse_fn(np.mean(top_preds, axis=0))
        # Use filtered test_only_df for y_true (some long transcripts removed by _prepare_df)
        test_y_raw = inverse_fn(test_only_df["TE_target"].values)
    else:
        pred = np.zeros_like(test_y_raw)

    out_df = {"y_true": test_y_raw, "y_pred": pred}
    if "txID" in test_only_df.columns:
        out_df = {"txID": test_only_df["txID"].values, **out_df}
    pd.DataFrame(out_df).to_csv(
        os.path.join(output_dir, "predictions.tsv"), sep='\t', index=False)
    metrics = compute_metrics(test_y_raw, pred)
    if inner_folds_limit is None:
        write_json_atomic(os.path.join(output_dir, "metrics.json"), metrics)
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
    parser.add_argument('--predict-only', action='store_true',
                        help='Skip training (reuse ckpt), regenerate inner test_pred only')
    parser.add_argument('--worker-id', type=int, default=None)
    parser.add_argument('--inner-folds', type=int, default=9)
    parser.add_argument('--inner-folds-limit', type=int, default=None,
                        help='Run only first N inner folds (e.g. 1 for inner_0 only preview)')
    parser.add_argument('--verbose', action='store_true', help='Show progress bar')
    parser.add_argument('--save-inner-indices-only', action='store_true',
                        help='Save RiboNN inner CV indices under each outer fold and exit')
    parser.add_argument('--finetune', type=str, default=None,
                        help='Path to multi-target pretrain output dir (e.g. .../celltype_te_multi/multitarget_all/RiboNN/fold_0)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = cfg["output_dir"]

    # Multitask data path for finetune (config-based, was hardcoded multitarget_all_TE.tsv)
    ft_mt_data_path = None
    if "celltype_te_multi" in cfg["datasets"]:
        _mt_cfg = cfg["datasets"]["celltype_te_multi"]
        _mt_name = _mt_cfg["names"][0]
        ft_mt_data_path = get_data_path(_mt_cfg, _mt_name)

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
        if args.predict_only:
            extra += ["--predict-only"]
        if args.verbose:
            extra += ["--verbose"]
        if args.save_inner_indices_only:
            extra += ["--save-inner-indices-only"]
        if args.finetune:
            extra += ["--finetune", args.finetune]
        if args.inner_folds_limit is not None:
            extra += ["--inner-folds-limit", str(args.inner_folds_limit)]
        run_parallel(__file__, all_jobs, gpus, args.jobs_per_gpu, extra_args=extra)
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    for ds_name, ds_cfg in cfg["datasets"].items():
        if "models" in ds_cfg and "RiboNN" not in ds_cfg["models"]:
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
                model_name = "RiboNN_ft" if args.finetune else "RiboNN"
                out = os.path.join(ds_output, model_name, f"fold_{fold_i}")

                # Check cache
                if not args.save_inner_indices_only:
                    if not args.force and not args.predict_only:
                        cached = check_cached(out)
                        if cached:
                            all_metrics.append(cached)
                            print(f"  fold {fold_i}: cached")
                            continue
                    elif args.force and os.path.exists(out):
                        # --force 일 때만 출력 디렉토리 삭제.
                        # predict-only 는 기존 ckpt 를 재사용해야 하므로 절대 삭제 안 함.
                        import shutil
                        shutil.rmtree(out)

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

                if args.save_inner_indices_only:
                    if args.finetune:
                        pretrain_dir = os.path.join(args.finetune, f"fold_{fold_i}")
                        if not os.path.exists(pretrain_dir):
                            pretrain_dir = args.finetune
                        mt_base = os.path.dirname(os.path.dirname(pretrain_dir))
                        mt_data_path = ft_mt_data_path
                        mt_indices = load_indices(os.path.join(mt_base, "splits", os.path.basename(pretrain_dir)))
                        mt_df = pd.read_csv(mt_data_path, sep='\t')
                        ct_from_path = name
                        te_col = "TE_" + ct_from_path.replace("_TE", "")
                        if te_col not in mt_df.columns:
                            raise ValueError(f"{te_col} not found in {mt_data_path}")
                        mt_train_val = mt_df.iloc[np.concatenate([mt_indices["train"], mt_indices["val"]])]
                        mt_train_val = mt_train_val[mt_train_val[te_col].notna()].reset_index(drop=True)
                        _, split_path, created = _load_or_create_inner_splits(
                            fold_dir=fold_dir,
                            train_val_size=len(mt_train_val),
                            inner_cv_folds=args.inner_folds,
                            split_file=INNER_FINETUNE_SPLIT_FILE,
                        )
                    else:
                        train_val_size = len(train_df) + len(val_df)
                        _, split_path, created = _load_or_create_inner_splits(
                            fold_dir=fold_dir,
                            train_val_size=train_val_size,
                            inner_cv_folds=args.inner_folds,
                            split_file=INNER_SPLIT_FILE,
                        )
                    action = "saved" if created else "reused"
                    print(f"  fold {fold_i}: {action} {os.path.basename(split_path)}")
                    continue

                with Timer() as t:
                    if args.finetune:
                        # Resolve pretrain dir for this fold
                        pretrain_dir = os.path.join(args.finetune, f"fold_{fold_i}")
                        if not os.path.exists(pretrain_dir):
                            pretrain_dir = args.finetune  # user passed fold-level dir directly
                        metrics = run_finetune_fold(train_df, val_df, test_df, ds_cfg, out, fold_dir,
                                                   args.gpu, pretrain_dir=pretrain_dir,
                                                   mt_data_path=ft_mt_data_path,
                                                   inner_cv_folds=args.inner_folds, verbose=args.verbose,
                                                   inner_folds_limit=args.inner_folds_limit)
                    else:
                        metrics = run_fold(train_df, val_df, test_df, ds_cfg, out, fold_dir,
                                           args.gpu, inner_cv_folds=args.inner_folds,
                                           verbose=args.verbose, inner_folds_limit=args.inner_folds_limit)
                metrics["elapsed_sec"] = t.elapsed
                all_metrics.append(metrics)
                print(f"  fold {fold_i}: pearson={metrics['pearson']:.4f} spearman={metrics['spearman']:.4f}")

            if all_metrics:
                avg = {k: np.mean([m[k] for m in all_metrics])
                       for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))}
                print(f"  AVG: pearson={avg['pearson']:.4f} spearman={avg['spearman']:.4f} "
                      f"r2={avg['r2']:.4f} rmse={avg['rmse']:.4f}")


if __name__ == "__main__":
    main()
