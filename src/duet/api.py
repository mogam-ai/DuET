#!/usr/bin/env python3
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

import os
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
from loguru import logger
from omegaconf import OmegaConf

from .models.module import Module
from .data.datamodule import DataModule
from .configs.config import load_cfgs

# WandB local-footprint minimization (avoid Lustre inode pressure). Cloud sync unaffected.
os.environ.setdefault('WANDB_CONSOLE', 'off')
os.environ.setdefault('WANDB_SILENT', 'true')
os.environ.setdefault('WANDB_QUIET', 'true')


def _ckpt_dir() -> Path:
    """Bundled checkpoint directory shipped with the package."""
    return Path(__file__).parent / "ckpts" / "duet"


def predict(query: list[dict],
            target=None,
            device: str = "cuda",
            batch_size: int = 256,
            use_tqdm: bool = True):
    """Predict translation efficiency for sequences with the bundled DuET model.

    DuET predicts TE for all cell types at once (multi-target).

    Args:
        query: list of dicts, each with 'utr5' and 'cds' keys.
        target: which cell type(s) to return.
            - None:       DataFrame (rows = query order, cols = all cell types)
            - str:        list[float] for that single cell type
            - list[str]:  DataFrame (rows = query, cols = selected cell types)
        device: inference device.
        batch_size: inference batch size.
        use_tqdm: show a progress bar.

    Returns:
        pandas.DataFrame or list[float] (see `target`).

    Cell-type names are the model's TE columns with the 'TE_' prefix stripped.
    Unknown names in `target` raise ValueError.
    """
    ckpt_dir = _ckpt_dir()
    cfg, dict_cfg = load_cfgs(
        [str(ckpt_dir / "config.yaml")],
        {"use_wandb": False, "datamodule.do_kfold_test": False},
    )

    model = Module.load_from_checkpoint(
        str(ckpt_dir / "model.ckpt"), cfg=cfg, dict_cfg=dict_cfg,
        strict=False, weights_only=True, map_location=device)
    model.to(device)
    model.eval()

    # Cell-type names ship alongside the checkpoint (predict-only input has no TE_ columns).
    names_path = ckpt_dir / "target_names.txt"
    if names_path.exists():
        cell_types = [ln.strip() for ln in names_path.read_text().splitlines() if ln.strip()]
        cell_types = [c[3:] if c.startswith("TE_") else c for c in cell_types]
    else:
        cell_types = None  # fall back to positional names once output width is known

    df = pd.DataFrame(query)
    df["txID"] = range(len(df))

    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), f"duet_predict_{os.getpid()}.tsv")
    df.to_csv(tmp_path, sep="\t", index=False)

    try:
        datamodule = DataModule(cfg, dataset_path=tmp_path)
        datamodule.setup(stage="test")
        dataloader = DataLoader(datamodule.dataset, batch_size=batch_size,
                                num_workers=0, shuffle=False)

        preds = []
        iterator = tqdm(dataloader, desc="Predicting") if use_tqdm else dataloader
        with torch.no_grad():
            for batch in iterator:
                for k in batch:
                    batch[k] = batch[k].to(device)
                output, _ = model.model.predict(batch)
                preds.append(output.cpu().numpy())
        pred = np.concatenate(preds, axis=0)          # (N, n_celltypes)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if cell_types is None or len(cell_types) != pred.shape[1]:
        cell_types = [f"target_{j}" for j in range(pred.shape[1])]

    result = pd.DataFrame(pred, columns=cell_types)

    if target is None:
        return result
    if isinstance(target, str):
        if target not in result.columns:
            raise ValueError(f"Unknown cell type '{target}'. Available: {list(result.columns)}")
        return result[target].tolist()
    # list[str]
    missing = [t for t in target if t not in result.columns]
    if missing:
        raise ValueError(f"Unknown cell type(s) {missing}. Available: {list(result.columns)}")
    return result[list(target)]


def train(config_paths: list[str], override_configs: dict = None, device: str = "cuda"):
    """Train model using config file(s).

    Args:
        config_paths: List of config file paths (later configs override earlier ones)
        override_configs: Dict of config overrides in dot notation
        device: Device to train on ('cuda', 'cuda:0', 'cpu', etc.)
    """
    from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

    if isinstance(config_paths, str):
        config_paths = [config_paths]

    override_configs = override_configs or {}
    cfg, dict_cfg = load_cfgs(config_paths, override_configs)

    import time
    _t_start = time.time()

    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision('high')

    # Setup output directory
    os.makedirs(cfg.log_dir, exist_ok=True)
    out_dir = os.path.join(cfg.log_dir, cfg.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    OmegaConf.save(dict_cfg, os.path.join(out_dir, 'config.yaml'))

    # Parse device
    if device == "cpu":
        accelerator, devices = "cpu", 1
    else:
        accelerator = "gpu"
        devices = [int(device.split(":")[1])] if ":" in device else 1

    # Setup logger
    loggers = []
    if cfg.use_wandb:
        import wandb
        from pytorch_lightning.loggers import WandbLogger
        loggers.append(WandbLogger(
            name=cfg.exp_name,
            project=cfg.project_name,
            notes=cfg.notes,
            save_dir=cfg.log_dir,
            settings=wandb.Settings(
                _disable_stats=True,
                _disable_meta=True,
                save_code=False,
                disable_code=True,
            )
        ))
    else:
        from pytorch_lightning.loggers.logger import DummyLogger
        loggers.append(DummyLogger())

    # Setup callbacks
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=os.path.join(out_dir, 'ckpts'),
            every_n_epochs=cfg.trainer.save_epochs,
            save_weights_only=True,
            save_top_k=1,
            monitor="val/spearman",
            mode="max"
        ),
    ]
    if cfg.trainer.early_stopping:
        callbacks.append(EarlyStopping(
            monitor='val/spearman',
            patience=cfg.trainer.early_stopping,
            mode='max',
            min_delta=cfg.trainer.min_delta
        ))

    # Create trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        callbacks=callbacks,
        logger=loggers,
    )

    # Setup data and model
    datamodule = DataModule(cfg, dataset_path=cfg.dataset.train)
    model = Module(cfg, dict_cfg)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.fit(model, datamodule=datamodule)
    test_results = trainer.test(ckpt_path='best', datamodule=datamodule)

    # Copy best checkpoint to model.ckpt
    import shutil
    best_ckpt = trainer.checkpoint_callback.best_model_path
    if best_ckpt:
        shutil.copy(best_ckpt, os.path.join(out_dir, "model.ckpt"))

    # Save test metrics
    if test_results:
        OmegaConf.save(OmegaConf.create(test_results[0]), os.path.join(out_dir, "metrics.yaml"))

    # Save train/val/test indices
    import json
    indices = {
        "train": datamodule.train_indices.tolist(),
        "val": datamodule.val_indices.tolist(),
        "test": datamodule.test_indices.tolist(),
    }
    with open(os.path.join(out_dir, "indices.json"), "w") as f:
        json.dump(indices, f, default=int, indent=2)

    # Persist run stats (peak VRAM + runtime), independent of wandb.
    stats = {"exp_name": cfg.exp_name, "runtime_sec": round(time.time() - _t_start, 1)}
    try:
        if torch.cuda.is_available():
            stats["peak_vram_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
            stats["peak_vram_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 4)
            stats["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as e:
        logger.warning(f"Could not read peak VRAM (non-fatal): {e}")
    with open(os.path.join(out_dir, "run_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Finalize WandB and clean up the local run cache (frees Lustre inodes).
    if cfg.use_wandb:
        import wandb
        try:
            import shutil
            run_dir_to_clean = None
            if wandb.run is not None and wandb.run.dir:
                run_root = os.path.dirname(wandb.run.dir)
                if run_root and os.path.basename(run_root).startswith('run-'):
                    run_dir_to_clean = run_root
            wandb.finish()
            if run_dir_to_clean and os.path.isdir(run_dir_to_clean):
                shutil.rmtree(run_dir_to_clean, ignore_errors=True)
                logger.info(f"Cleaned up local wandb run dir: {run_dir_to_clean}")
        except Exception as e:
            logger.warning(f"WandB cleanup failed (non-fatal): {e}")

