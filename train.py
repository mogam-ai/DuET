#!/usr/bin/env python3
# Program:     DuET v1.0.0
# Author:      Sungho Lee, Jae-Won Lee, Jinhee Hong
# Affiliation: MOGAM Institute for Biomedical Research
# Contact:     sungho.lee@mogam.re.kr; jaewon.lee@mogam.re.kr; jinhee.hong@mogam.re.kr
# Citation:    TBD


import sys
import os
import torch
import pytorch_lightning as pl
import joblib
import wandb

from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from glob import glob
from loguru import logger
from omegaconf import OmegaConf

from models.module import Module
from data.datamodule import DataModule
from configs.config import parse_args, load_cfgs


config_list, override_configs = parse_args()
cfg, dict_cfg = load_cfgs(config_list, override_configs)

pl.seed_everything(cfg.seed, workers=True)
torch.set_float32_matmul_precision('high')

if not os.path.exists(cfg.log_dir):
  os.makedirs(cfg.log_dir, exist_ok=True)
out_dir = os.path.join(cfg.log_dir, cfg.exp_name)
os.makedirs(out_dir, exist_ok=True)
OmegaConf.save(dict_cfg, os.path.join(out_dir, 'config.yaml'))

loggers = []
if cfg.use_wandb:
  from pytorch_lightning.loggers import WandbLogger
  wdb_logger = WandbLogger(name    = cfg.exp_name,
                           project  = cfg.project_name,
                           notes    = cfg.notes,
                           save_dir = cfg.log_dir,
                           settings = wandb.Settings(_disable_stats=True)
                           )
  loggers.append(wdb_logger)
else:
  from pytorch_lightning.loggers.logger import DummyLogger
  loggers.append(DummyLogger())

# kfold test
current_fold = 0
max_fold = 1 if not cfg.aggregate_kfold else cfg.datamodule.kfold_test_k
all_results = {}
for current_fold in range(max_fold):
  
  scaler_prefix = f"split{current_fold}_" if max_fold != 1 else ""
  scaler_path = os.path.join(out_dir, f"{scaler_prefix}label_scaler.joblib")
  
  if cfg.aggregate_kfold:
    logger.info(f"K-fold test: {current_fold+1}/{max_fold}")
    cfg.datamodule.kfold_test_fold = current_fold
    ckpt_prefix = f"{current_fold:>0{len(str(max_fold))}}" 

  lr_monitor = LearningRateMonitor(logging_interval="step")
  checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(out_dir, 'ckpts'),
    filename=(f"split{current_fold}_" + "{epoch}-{step}") if cfg.aggregate_kfold else None,
    every_n_epochs=cfg.trainer.save_epochs,
    save_weights_only=True,
    save_top_k=1,
    monitor="val/spearman",
    mode="max",
    )
  callbacks = [lr_monitor, checkpoint_callback]
  
  if cfg.trainer.early_stopping is not None:
    early_stop_callback = EarlyStopping(
      monitor='val/spearman',
      patience=cfg.trainer.early_stopping,
      verbose=True,
      mode='max',
      min_delta=cfg.trainer.min_delta,
      )
    callbacks.append(early_stop_callback)

  max_epochs = cfg.trainer.max_epochs
  trainer = pl.Trainer(
    accelerator='gpu',
    strategy='ddp',
    max_epochs=max_epochs,
    callbacks=callbacks,
    logger=loggers,
  )
  
  label_scaler = None
  if (not cfg.trainer.run_fit) and (cfg.load_model_path is not None):
    if cfg.dataset.param.scale_label:
      label_scaler = joblib.load(cfg.load_scaler_path)
      logger.info(f"Loaded standard scaler from {cfg.load_scaler_path}.")
  datamodule = DataModule(cfg, dataset_path=cfg.dataset.train, scaler_obj=label_scaler)
  
  if not cfg.trainer.run_fit and cfg.load_model_path is not None:
    if cfg.dataset.param.scale_label:
      with open(cfg.load_scaler_path, "rb") as p:
        datamodule.label_scaler = joblib.load(p)
      datamodule.label_mean = datamodule.label_scaler.mean_[0]
      datamodule.label_scale = datamodule.label_scaler.scale_[0]
      logger.info(f"Loaded standard scaler. Mean: {datamodule.label_mean:.4f}, Scale: {datamodule.label_scale:.4f}")

  model = Module(cfg, dict_cfg, sweep=False,
                 label_mean=datamodule.label_mean,  # pass label mean and scale for inverse transform
                 label_scale=datamodule.label_scale)
  
  if cfg.load_model_path is not None:
    model = Module.load_from_checkpoint(
      cfg.load_model_path,
      cfg=cfg, 
      dict_cfg=dict_cfg,
      strict=cfg.load_model_strict,
      label_mean=datamodule.label_mean,
      label_scale=datamodule.label_scale
      )

  if cfg.trainer.run_fit:
    trainer.fit(model, datamodule=datamodule)
    if cfg.dataset.param.scale_label:
      joblib.dump(datamodule.label_scaler, scaler_path)
      with open(out_dir + "/mean_sd.txt", "w") as f:
        print(datamodule.label_scaler.mean_[0], datamodule.label_scaler.scale_[0], file=f)
      
  if cfg.trainer.run_test:
    test_data_paths = glob(cfg.dataset.test)
    for test_data_path in test_data_paths:
      print(f"{test_data_path}")
      if not cfg.datamodule.do_kfold_test:  # if kfold test, test dataset is already set
        datamodule = DataModule(cfg, dataset_path=test_data_path,
                                scaler_obj=datamodule.label_scaler)
      if cfg.trainer.run_fit:
        trainer.test(ckpt_path='best', datamodule=datamodule)
        if cfg.aggregate_kfold:
          for k, v in model.result.items():  # save results for aggregation
            all_results.setdefault(k, []).append(v)
      else: # run pickled model
        trainer.test(model=model, datamodule=datamodule)

      if trainer.interrupted:  # in case of KeyboardInterrupt
        break

# aggregate kfold results
if cfg.aggregate_kfold:
  logger.info("Kfold test: aggregating results")
  from pprint import pformat
  logger.info(f'all_results:\n{pformat(all_results)}')
  for k, v in all_results.items():
    logger.info(f"{k} mean: {np.mean(v)}, std: {np.std(v)}")
  # log the aggregated results. results_dict is a dictionary of lists.
  # We log the mean and std of the results.
  if cfg.use_wandb:
    import numpy as np
    for k, v in all_results.items():
        values = np.array(v)
        wdb_logger.experiment.log({f'{k}_mean': np.mean(values)})
        wdb_logger.experiment.log({f'{k}_std': np.std(values)})
  logger.info("Kfold test: done")
