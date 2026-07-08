# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from loguru import logger
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from abc import ABCMeta, abstractmethod

from ..configs.config import Config
from ..data.sequence_feature_store import find_matching_columns


class Module(pl.LightningModule):
  """LightningModule wrapping a DuET model with training, evaluation, and metric logging.

  Selects the model from cfg.model.name ('duet' = multi-cell-type, 'duet_singletask'
  = single scalar) and configures the optimizer/scheduler from cfg.
  """
  def __init__(self, 
               cfg: Config,
               dict_cfg: dict,
    ):
    
    super().__init__()
    self.save_hyperparameters(dict_cfg)
    self.cfg = cfg
    self.optim_cfg = self.cfg.optimizer
    self.sched_cfg = self.cfg.scheduler

    self.test_name = None

    # model setup
    self.model_cfg = self.cfg.model
    self.loss_cfg = self.cfg.loss
    
    if cfg.dataset.param.use_sequence_feature:
      if 'sequence_feature_cols' in cfg.dataset.param:
        sequence_feature_cols = find_matching_columns(cfg.dataset.param.sequence_feature_cols, 
                                                      fname=cfg.dataset.param.sequence_feature_path)
      else:
        sequence_feature_cols = find_matching_columns("*", 
                                                      fname=cfg.dataset.param.sequence_feature_path)
    sequence_feature_size = len(sequence_feature_cols) if cfg.dataset.param.use_sequence_feature else 0
    
    if self.model_cfg.name == 'duet':
      from .duet import DuetMultiModel
      model_params = dict(self.model_cfg.param)
      model_params.setdefault('use_sequence_feature', cfg.dataset.param.use_sequence_feature)
      model_params.setdefault('sequence_feature_size', sequence_feature_size)
      model_params.setdefault('use_codon_encoding', cfg.dataset.param.use_codon_encoding)
      self.model = DuetMultiModel(**model_params, loss_cfg=self.loss_cfg)
    elif self.model_cfg.name == 'duet_singletask':
      from .duet import DuetSingletaskModel
      model_params = dict(self.model_cfg.param)
      model_params.setdefault('use_sequence_feature', cfg.dataset.param.use_sequence_feature)
      model_params.setdefault('sequence_feature_size', sequence_feature_size)
      model_params.setdefault('use_codon_encoding', cfg.dataset.param.use_codon_encoding)
      self.model = DuetSingletaskModel(**model_params, loss_cfg=self.loss_cfg)
    else:
      raise NotImplementedError(f"model {self.model_cfg.name} is not implemented")

  def on_train_start(self):
    super().on_train_start()
    from pytorch_lightning.utilities.model_summary import LayerSummary
    self.log("model_size", LayerSummary(self.model).num_parameters, sync_dist=True)
    logger.info(f"Now training: {self.cfg.exp_name}")

  def training_step(self, batch, batch_idx):
    _, loss = self.model(batch)
    self.log('train/loss', loss.item())
    return loss

  def init_pcc(self):
    self.y = []
    self.yhat = []

  def step_pcc(self, y, yhat):
    self.y.append(y.detach().cpu().numpy())
    self.yhat.append(yhat.detach().cpu().numpy())

  def prepare_calc_pcc(self):
    yhat = np.concatenate(self.yhat)
    y = np.concatenate(self.y)
    # Flatten multi-target outputs (e.g., multi-task model)
    is_multitask = y.ndim > 1
    if is_multitask:
      yhat = yhat.flatten()
      y = y.flatten()
    # Mask NaN values
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    # NaN is expected for multi-task (missing cell-type labels); only warn for single-task.
    if np.sum(~mask) > 0:
      if not is_multitask:
        logger.warning(f"Number of NaN values: {np.sum(~mask)}")
      y = y[mask]
      yhat = yhat[mask]

    return y, yhat

  def calc_pcc(self):
    y, yhat = self.prepare_calc_pcc()

    result = {}
    # Need >=2 valid samples for pearson/spearman; return sentinel metrics otherwise.
    if len(y) < 2:
      logger.warning(f"calc_pcc: only {len(y)} valid sample(s) after NaN masking; "
                     f"returning sentinel metrics.")
      return {'pearson': 0.0, 'spearman': 0.0, 'r2': float('-inf'),
              'rmse': float('inf'), 'mae': float('inf')}
    r, pval = stats.pearsonr(y, yhat)
    rho, pval = stats.spearmanr(y, yhat)
    result['pearson'] = r
    result['spearman'] = rho
    result['r2'] = r2_score(y, yhat)
    result['rmse'] = np.sqrt(mean_squared_error(y, yhat))
    result['mae'] = mean_absolute_error(y, yhat)
    return result

  def on_validation_epoch_start(self):
    self.init_pcc()

  def validation_step(self, batch, batch_idx):
    with torch.no_grad():
      yhat, loss = self.model(batch)
      self.log('val/loss', loss.item(), on_epoch=True, batch_size=self.cfg.datamodule.batch_size, sync_dist=True)
      self.step_pcc(batch['y'], yhat)
    return loss

  def on_validation_epoch_end(self):
    result = self.calc_pcc()
    for k, v in result.items():
      self.log(f'val/{k}', v, sync_dist=True)

  def setup_test(self, test_name:str):
    self.test_name = test_name
    self.init_pcc()

  def on_test_epoch_start(self):
    self.init_pcc()
    self.test_name = self.trainer.datamodule.get_dataset_name()

  def test_step(self, batch, batch_idx):
    with torch.no_grad():
      pred, _ = self.model.predict(batch)
      self.step_pcc(batch['y'], pred)

  def on_test_epoch_end(self):
    if self.test_name is not None:
      with open(os.path.join(self.cfg.log_dir, self.cfg.exp_name, f"{self.test_name}_prediction.tsv"), "w") as output:
          print("y", "pred_y", sep="\t", file=output)
          for y, yhat in zip(*self.prepare_calc_pcc()):
              print(y, yhat, sep="\t", file=output)
      result = self.calc_pcc()
      for k, v in result.items():
        if self.loss_cfg.ignore_test_name:
          self.log(k, v, sync_dist=True)
        else:
          self.log(f'{self.test_name}_{k}', v, sync_dist=True)
      self.result = result

      # Main DuET is multi-cell-type: dump per-cell-type metrics & full prediction
      # matrix to TSV. wandb logging above keeps pooled metrics; this is local-only.
      # The single-task variant ('duet_singletask') produces scalar output -> skip.
      if self.model_cfg.name != 'duet_singletask':
        self._dump_multitask_metrics()

  def _dump_multitask_metrics(self):
    """Write per-cell-type metric TSV and (genes x celltypes) prediction matrix.

    Called only in multitask test path. Does not log to wandb; pooled metrics
    are already logged by the surrounding on_test_epoch_end.
    """
    yhat = np.concatenate(self.yhat)  # (N, n_targets)
    y = np.concatenate(self.y)        # (N, n_targets) with NaN

    # Resolve cell-type column names from the underlying dataset.
    dm = self.trainer.datamodule
    base_ds = getattr(dm, '_dataset', None) or dm.dataset
    target_names = getattr(base_ds, 'target_names',
                           [f'target_{j}' for j in range(y.shape[1])])

    out_dir = os.path.join(self.cfg.log_dir, self.cfg.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # 1) per-cell-type metrics
    metric_path = os.path.join(out_dir, f"{self.test_name}_celltype_metrics.tsv")
    with open(metric_path, "w") as f:
      print("celltype", "n", "pearson", "spearman", "r2", "rmse", "mae", sep="\t", file=f)
      for j, name in enumerate(target_names):
        m = ~np.isnan(y[:, j]) & ~np.isnan(yhat[:, j])
        n = int(m.sum())
        if n < 10:
          print(name, n, "nan", "nan", "nan", "nan", "nan", sep="\t", file=f)
          continue
        yj, pj = y[m, j], yhat[m, j]
        r, _ = stats.pearsonr(yj, pj)
        rho, _ = stats.spearmanr(yj, pj)
        print(name, n,
              f"{r:.6f}", f"{rho:.6f}",
              f"{r2_score(yj, pj):.6f}",
              f"{np.sqrt(mean_squared_error(yj, pj)):.6f}",
              f"{mean_absolute_error(yj, pj):.6f}",
              sep="\t", file=f)
    logger.info(f"Wrote per-cell-type metrics: {metric_path}")

    # 2) full prediction matrix: 2 rows per gene (y, yhat), columns = cell types.
    pred_path = os.path.join(out_dir, f"{self.test_name}_prediction_matrix.tsv")
    with open(pred_path, "w") as f:
      print("kind\tgene_idx\t" + "\t".join(target_names), file=f)
      for i in range(y.shape[0]):
        print("y", i, *(f"{v}" for v in y[i]), sep="\t", file=f)
        print("yhat", i, *(f"{v}" for v in yhat[i]), sep="\t", file=f)
    logger.info(f"Wrote multitask prediction matrix: {pred_path}")

  def configure_optimizers(self):
    if self.optim_cfg.name.lower() == 'adamw':
      weight_decay = 0.01
      if self.optim_cfg.param is not None:
        weight_decay = self.optim_cfg.param.get('weight_decay', weight_decay)
      if self.optim_cfg.layerwise_lr is not None:
        optimizer = torch.optim.AdamW([
          {"params": self.model.layerwise_lr[name], "lr": lr, "name": name}
           for name, lr in self.optim_cfg.layerwise_lr.items()
          ], lr=self.optim_cfg.lr, weight_decay=weight_decay)
      else:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.optim_cfg.lr, weight_decay=weight_decay)
    else:
      raise NotImplementedError(f"optimizer {self.optim_cfg.name} is not implemented")

    if self.sched_cfg.name == 'ReduceLROnPlateau':
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
          optimizer,
          mode="min",
          factor=self.sched_cfg.param.factor,
          patience=self.sched_cfg.param.patience,
          min_lr=self.sched_cfg.param.min_lr,
      )
    elif self.sched_cfg.name == 'CosineAnnealingLR':
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer,
          T_max=self.sched_cfg.param.T_max,
      )
    else:
      raise NotImplementedError(f"scheduler {self.sched_cfg.name} is not implemented")

    return {
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "monitor": "train/loss",
    }


class Model(nn.Module, metaclass=ABCMeta):
  """Abstract base for DuET models. Subclasses implement predict() and forward()."""

  def __init__(self):
    super().__init__()

  @abstractmethod
  def predict(self, batch):
    """Return (yhat, internal_pred_for_loss)."""
    pass

  @abstractmethod
  def forward(self, batch):
    """Return (yhat, loss). yhat may be unprocessed and require a converter."""
    pass