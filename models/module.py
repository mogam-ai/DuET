# Program:     DuET v1.0.0
# Author:      Sungho Lee, Jae-Won Lee, Jinhee Hong
# Affiliation: MOGAM Institute for Biomedical Research
# Contact:     sungho.lee@mogam.re.kr; jaewon.lee@mogam.re.kr; jinhee.hong@mogam.re.kr
# Citation:    TBD


import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from loguru import logger
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from abc import ABCMeta, abstractmethod

from configs.config import Config
from data.sequence_feature_store import find_matching_columns


class Module(pl.LightningModule):
  """Class holding the model components for training and evaluation.
  
  This class is responsible for managing the training and evaluation steps, 
  as well as logging metrics. 
  It also handles the configuration of the model, optimizer, and scheduler 
  based on the provided configuration dataclass.
  
  Attributes:
    cfg (Config): Configuration dataclass containing model, optimizer, scheduler, and other settings.
    model (nn.Module): The model to be trained and evaluated.
    optim_cfg (OptimizerConfig): Configuration for the optimizer.
    sched_cfg (SchedulerConfig): Configuration for the learning rate scheduler.
    model_cfg (ModelConfig): Configuration for the model architecture and parameters.
    loss_cfg (LossConfig): Configuration for the loss function.
    output_converter (callable): Function to convert model outputs to the original scale for evaluation.
    test_name (str): Name of the test set for logging purposes. 
  """
  def __init__(self, 
               cfg: Config,
               dict_cfg: dict,
               sweep: bool=False,
               label_mean: float=0.0,
               label_scale: float=1.0,
    ):
    """Initialize Module instance.
    
    Args:
      cfg (Config): Configuration dataclass instance.
      dict_cfg (Dict): Nested dict object holding the configuration for logging.
      sweep (bool): Whether the training is from wandb sweep.
                    If true, disable hyperparameter logging.
      label_mean (float): Mean of the labels for standard scaling.
      label_scale (float): Scale of the labels for standard scaling.
    """
    
    super().__init__()
    if not sweep: 
      self.save_hyperparameters(dict_cfg)
    self.cfg = cfg
    self.optim_cfg = self.cfg.optimizer
    self.sched_cfg = self.cfg.scheduler

    self.test_name = None
    self.output_converter = None

    # model setup
    self.model_cfg = self.cfg.model
    self.model_cfg.param.label_mean = float(label_mean)  # inject standard scaling parameters
    self.model_cfg.param.label_scale = float(label_scale)
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
      from models.duet import DuetModel
      self.model = DuetModel(**self.model_cfg.param,
                                 utr_input_channels=cfg.dataset.param.utr_channel_size,
                                 cds_input_channels=cfg.dataset.param.cds_channel_size,
                                 use_sequence_feature=cfg.dataset.param.use_sequence_feature,
                                 sequence_feature_size=sequence_feature_size,
                                 utr_seq_size=cfg.dataset.param.utr_seq_size,
                                 cds_seq_size=cfg.dataset.param.cds_seq_size,
                                 log_label=cfg.dataset.param.log_label,
                                 scale_label=cfg.dataset.param.scale_label,
                                 loss_cfg=self.loss_cfg)
    elif self.model_cfg.name == 'translatelstm':
      from models.translatelstm import LstmModel
      self.model = LstmModel(**self.model_cfg.param,
                             sequence_feature_size=sequence_feature_size,
                             loss_cfg=self.loss_cfg)
    elif self.model_cfg.name == 'translatelstm_dual':
      from models.translatelstm import LstmDualModel
      self.model = LstmDualModel(**self.model_cfg.param,
                             sequence_feature_size=sequence_feature_size,
                             loss_cfg=self.loss_cfg)
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
    # calculate r squared
    yhat = np.concatenate(self.yhat)
    y = np.concatenate(self.y)
    if self.output_converter is not None:
      yhat = self.output_converter(yhat)
    # Mask NaN values
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    # if num of nan > 0, report
    if np.sum(~mask) > 0:
      logger.warning(f"Number of NaN values: {np.sum(~mask)}")
      y = y[mask]
      yhat = yhat[mask]

    return y, yhat

  def calc_pcc(self, save_fig_path=None):
    y, yhat = self.prepare_calc_pcc()

    result = {}
    r, pval = stats.pearsonr(y, yhat)
    rho, pval = stats.spearmanr(y, yhat)
    result['pearson'] = r
    result['spearman'] = rho
    result['r2'] = r2_score(y, yhat)
    result['rmse'] = np.sqrt(mean_squared_error(y, yhat))
    result['mae'] = mean_absolute_error(y, yhat)
    
    if save_fig_path is not None:
      import matplotlib.pyplot as plt
      max_scatter = 5000
      max_scatter = min(max_scatter, len(y))
      if max_scatter < 10:
        logger.warning(f"Too few samples to plot scatter plot: {max_scatter}")
        return result

      # random selection
      idx = np.random.choice(len(y), max_scatter, replace=False)
      plt.figure(figsize=(6,6))
      plt.scatter(y[idx], yhat[idx], s=2, alpha=0.5)
      plt.xlabel("True")
      plt.ylabel("Predicted")
      plt.title(f"Pearson Correlation: {r:.4f}, R^2: {result['r2']:.4f}")
      plt.savefig(save_fig_path)
      plt.close()
    return result

  def on_validation_epoch_start(self):
    self.init_pcc()
    self.output_converter = self.trainer.datamodule.get_output_converter()

  def validation_step(self, batch, batch_idx):
    with torch.no_grad():
      yhat, loss = self.model(batch)
      self.log('val/loss', loss.item(), on_epoch=True, batch_size=self.cfg.datamodule.batch_size, sync_dist=True)
      if self.model_cfg.task == 'finetune':
        self.step_pcc(batch['y'], yhat)
    return loss

  def on_validation_epoch_end(self):
    if self.model_cfg.task == 'finetune':
      result = self.calc_pcc()
      for k, v in result.items():
        self.log(f'val/{k}', v, sync_dist=True)

  def setup_test(self, test_name:str):
    self.test_name = test_name
    self.init_pcc()

  def on_test_epoch_start(self):
    self.init_pcc()
    self.test_name = self.trainer.datamodule.get_dataset_name()
    self.output_converter = self.trainer.datamodule.get_output_converter()
    if self.cfg.aggregate_kfold:
      # if aggregate kfold, use the first fold name
      self.test_name = f"fold{self.cfg.datamodule.kfold_test_fold + 1}"

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
      save_fig_path = None
      if self.cfg.trainer.save_fig:
        save_fig_path=os.path.join(self.cfg.log_dir, self.cfg.exp_name, f"{self.test_name}_pcc.png")
      result = self.calc_pcc(save_fig_path=save_fig_path)
      for k, v in result.items():
        if self.loss_cfg.ignore_test_name:
          self.log(k, v, sync_dist=True)
        else:
          self.log(f'{self.test_name}_{k}', v, sync_dist=True)
      self.result = result

  def configure_optimizers(self):
    if self.optim_cfg.name.lower() == 'adamw':
      # weight decay setup
      weight_decay = 0.01 # default value
      if self.optim_cfg.param is not None:
        weight_decay = self.optim_cfg.param.get('weight_decay', weight_decay)
      # apply layerwise lr
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
  """
  Abstract class for Custom UTR model.
  Below methods help to log the loss and calculate the PCC.
  """

  def __init__(self):
    super().__init__()

  @abstractmethod
  def predict(self, batch):
    """
    Predict should return (yhat, internal_pred_for_loss).

    Args:
        batch (dict): Input dictionary.

    Returns:
        tuple: (yhat, internal_pred_for_loss)
    """
    pass

  @abstractmethod
  def forward(self, batch):
    """
    Forward should return (yhat, loss).
    yhat could be unprocessed and may require a converter.

    Args:
        batch (dict): Input dictionary.

    Returns:
        tuple: (yhat, loss)
    """
    pass