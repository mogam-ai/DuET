# Program:     DuET v1.0.0
# Author:      Sungho Lee, Jae-Won Lee, Jinhee Hong
# Affiliation: MOGAM Institute for Biomedical Research
# Contact:     sungho.lee@mogam.re.kr; jaewon.lee@mogam.re.kr; jinhee.hong@mogam.re.kr
# Citation:    TBD


import argparse
import yaml

from loguru import logger
from dataclasses import dataclass
from typing import List, Optional
from omegaconf import OmegaConf, DictConfig


@dataclass
class ModelConfig:
  name: str
  param: DictConfig
  task: str = 'finetune'


@dataclass
class OptimizerConfig:
  name: str
  lr: str
  layerwise_lr: Optional[DictConfig] = None
  param: Optional[DictConfig] = None


@dataclass
class SchedulerConfig:
  name: str
  param: DictConfig


@dataclass
class LossConfig:
  name: str
  ignore_test_name: bool = False
  param: Optional[DictConfig] = None


@dataclass
class DatasetConfig:
  name: str # dataset type
  param: DictConfig
  path: Optional[str] = None # adaptive setup for different datasets
  train: Optional[str] = None # not used - only for compatibility
  test: Optional[str] = None # not used - only for compatibility


@dataclass
class DataModuleConfig:
  batch_size: int
  num_workers: int
  dataset: DatasetConfig
  sample_per_epoch: Optional[int] = None
  shuffle_train: bool = True
  train_size: float = 0.9

  val_random_split: bool = True # If False, use fixed indices for train and val
  val_split_seed: int = 42

  do_kfold_test: bool = False # If True, do k-fold test. also dataset.test is ignored
  kfold_test_k: int = 10 # number of folds
  kfold_test_fold: int = 0 # current fold, 0-indexed
  kfold_test_seed: int = 42 # seed for k-fold test

  def __post_init__(self):
    # ensure kfold_test_fold and kfold_test_k are integers
    self.kfold_test_fold = int(self.kfold_test_fold)
    self.kfold_test_k = int(self.kfold_test_k)
    assert self.kfold_test_fold < self.kfold_test_k, "kfold_test_fold should be less than kfold_test_k"


@dataclass
class TrainerConfig:
  max_epochs: int
  save_epochs: int
  early_stopping: Optional[int] = None
  min_delta: Optional[float] = 0.0
  save_fig: bool = False
  run_fit: bool = True
  run_test: bool = True


@dataclass
class Config:
  exp_name: str
  notes: str
  use_wandb: bool
  project_name: str
  debug: bool
  log_dir: str
  seed: int

  model: ModelConfig
  dataset: DatasetConfig
  datamodule: DataModuleConfig
  optimizer: OptimizerConfig
  scheduler: SchedulerConfig
  loss: LossConfig
  trainer: TrainerConfig

  aggregate_kfold: bool = False
  load_model_strict: bool = True
  load_model_path: Optional[str] = None
  load_scaler_path: Optional[str] = None
  base_config: Optional[str] = None # for sweep

  def __post_init__(self):
    self.model = ModelConfig(**self.model)
    self.dataset = DatasetConfig(**self.dataset)
    self.datamodule = DataModuleConfig(**self.datamodule,
                                          dataset=self.dataset)
    self.optimizer = OptimizerConfig(**self.optimizer)
    self.scheduler = SchedulerConfig(**self.scheduler)
    self.loss = LossConfig(**self.loss)
    self.trainer = TrainerConfig(**self.trainer)


def parse_args():
  '''
  Parse arguments from command line

  Returns:
    config_list: list of config files
    load_model_path: path to load model
  '''

  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, nargs='+', default=None)
  parser.add_argument('--load_model_path', type=str, default=None, help='load model')
  parser.add_argument('--load_scaler_path', type=str, default=None, help='load standard scaler')
  parser.add_argument('--debug', action='store_true', help='Force set use_wandb=False & debug=True')
  parser.add_argument('--override-configs', '--override', type=str, nargs='+', default=[],
                      help='override configs. ex) exp_name=foo dataset.param.train=P1.tsv debug=False model.param.block_sizes=[1,2,3]')

  args = parser.parse_args()
  config_list = args.config
  override_configs = {a:yaml.safe_load(b) for a,b in (k.split("=") for k in args.override_configs)}
  if args.load_model_path:
    override_configs["load_model_path"] = args.load_model_path
  if args.load_scaler_path:
    override_configs["load_scaler_path"] = args.load_scaler_path
  if args.debug:
    override_configs["debug"] = True
    override_configs["use_wandb"] = False

  if override_configs:
    from pprint import pformat
    logger.info(f"Overriding following parameters:\n{pformat(override_configs)}")

  return config_list, override_configs


def load_cfgs(cfgs: List[str], 
              override_configs: Optional[dict]=None,
              sweep_config: Optional[dict]=None,
              compat_xref: Optional[dict]=None) -> tuple[Config, dict]:
  """Load a list of config files into Config and dictionary objects.
  
  Args:
    cfgs: list of config files
    override_configs: dictionary of override config(s)
    sweep_config: wandb sweep config
  Returns:
    UTRConfig: merged config
    dict: merged config as a dictionary
  """
  configs = [OmegaConf.load(cfg) for cfg in cfgs]

  if sweep_config is not None:
    configs.append(OmegaConf.create(dict(sweep_config)))
  
  raw_config = OmegaConf.merge(*configs)
  if override_configs is not None:
    for key, value in override_configs.items():
       OmegaConf.update(raw_config, key, value) # replaces raw_config[key] = value for deep update
  
  # Compatibility handling: rename or delete old keys at runtime
  if compat_xref is not None:
    for old_key, new_key in compat_xref.items():
      if OmegaConf.select(raw_config, old_key) is not None:
        if new_key is None:
          delete_nested_key(raw_config, old_key)
        elif OmegaConf.select(raw_config, new_key) is None:
          OmegaConf.update(raw_config, new_key, OmegaConf.select(raw_config, old_key))
          delete_nested_key(raw_config, old_key)

  dict_config = OmegaConf.to_container(raw_config, resolve=True)
  return Config(**raw_config), dict_config


def delete_nested_key(conf: DictConfig, key_string: str):
    """
    Delete an item from a nested DictConfig using a dot-separated key string.
    
    Args:
        conf (DictConfig): OmegaConf's DictConfig object.
        key_string (str): The path of the item to delete (e.g., 'model.params.learning_rate').
    """
    keys = key_string.split('.')
    parent_keys = keys[:-1]
    final_key = keys[-1]
    
    parent_obj = conf
    for key in parent_keys:
        parent_obj = parent_obj[key]
        
    if final_key in parent_obj:
        del parent_obj[final_key]
