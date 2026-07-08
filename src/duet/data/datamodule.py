# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

import os
import numpy as np

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Optional
from loguru import logger

from ..configs.config import Config
from .duet_singletask import DuetSingletaskDataset


class DataModule(LightningDataModule):
  """Handles dataset loading, train/val/test splitting, and dataloaders.

  Selects the dataset from cfg.datamodule.dataset.name ('duet' = multitask,
  'duet_singletask' = single-target) and optionally applies k-fold and label scaling.
  """
  def __init__(self,
               cfg: Config,
               dataset_path: str):
    super().__init__()

    self.cfg = cfg.datamodule

    self.cfg.dataset.path = dataset_path  # path injection to support multiple datasets

    self.model_params = cfg.model.param
    self.dataset_name = os.path.basename(dataset_path).split(".")[0]
    self.exp_name = cfg.exp_name
    self.out_dir = os.path.join(cfg.log_dir, os.path.basename(cfg.exp_name))

  def setup(self, stage: Optional[str]=None):
    if self.cfg.dataset.name == 'duet':
      from .duet import DuetDataset
      self.dataset = DuetDataset(data_path=self.cfg.dataset.path,
                                           **self.cfg.dataset.param)
    elif self.cfg.dataset.name == 'duet_singletask':
      self.dataset = DuetSingletaskDataset(data_path=self.cfg.dataset.path,
                                  encoder=self.model_params.get('encoder', 'dual'),
                                  use_start_context=self.model_params.get('use_start_context', False),
                                  **self.cfg.dataset.param)
    else:
      raise NotImplementedError(f"dataset {self.cfg.dataset.name} is not implemented")

    if stage == 'fit':
      all_idx = np.arange(len(self.dataset))

      # Step 1: kfold split -> absolute indices
      if self.cfg.do_kfold_test:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.cfg.kfold_test_k,
                   shuffle=True,
                   random_state=self.cfg.kfold_test_seed)
        trainval_idx, test_idx = list(kf.split(all_idx))[self.cfg.kfold_test_fold]
        logger.debug(f"K-fold test: {self.cfg.kfold_test_fold}-th fold")
        logger.debug(f"Original dataset size: {len(all_idx)}")
      else:
        trainval_idx = all_idx
        test_idx = np.array([], dtype=int)

      # Step 2: train/val split -> absolute indices
      if self.cfg.val_random_split:
        rng = np.random.RandomState(self.cfg.val_split_seed)
        trainval_idx = rng.permutation(trainval_idx)
      n_train = int(len(trainval_idx) * self.cfg.train_size)
      train_idx = trainval_idx[:n_train]
      val_idx = trainval_idx[n_train:]

      # Store absolute indices
      self.train_indices = train_idx
      self.val_indices = val_idx
      self.test_indices = test_idx

      logger.debug(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")

      # Step 3: single-layer Subsets from original dataset
      self.train_dataset = DatasetWrapper(Subset(self.dataset, train_idx), self.cfg.sample_per_epoch)
      self.val_dataset = Subset(self.dataset, val_idx)
      self.test_dataset = Subset(self.dataset, test_idx)

    if stage == 'test':
      if self.cfg.do_kfold_test:
        logger.debug(f"K-fold test: skipping test dataset setup")
      else:
        self.test_dataset = self.dataset

  def train_dataloader(self):
    return DataLoader(self.train_dataset,
                      batch_size=self.cfg.batch_size,
                      num_workers=self.cfg.num_workers,
                      shuffle=self.cfg.shuffle_train,
                      drop_last=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset,
                      batch_size=self.cfg.batch_size,
                      num_workers=self.cfg.num_workers,
                      shuffle=False)

  def test_dataloader(self):
    return DataLoader(self.test_dataset,
                      batch_size=self.cfg.batch_size,
                      num_workers=self.cfg.num_workers,
                      shuffle=False)

  def get_dataset_name(self) -> str:
    """Return dataset name. This is used for test time logging."""
    return self.dataset_name

class DatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, samples_per_epoch: Optional[int] = None):
        self.dataset = dataset
        self.samples_per_epoch = len(self.dataset)
        if samples_per_epoch is not None and samples_per_epoch > 1:
          self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        if index >= self.samples_per_epoch:
            raise IndexError("Index out of range")
        # Use modulo to ensure the index wraps around the dataset length
        actual_index = index % len(self.dataset)
        return self.dataset[actual_index]
