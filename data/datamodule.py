# Program:     DuET v1.0.0
# Author:      Sungho Lee, Jae-Won Lee, Jinhee Hong
# Affiliation: MOGAM Institute for Biomedical Research
# Contact:     sungho.lee@mogam.re.kr; jaewon.lee@mogam.re.kr; jinhee.hong@mogam.re.kr
# Citation:    TBD


import os
import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.preprocessing import StandardScaler
from typing import Optional
from loguru import logger

from configs.config import Config
from data.duet import DuetDataset


class DataModule(LightningDataModule):
  """Class for handling data loading and preparation for training and evaluation.
  
  This class is responsible for setting up the dataset, 
  splitting it into training, validation, and test sets, 
  and providing data loaders for each of these sets during training and evaluation.
  
  Attributes:
    cfg (Config): Configuration object containing parameters for data loading and processing.
    dataset (Dataset): The complete dataset loaded from the specified path.
    train_dataset (Dataset): Subset of the dataset used for training.
    val_dataset (Dataset): Subset of the dataset used for validation.
    test_dataset (Dataset): Subset of the dataset used for testing.
    label_scaler (StandardScaler): Scaler object for normalizing labels if required.
    label_mean (float): Mean value of the labels used for scaling.
    label_scale (float): Scale value of the labels used for scaling.
  """
  def __init__(self, 
               cfg: Config, 
               dataset_path: str, 
               scaler_obj: StandardScaler=None):
    """Initialize DataModule instance.
    
    Args:
      cfg (Config): Configuration object containing parameters for data loading and processing.
      dataset_path (str): Path to the dataset file.
      scaler_obj (StandardScaler, optional): Pre-fitted scaler object for label normalization. 
                                             Defaults to None.
    """
    super().__init__()

    self.cfg = cfg.datamodule
    
    self.cfg.dataset.path = dataset_path  # path injection to support multiple datasets

    self.model_params = cfg.model.param
    self.dataset_name = os.path.basename(dataset_path).split(".")[0]
    self.exp_name = cfg.exp_name
    self.out_dir = os.path.join(cfg.log_dir, os.path.basename(cfg.exp_name))
    
    self.label_scaler = None
    self.label_mean = 0.0
    self.label_scale = 1.0

    self.scale_label = cfg.dataset.param.scale_label
    if self.scale_label and scaler_obj is not None:
      self.label_scaler = scaler_obj
      self.label_mean = scaler_obj.mean_[0]
      self.label_scale = scaler_obj.scale_[0]

  def setup(self, stage: Optional[str]=None):
    if self.cfg.dataset.name == 'duet':
      self.dataset = DuetDataset(data_path=self.cfg.dataset.path,
                                  encoder=self.model_params.get('encoder', 'dual'),
                                  use_start_context=self.model_params.get('use_start_context', False),
                                  **self.cfg.dataset.param)
    elif self.cfg.dataset.name == 'translatelstm':
      from data.translatelstm import LstmDataset
      self.dataset = LstmDataset(data_path=self.cfg.dataset.path,
                                 **self.cfg.dataset.param)
    else:
      raise NotImplementedError(f"dataset {self.cfg.dataset.name} is not implemented")

    if stage == 'fit':  # split train and test
      
      self._dataset = self.dataset
      if self.cfg.do_kfold_test:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.cfg.kfold_test_k,
                   shuffle=True,
                   random_state=self.cfg.kfold_test_seed)
        
        train_idx, test_idx = list(kf.split(self.dataset.data))[self.cfg.kfold_test_fold]
        self.dataset = Subset(self._dataset, train_idx)
        self.test_dataset = Subset(self._dataset, test_idx)

        logger.debug(f"K-fold test: {self.cfg.kfold_test_fold}-th fold")
        logger.debug(f'Original dataset size: {len(self._dataset)}')
        logger.debug(f"Train size: {len(self.dataset)}")
        logger.debug(f"Test size: {len(self.test_dataset)}")

      if self.cfg.val_random_split:
        self.train_dataset, self.val_dataset = random_split(
          dataset = self.dataset,
          lengths = [self.cfg.train_size, 1 - self.cfg.train_size],
          generator = torch.Generator().manual_seed(self.cfg.val_split_seed))
      else:  # fixed indices for train and val
        train_size = int(len(self.dataset) * self.cfg.train_size)
        self.train_dataset = Subset(self.dataset, range(train_size))
        self.val_dataset = Subset(self.dataset, range(train_size, len(self.dataset)))
      self.train_dataset = DatasetWrapper(self.train_dataset, self.cfg.sample_per_epoch)
      
      if self.scale_label:
        # Subset of subset; indices needs to be mapped to 'real' dataset indices
        if self.cfg.do_kfold_test:
            train_idx_mapped = self.dataset.indices[self.train_dataset.dataset.indices]
            val_idx_mapped = self.dataset.indices[self.val_dataset.indices]
        else:
            train_idx_mapped = self.train_dataset.dataset.indices
            val_idx_mapped = self.val_dataset.indices
        self.train_idx_mapped = train_idx_mapped
        self.val_idx_mapped = val_idx_mapped
      
        if self.label_scaler is None: 
          self.label_scaler = StandardScaler()
          self.label_scaler.fit(self._dataset.data['label'][train_idx_mapped].values.reshape(-1, 1))
          self.label_mean = torch.tensor(self.label_scaler.mean_, dtype=torch.float32)
          self.label_scale = torch.tensor(self.label_scaler.scale_, dtype=torch.float32)
          logger.info("Scaler fitted with train dataset.")
        else:
          logger.info("Using loaded scaler.")
        logger.info(f"Scaler mean: {self.label_mean.item():.4f}, Scale: {self.label_scale.item():.4f}\n")
        
        self._scale_label(self._dataset.data, train_idx_mapped)
        self._scale_label(self._dataset.data, val_idx_mapped)
        
        if self.cfg.do_kfold_test:
          self._scale_label(self._dataset.data, test_idx)

    if stage == 'test':
      
      if self.cfg.do_kfold_test:
        logger.debug(f"K-fold test: skipping test dataset setup")
        pass # already handled in 'fit'
      else:
        self.test_dataset = self.dataset
        if self.scale_label:
          self._scale_label(self.test_dataset.data, list(range(len(self.test_dataset))))

  def _scale_label(self, dataset, index, column='label'):
    scaled_label = self.label_scaler.transform(dataset[column][index].values.reshape(-1, 1))
    dataset.loc[index, column] = scaled_label.flatten()
    
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

  def get_output_converter(self):
    return None

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
