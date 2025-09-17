# Program:     DuET v1.0.0
# Author:      Sungho Lee, Jae-Won Lee, Jinhee Hong
# Affiliation: MOGAM Institute for Biomedical Research
# Contact:     sungho.lee@mogam.re.kr; jaewon.lee@mogam.re.kr; jinhee.hong@mogam.re.kr
# Citation:    TBD


import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from typing import Optional
from loguru import logger

from .utils import seq2tensor
from .sequence_feature_store import SequenceFeatureStore


class LstmDataset(Dataset):
  """Dataset for TranslateLSTM model."""
  def __init__(self,
               data_path: str,
               utr_seq_size: int,
               
               use_sequence_feature: bool,
               sequence_feature_path: str,
               sequence_feature_cols: list=None,
               
               utr_col: str='utr5',
               label_col: str='te',
               join_col: str='utr',
    ):
    self.data = pd.read_csv(data_path, sep='\t')
    self.data = self.data[[utr_col, label_col, join_col]]  # select columns
    self.data.columns = ['input', 'label', join_col]
    self.utr_seq_size = utr_seq_size
    self.data['input'] = self.data['input'].astype(str)

    self.sequence_feature_size = len(sequence_feature_cols) if use_sequence_feature else 0

    self.use_sequence_feature = use_sequence_feature
    if use_sequence_feature:
      sequenceFeatureStore = SequenceFeatureStore(sequence_feature_path, 
                                                  join_col=join_col,
                                                  cols_to_use=sequence_feature_cols,)
      data_len_before_merge = len(self.data)
      self.data = sequenceFeatureStore.merge_sequence_feature(self.data, on=join_col)
      logger.debug(f"Sequence feature merged: {data_len_before_merge} -> {len(self.data)} rows")
      self.sequence_feature_cols = sequenceFeatureStore.get_sequence_feature_cols()
      logger.info(f"Using following sequence feature columns: {','.join(self.sequence_feature_cols)}")
  
  def __getitem__(self, i: int):
    irow = self.data.iloc[i]

    seq_len = min(len(irow.input), self.utr_seq_size)
    x = seq2tensor(seq=irow.input, max_len=self.utr_seq_size)
    y = torch.tensor(irow.label, dtype=torch.float32)

    sequence_feature = []
    if self.use_sequence_feature:
      sequence_feature = torch.tensor(irow[self.sequence_feature_cols].values.astype(np.float32),
                              dtype=torch.float32)

    return {"x": x,  # (utr_seq_size, 4)
            "y": y,  # (1,)
            "seq_len": seq_len,  # (1,)
            "sequence_feature": sequence_feature,  # (sequence_feature_size,)
            }

  def __len__(self) -> int:
    return len(self.data)