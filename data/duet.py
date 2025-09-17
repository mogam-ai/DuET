# Program:     DuET v1.0.0
# Author:      Sungho Lee, Jae-Won Lee, Jinhee Hong
# Affiliation: MOGAM Institute for Biomedical Research
# Contact:     sungho.lee@mogam.re.kr; jaewon.lee@mogam.re.kr; jinhee.hong@mogam.re.kr
# Citation:    TBD


import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from numpy import log1p
from functools import partial
from loguru import logger
from typing import Literal

from .utils import tensorize, CODON_CODES
from .sequence_feature_store import SequenceFeatureStore


class DuetDataset(Dataset):
  """Dataset class for DuET model."""
  def __init__(self,
               data_path: str,
               utr_seq_size: int,
               cds_seq_size: int,
               utr_channel_size: int,
               cds_channel_size: int,

               use_sequence_feature: bool=False,
               sequence_feature_path: str='',
               sequence_feature_cols: list=None,

               utr_col: str='utr5',
               cds_col: str='cds',
               input_col: str='',
               label_col: str='te',
               join_col: str='txID',
              
               log_label: bool=True,
               scale_label: bool=True,  # dummy
               use_codon_encoding: bool=False,
              
               encoder: Literal['single', 'dual'] = 'dual',
               use_start_context: bool=False,
    ):
    self.data: pd.DataFrame = pd.read_csv(data_path, sep='\t')

    self.dual = encoder == 'dual'
    if self.dual:
      assert utr_seq_size and cds_seq_size, "Both of UTR and CDS sequence size must be specified for dual encoder."

    if join_col == 'UTR5':  # This can cause conflict for some dataset.
        join_col = 'utr_seq_'  # Name mangling to avoid conflict with sequence_feature column name.
        self.data[join_col] = self.data['UTR5']

    if use_codon_encoding:
      if not self.dual and utr_seq_size:
        raise RuntimeError("UTR and codon-encoded CDS cannot be used together in single encoder mode.")

    self.utr_seq_size = utr_seq_size
    self.cds_seq_size = cds_seq_size
    self.seq_size = utr_seq_size + cds_seq_size
    
    # Padding is done at dataset loading time.
    # Only one-hot encoding will be processed in __getitem__.
    self.utr_channel_size = utr_channel_size
    if self.utr_seq_size:
      self.data[utr_col] = self.data[utr_col].str[-self.utr_seq_size:]  # Truncate leftmost bases and apply padding
    else:
      self.data[utr_col] = "" # negative indexing doesn't work for utr_seq_size = 0
    self.data[utr_col] = self.data[utr_col].str.pad(self.utr_seq_size, side='left', fillchar='N')

    self.cds_channel_size = cds_channel_size
    if self.cds_seq_size:
        self.data[cds_col] = self.data[cds_col].str[:self.cds_seq_size]  # Truncate rightmost bases and apply padding
    else:
        self.data[cds_col] = ""
    self.data[cds_col] = self.data[cds_col].str.pad(self.cds_seq_size, side='right', fillchar='N')
    
    if not self.dual:  # single encoder: merge cds column into utr column
      self.utr_seq_size = self.utr_seq_size + self.cds_seq_size
      self.data[utr_col] = self.data[utr_col] + self.data[cds_col]
      self.data = self.data[[utr_col, label_col, join_col]]
      self.data.columns = ['utr', 'label', join_col]
    else:
      self.data = self.data[[utr_col, cds_col, label_col, join_col]]
      self.data.columns = ['utr', 'cds', 'label', join_col]
    
    if log_label:
      self.data['label'] = log1p(self.data['label'])

    if use_start_context:
      self.data['start'] = self.data['utr'].str[-30:] + self.data['cds'].str[:33]  # 15+15bp around AUG(3bp)

    # Process sequence_features
    self.use_sequence_feature = use_sequence_feature
    if self.use_sequence_feature and sequence_feature_cols is None:
      raise ValueError("sequence_feature_cols must be specified when use_sequence_feature is True.")

    self.sequence_feature_size = len(sequence_feature_cols) if use_sequence_feature else 0

    if self.use_sequence_feature:
      sequenceFeatureStore = SequenceFeatureStore(sequence_feature_path, 
                                                  join_col=join_col, 
                                                  cols_to_use=sequence_feature_cols,)
      self.sequence_feature_cols = sequenceFeatureStore.get_sequence_feature_cols()
      logger.info(f"Using following sequence_feature columns: {', '.join(self.sequence_feature_cols)}")
      data_len_before = len(self.data)
      if join_col == 'utr_seq_': 
        self.data = self.data.merge(sequenceFeatureStore.sequence_features, 
                                    left_on=join_col, right_on='UTR5', how='inner')
        self.data.rename(columns={'UTR5_x':'UTR5'}, inplace=True)
      else:
        self.data = sequenceFeatureStore.merge_sequence_feature(self.data, on=join_col)
      logger.debug(f"sequence_feature merged: len {data_len_before} -> {len(self.data)}")
    
    self.tensorize_utr = partial(tensorize, channel_size=self.utr_channel_size)
    if use_codon_encoding:
      self.tensorize_cds = partial(tensorize, channel_size=self.cds_channel_size, 
                                   code=CODON_CODES, default=64, 
                                   quantizer=lambda x: [x[k:k+3] for k in range(0, len(x), 3)])
      # this quantizer does not ensure triplet, so the last codon may be incomplete.
    else:
      self.tensorize_cds = partial(tensorize, channel_size=self.cds_channel_size)

    if (not self.dual) and utr_seq_size == 0:  # single encoder + CDS only mode.
      logger.info(f"Single CDS encoder mode. utr_channel_size set to {self.cds_channel_size=}.")
      self.utr_channel_size = self.cds_channel_size
      self.tensorize_utr = self.tensorize_cds

    self.id_col = self.data.columns[0] # first column (entry key) needs to be unique.

  def __getitem__(self, i: int):
    irow = self.data.iloc[i]
    
    utr = self.tensorize_utr(seq=irow.utr)
    
    if self.dual:
      cds = self.tensorize_cds(seq=irow.cds)
    else:
      cds = torch.tensor([], dtype=torch.float32)
      
    y = torch.tensor(irow.label, dtype=torch.float32)
    
    sequence_feature = torch.tensor(irow[self.sequence_feature_cols].values.astype(np.float32),
                        dtype=torch.float32) if self.use_sequence_feature else torch.tensor([], dtype=torch.float32)

    # TIS site encoding if required
    start = self.tensorize_utr(seq=irow.start) if 'start' in irow else torch.tensor([], dtype=torch.float32)
    
    return {"utr": utr,  # one-hot encoded 5' UTR
            "cds": cds,  # one-hot encoded CDS
            "start": start,  # one-hot encoded start context (if use_start_context is True)
            "y": y,  # TE label
            "sequence_feature": sequence_feature,  # sequence feature vector (if use_sequence_feature is True)
            "txid": irow[self.id_col]  # unique identifier for the entry
            }

  def __len__(self) -> int:
    return len(self.data.utr)
