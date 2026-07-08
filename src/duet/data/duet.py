# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""Multi-task dataset for DuET: one row per gene, multiple TE columns."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from functools import partial
from typing import List, Optional

from .utils import tensorize, CODON_CODES


class DuetDataset(Dataset):
  """Dataset for multi-task TE prediction (all cell types simultaneously).

  Each sample returns:
    utr: one-hot encoded 5'UTR
    cds: one-hot/codon encoded CDS
    y:   (N_celltypes,) tensor with NaN for missing cell types
    mask: (N_celltypes,) bool tensor (True = valid, False = NaN)
  """

  def __init__(self,
               data_path: str,
               utr_seq_size: int,
               cds_seq_size: int,
               utr_channel_size: int = 4,
               cds_channel_size: int = 64,
               utr_col: str = 'utr5',
               cds_col: str = 'cds',
               join_col: str = 'txID',
               te_column_prefix: str = 'TE_',
               use_codon_encoding: bool = True,
               use_sequence_feature: bool = False,
               sequence_feature_path: str = '',
               sequence_feature_cols: list = None,
               # Ignored params for config compat
               label_col: str = 'logratio_te',
               **kwargs):
    self.data = pd.read_csv(data_path, sep='\t')

    # Identify TE columns
    self.te_cols = sorted([c for c in self.data.columns if c.startswith(te_column_prefix)])
    self.n_targets = len(self.te_cols)
    self.target_names = self.te_cols

    # Sequence processing (same as DuetSingletaskDataset)
    self.utr_seq_size = utr_seq_size
    self.cds_seq_size = cds_seq_size
    self.utr_channel_size = utr_channel_size
    self.cds_channel_size = cds_channel_size

    self.data[utr_col] = self.data[utr_col].str[-self.utr_seq_size:]
    self.data[utr_col] = self.data[utr_col].str.pad(self.utr_seq_size, side='left', fillchar='N')
    self.data[cds_col] = self.data[cds_col].str[:self.cds_seq_size]
    self.data[cds_col] = self.data[cds_col].str.pad(self.cds_seq_size, side='right', fillchar='N')

    self.data = self.data[[join_col, utr_col, cds_col] + self.te_cols]
    self.data.columns = [join_col, 'utr', 'cds'] + self.te_cols

    # Tensorizers
    self.tensorize_utr = partial(tensorize, channel_size=self.utr_channel_size)
    if use_codon_encoding:
      self.tensorize_cds = partial(tensorize, channel_size=self.cds_channel_size,
                                   code=CODON_CODES, default=64,
                                   quantizer=lambda x: [x[k:k+3] for k in range(0, len(x), 3)])
    else:
      self.tensorize_cds = partial(tensorize, channel_size=self.cds_channel_size)

    # Sequence features (unused for now but kept for compat)
    self.use_sequence_feature = use_sequence_feature
    self.sequence_feature_size = 0

  def __getitem__(self, i: int):
    irow = self.data.iloc[i]
    utr = self.tensorize_utr(seq=irow.utr)
    cds = self.tensorize_cds(seq=irow.cds)

    # Multi-target: (N_celltypes,) with NaN preserved.
    # Predict-only input (no TE_ columns) yields an empty label tensor; the
    # predict path does not use y/mask.
    if self.te_cols:
      y_values = irow[self.te_cols].values.astype(np.float32)
    else:
      y_values = np.empty((0,), dtype=np.float32)
    y = torch.from_numpy(y_values)
    mask = ~torch.isnan(y)

    return {
      "utr": utr,
      "cds": cds,
      "y": y,
      "mask": mask,
      "sequence_feature": torch.tensor([], dtype=torch.float32),
      "start": torch.tensor([], dtype=torch.float32),
    }

  def __len__(self):
    return len(self.data)
