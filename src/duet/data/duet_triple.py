# Program: DuET v1.2.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""Multi-task dataset carrying a 3' UTR channel, for the three-region model.

The data file must contain a utr3 column; build one with
preprocessing/make_multitask_utr3.py.
"""

import pandas as pd
import torch
from functools import partial

from .duet import DuetDataset
from .utils import tensorize


class DuetTripleDataset(DuetDataset):
  """Multi-task TE dataset with an extra 3' UTR sequence channel."""

  def __init__(self,
               data_path: str,
               utr_seq_size: int,
               cds_seq_size: int,
               utr3_seq_size: int = 1000,
               utr3_channel_size: int = 4,
               utr3_col: str = 'utr3',
               **kwargs):
    # DuetDataset trims self.data to the join/sequence/TE columns, so utr3 is read
    # from the raw file before that happens.
    raw = pd.read_csv(data_path, sep='\t', usecols=[utr3_col])

    super().__init__(data_path=data_path, utr_seq_size=utr_seq_size,
                     cds_seq_size=cds_seq_size, **kwargs)

    self.utr3_seq_size = utr3_seq_size
    self.utr3_channel_size = utr3_channel_size

    utr3 = raw[utr3_col].astype(str).str[:utr3_seq_size]
    utr3 = utr3.str.pad(utr3_seq_size, side='right', fillchar='N').tolist()
    if len(utr3) != len(self.data):
      raise ValueError(f"utr3 rows ({len(utr3)}) != data rows ({len(self.data)})")
    self.data['utr3'] = utr3

    self.tensorize_utr3 = partial(tensorize, channel_size=self.utr3_channel_size)

  def __getitem__(self, i: int):
    item = super().__getitem__(i)
    item['utr3'] = self.tensorize_utr3(seq=self.data.iloc[i]['utr3'])
    return item
