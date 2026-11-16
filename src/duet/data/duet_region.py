# Program: DuET v1.2.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""Dataset variants that skip the tensors a region-ablated model never reads.

A region-ablated model deletes an encoder, but the dataset would still build all three
sequence tensors, so the dropped region keeps its data-preparation cost. Skipping it
here makes the measured training time the cost of the deployed configuration. The key
stays in the batch as an empty tensor so collation is unaffected.

SKIP is a frozenset rather than a string because 'utr' is a prefix of 'utr3': a
substring test would drop the 3' UTR tensor whenever the 5' UTR one was dropped.
"""

import numpy as np
import torch

from .duet_triple import DuetTripleDataset

EMPTY = torch.tensor([], dtype=torch.float32)
REGIONS = frozenset({'utr', 'cds', 'utr3'})


class _RegionDropDataset(DuetTripleDataset):
  """Shared assembly; subclasses say which sequence tensors to skip."""

  SKIP = frozenset()

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if not (isinstance(cls.SKIP, frozenset) and cls.SKIP <= REGIONS):
      raise TypeError(f"{cls.__name__}: SKIP must be a frozenset of {sorted(REGIONS)}")

  def __getitem__(self, i: int):
    irow = self.data.iloc[i]
    if self.te_cols:
      y_values = irow[self.te_cols].values.astype(np.float32)
    else:
      y_values = np.empty((0,), dtype=np.float32)
    y = torch.from_numpy(y_values)

    return {
      "utr": EMPTY if 'utr' in self.SKIP else self.tensorize_utr(seq=irow.utr),
      "cds": EMPTY if 'cds' in self.SKIP else self.tensorize_cds(seq=irow.cds),
      "utr3": EMPTY if 'utr3' in self.SKIP else self.tensorize_utr3(seq=irow.utr3),
      "y": y,
      "mask": ~torch.isnan(y),
      "sequence_feature": EMPTY,
      "start": EMPTY,
    }


class DuetNoCdsDataset(_RegionDropDataset):
  """5' UTR + 3' UTR."""
  SKIP = frozenset({'cds'})


class DuetNoUtr5Dataset(_RegionDropDataset):
  """CDS + 3' UTR."""
  SKIP = frozenset({'utr'})


class DuetUtr5OnlyDataset(_RegionDropDataset):
  """5' UTR only."""
  SKIP = frozenset({'cds', 'utr3'})


class DuetCdsOnlyDataset(_RegionDropDataset):
  """CDS only."""
  SKIP = frozenset({'utr', 'utr3'})


class DuetUtr3OnlyDataset(_RegionDropDataset):
  """3' UTR only."""
  SKIP = frozenset({'utr', 'cds'})
