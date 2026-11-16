# Program: DuET v1.2.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

# DuET three-region model.
#
# DuET with a third encoder branch for the 3' UTR. The 3' UTR branch has the same
# architecture as the 5' UTR branch and is encoded the same way (nucleotide), but its
# width is set independently via utr3_cnn_filters / utr3_gru_hidden_dim / utr3_embed_dim,
# each falling back to the 5' UTR branch setting. Data must provide batch['utr3']
# (see duet.data.duet_triple.DuetTripleDataset).

import torch
import torch.nn as nn

from typing import Dict, List

from .duet import BranchEncoder, DuetMultiModel


class DuetTripleMultiModel(DuetMultiModel):
  """DuET multi-target model over the 5' UTR, CDS, and 3' UTR."""

  def __init__(self, n_targets: int,
               utr_vocab_size: int = 5, utr_embed_dim: int = 96, utr_padding_idx: int = 4,
               cnn_filters: int = 64, gru_hidden_dim: int = 96, dropout: float = 0.4,
               utr_kernel_taper: List[int] = [7, 5, 5, 3],
               utr3_embed_dim: int = None, utr3_cnn_filters: int = None,
               utr3_gru_hidden_dim: int = None, utr3_kernel_taper: List[int] = None,
               **kwargs):
    super().__init__(n_targets=n_targets, utr_vocab_size=utr_vocab_size,
                     utr_embed_dim=utr_embed_dim, utr_padding_idx=utr_padding_idx,
                     cnn_filters=cnn_filters, gru_hidden_dim=gru_hidden_dim,
                     dropout=dropout, utr_kernel_taper=utr_kernel_taper, **kwargs)

    self.utr3_branch = BranchEncoder(utr_vocab_size,
                                     utr3_embed_dim or utr_embed_dim,
                                     utr_padding_idx,
                                     utr3_cnn_filters or cnn_filters,
                                     utr3_kernel_taper or utr_kernel_taper,
                                     utr3_gru_hidden_dim or gru_hidden_dim,
                                     dropout)
    self._rebuild_head(self._fusion_in(), n_targets, dropout)

  def _fusion_in(self):
    return self.head[1].in_features + self.utr3_branch.output_dim

  def _rebuild_head(self, fusion_in, out_dim, dropout):
    fusion_hidden = self.head[1].out_features
    self.head = nn.Sequential(
      nn.LayerNorm(fusion_in), nn.Linear(fusion_in, fusion_hidden),
      nn.GELU(), nn.LayerNorm(fusion_hidden), nn.Dropout(dropout),
      nn.Linear(fusion_hidden, out_dim))

  def _features(self, batch: Dict) -> torch.Tensor:
    utr = self._encode_utr(batch['utr'].float())
    cds = self._encode_cds(batch['cds'].float())
    utr3 = self._encode_utr(batch['utr3'].float())
    features = [self.utr_branch(utr), self.cds_branch(cds), self.utr3_branch(utr3)]
    if self.use_start_context:
      features.append(self.tis_branch(self._encode_utr(batch['start'].float())))
    if self.use_sequence_feature:
      features.append(self.feat_proj(batch['sequence_feature']))
    return torch.cat(features, dim=1)

  def predict(self, batch: Dict) -> tuple:
    score = self.head(self._features(batch))
    return score.detach(), score

  def forward(self, batch: Dict) -> tuple:
    yhat, logit = self.predict(batch)
    y = batch['y']
    mask = ~torch.isnan(y)
    loss = self.loss(logit[mask], y[mask]) if mask.sum() > 0 else (logit * 0).sum()
    return yhat, loss
