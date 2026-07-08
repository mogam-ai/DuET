# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

# DuET model.
#
# Architecture: CNN backbone + BiGRU (2-layer) with a CNN skip-concat into
# AttentionPooling, then a fusion head with a mid-LayerNorm. Default dims are
# cnn_filters=64, gru_hidden_dim=96; other dims are settable via config for ablations.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List

from .module import Model
from .loss import build_loss
from ..configs.config import LossConfig


class SqueezeExcitation(nn.Module):
  def __init__(self, channels: int, reduction: int = 4):
    super().__init__()
    self.fc = nn.Sequential(
      nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True),
      nn.Linear(channels // reduction, channels), nn.Sigmoid())

  def forward(self, x):
    return x * self.fc(x.mean(dim=-1)).unsqueeze(-1)


class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, dropout):
    super().__init__()
    self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
    self.bn = nn.BatchNorm1d(out_channels)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.dropout(F.relu(self.bn(self.conv(x))))


class ResConvBlock(nn.Module):
  def __init__(self, channels, kernel_size, dropout):
    super().__init__()
    self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
    self.bn = nn.BatchNorm1d(channels)
    self.se = SqueezeExcitation(channels)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return x + self.dropout(self.se(F.relu(self.bn(self.conv(x)))))


class AttentionPooling(nn.Module):
  def __init__(self, hidden_dim):
    super().__init__()
    self.attention = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))

  def forward(self, x, mask=None):
    scores = self.attention(x).squeeze(-1)
    if mask is not None:
      scores = scores.masked_fill(~mask, float('-inf'))
    weights = torch.nan_to_num(F.softmax(scores, dim=1), nan=0.0)
    return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class BranchEncoder(nn.Module):
  """Branch: ConvBlock -> ResConv x(N-1) -> BiGRU(2-layer) -> CNN+GRU concat -> AttentionPooling."""

  def __init__(self, vocab_size, embed_dim, padding_idx, cnn_filters,
               kernel_taper, gru_hidden_dim, dropout):
    super().__init__()
    assert len(kernel_taper) >= 2
    self.padding_idx = padding_idx
    self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
    self.initial_conv = ConvBlock(embed_dim, cnn_filters, kernel_taper[0], dropout * 0.5)
    self.res_blocks = nn.ModuleList([
      ResConvBlock(cnn_filters, k, dropout * 0.5) for k in kernel_taper[1:]
    ])
    self.gru = nn.GRU(cnn_filters, gru_hidden_dim, num_layers=2,
                      batch_first=True, bidirectional=True, dropout=dropout * 0.5)
    # attn_pool input = cnn_filters + gru_hidden_dim*2 (CNN skip concat)
    attn_input_dim = cnn_filters + gru_hidden_dim * 2
    self.attn_pool = AttentionPooling(attn_input_dim)
    self.output_dim = attn_input_dim

  def forward(self, x):
    mask = (x != self.padding_idx)
    emb = self.embedding(x).transpose(1, 2)
    out = self.initial_conv(emb)
    for block in self.res_blocks:
      out = block(out)
    cnn_out = out.transpose(1, 2)            # (B, L, cnn_filters)
    gru_out, _ = self.gru(cnn_out)           # (B, L, gru_hidden*2)
    combined = torch.cat([cnn_out, gru_out], dim=-1)  # (B, L, cnn_filters + gru_hidden*2)
    return self.attn_pool(combined, mask)


class DuetSingletaskModel(Model):
  """DuET single-task model: dual-branch (5' UTR + CDS) encoder with an attention-pooled fusion head."""

  def __init__(self,
               utr_vocab_size: int = 5, utr_embed_dim: int = 96, utr_padding_idx: int = 4,
               codon_vocab_size: int = 65, cds_embed_dim: int = 128, cds_padding_idx: int = 64,
               cnn_filters: int = 64, cnn_layers: int = 4, gru_hidden_dim: int = 96,
               fusion_hidden: int = 256, dropout: float = 0.4,
               utr_kernel_taper: List[int] = [7, 5, 5, 3],
               cds_kernel_taper: List[int] = [5, 3, 3, 3],
               use_sequence_feature: bool = False, sequence_feature_size: int = 0,
               use_codon_encoding: bool = True,
               loss_cfg: LossConfig = None, **kwargs):
    super().__init__()

    if len(utr_kernel_taper) != cnn_layers:
      raise ValueError(f"utr_kernel_taper length ({len(utr_kernel_taper)}) != cnn_layers ({cnn_layers})")
    if len(cds_kernel_taper) != cnn_layers:
      raise ValueError(f"cds_kernel_taper length ({len(cds_kernel_taper)}) != cnn_layers ({cnn_layers})")

    self.loss = build_loss(loss_cfg)
    self.utr_padding_idx = utr_padding_idx
    self.use_sequence_feature = use_sequence_feature
    self.use_codon_encoding = use_codon_encoding
    self.use_start_context = kwargs.get('use_start_context', False)

    if not use_codon_encoding:
      cds_vocab_size = utr_vocab_size
      cds_embed = utr_embed_dim
      cds_pad_idx = utr_padding_idx
      cds_kernel_taper_eff = utr_kernel_taper
    else:
      cds_vocab_size = codon_vocab_size
      cds_embed = cds_embed_dim
      cds_pad_idx = cds_padding_idx
      cds_kernel_taper_eff = cds_kernel_taper
    self.cds_padding_idx = cds_pad_idx

    self.utr_branch = BranchEncoder(utr_vocab_size, utr_embed_dim, utr_padding_idx,
                                    cnn_filters, utr_kernel_taper, gru_hidden_dim, dropout)
    self.cds_branch = BranchEncoder(cds_vocab_size, cds_embed, cds_pad_idx,
                                    cnn_filters, cds_kernel_taper_eff, gru_hidden_dim, dropout)

    if self.use_start_context:
      tis_gru_hidden = 64
      tis_kernel_taper = [utr_kernel_taper[0], utr_kernel_taper[-1]]
      self.tis_branch = BranchEncoder(utr_vocab_size, utr_embed_dim, utr_padding_idx,
                                      cnn_filters, tis_kernel_taper, tis_gru_hidden, dropout)
      tis_out_dim = self.tis_branch.output_dim
    else:
      tis_out_dim = 0

    fusion_in = self.utr_branch.output_dim + self.cds_branch.output_dim + tis_out_dim
    if use_sequence_feature:
      feat_proj_dim = min(sequence_feature_size, 32)
      self.feat_proj = nn.Sequential(
        nn.Linear(sequence_feature_size, feat_proj_dim), nn.GELU(), nn.LayerNorm(feat_proj_dim))
      fusion_in += feat_proj_dim

    # mid-LayerNorm in head
    self.head = nn.Sequential(
      nn.LayerNorm(fusion_in), nn.Linear(fusion_in, fusion_hidden),
      nn.GELU(), nn.LayerNorm(fusion_hidden), nn.Dropout(dropout), nn.Linear(fusion_hidden, 1))

  def _encode_utr(self, utr_onehot):
    mask = utr_onehot.sum(dim=-1) > 0
    indices = utr_onehot.argmax(dim=-1)
    indices[~mask] = self.utr_padding_idx
    return indices

  def _encode_cds(self, cds_onehot):
    mask = cds_onehot.sum(dim=-1) > 0
    indices = cds_onehot.argmax(dim=-1)
    indices[~mask] = self.cds_padding_idx
    return indices

  def extract_embeddings(self, batch: Dict) -> torch.Tensor:
    """Return pre-head fused branch embeddings: cat([utr_branch, cds_branch]).

    Layout is [utr_branch(output_dim) | cds_branch(output_dim)] so downstream
    code can slice by self.utr_branch.output_dim / self.cds_branch.output_dim.
    TIS and sequence-feature branches are intentionally excluded to keep that
    slicing valid (UMAP / branch-norm analysis only inspect the two main branches).
    """
    utr = self._encode_utr(batch['utr'].float())
    cds = self._encode_cds(batch['cds'].float())
    return torch.cat([self.utr_branch(utr), self.cds_branch(cds)], dim=1)

  def predict(self, batch: Dict) -> tuple:
    utr = self._encode_utr(batch['utr'].float())
    cds = self._encode_cds(batch['cds'].float())
    features = [self.utr_branch(utr), self.cds_branch(cds)]
    if self.use_start_context:
      start = self._encode_utr(batch['start'].float())
      features.append(self.tis_branch(start))
    if self.use_sequence_feature:
      features.append(self.feat_proj(batch['sequence_feature']))
    score = self.head(torch.cat(features, dim=1)).squeeze(-1)
    return score.detach(), score

  def forward(self, batch: Dict) -> tuple:
    yhat, yhat_logit = self.predict(batch)
    loss = self.loss(yhat_logit, batch['y'])
    return yhat, loss


class DuetMultiModel(DuetSingletaskModel):
  """DuET multi-target variant (NaN-masked Huber loss)."""

  def __init__(self, n_targets: int, **kwargs):
    super().__init__(**kwargs)
    fusion_in = self.head[1].in_features
    fusion_hidden = self.head[1].out_features
    dropout = kwargs.get('dropout', 0.4)
    self.head = nn.Sequential(
      nn.LayerNorm(fusion_in), nn.Linear(fusion_in, fusion_hidden),
      nn.GELU(), nn.LayerNorm(fusion_hidden), nn.Dropout(dropout),
      nn.Linear(fusion_hidden, n_targets))
    self.n_targets = n_targets

  def predict(self, batch: Dict) -> tuple:
    utr = self._encode_utr(batch['utr'].float())
    cds = self._encode_cds(batch['cds'].float())
    features = [self.utr_branch(utr), self.cds_branch(cds)]
    if self.use_start_context:
      start = self._encode_utr(batch['start'].float())
      features.append(self.tis_branch(start))
    if self.use_sequence_feature:
      features.append(self.feat_proj(batch['sequence_feature']))
    score = self.head(torch.cat(features, dim=1))  # (B, n_targets)
    return score.detach(), score

  def forward(self, batch: Dict) -> tuple:
    yhat, logit = self.predict(batch)
    y = batch['y']
    mask = ~torch.isnan(y)
    loss = self.loss(logit[mask], y[mask]) if mask.sum() > 0 else (logit * 0).sum()
    return yhat, loss
