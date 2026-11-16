# Program: DuET v1.2.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

# Region-ablated configurations of the three-region model.
#
# The seven non-empty combinations of the 5' UTR, CDS, and 3' UTR:
#
#   5' UTR + CDS + 3' UTR    duet_triple        (duet.models.duet_triple)
#   5' UTR + CDS             duet               (duet.models.duet)
#   5' UTR + 3' UTR          duet_nocds         (here)
#   CDS + 3' UTR             duet_noutr5        (here)
#   5' UTR                   duet_utr5only      (here)
#   CDS                      duet_cdsonly       (here)
#   3' UTR                   duet_utr3only      (here)
#
# The encoder of a dropped region is deleted rather than masked at inference, so that
# training time reflects the configuration that would be deployed. These models are
# sequence-only: pre-engineered features are computed over the whole transcript and
# would route information from the dropped region back into the model, so the
# constructors reject that configuration.

import torch

from typing import Dict

from .duet_triple import DuetTripleMultiModel


class _RegionAblationBase(DuetTripleMultiModel):
  """Builds the three-region model, then drops the branches this arm does not keep."""

  def __init__(self, n_targets: int, dropout: float = 0.4, **kwargs):
    super().__init__(n_targets=n_targets, dropout=dropout, **kwargs)
    if self.use_sequence_feature:
      raise ValueError("region ablation is sequence-only; disable use_sequence_feature")
    if self.use_start_context:
      raise ValueError("region ablation is sequence-only; disable use_start_context")
    self._drop_branch()
    self._rebuild_head(sum(self._kept_dims()), n_targets, dropout)

  def _drop_branch(self):
    raise NotImplementedError

  def _kept_dims(self):
    raise NotImplementedError


class DuetNoCdsMultiModel(_RegionAblationBase):
  """5' UTR + 3' UTR."""

  def _drop_branch(self):
    del self.cds_branch

  def _kept_dims(self):
    return [self.utr_branch.output_dim, self.utr3_branch.output_dim]

  def _features(self, batch: Dict) -> torch.Tensor:
    utr = self._encode_utr(batch['utr'].float())
    utr3 = self._encode_utr(batch['utr3'].float())
    return torch.cat([self.utr_branch(utr), self.utr3_branch(utr3)], dim=1)


class DuetNoUtr5MultiModel(_RegionAblationBase):
  """CDS + 3' UTR."""

  def _drop_branch(self):
    del self.utr_branch

  def _kept_dims(self):
    return [self.cds_branch.output_dim, self.utr3_branch.output_dim]

  def _features(self, batch: Dict) -> torch.Tensor:
    cds = self._encode_cds(batch['cds'].float())
    utr3 = self._encode_utr(batch['utr3'].float())
    return torch.cat([self.cds_branch(cds), self.utr3_branch(utr3)], dim=1)


class DuetUtr5OnlyMultiModel(_RegionAblationBase):
  """5' UTR only."""

  def _drop_branch(self):
    del self.cds_branch
    del self.utr3_branch

  def _kept_dims(self):
    return [self.utr_branch.output_dim]

  def _features(self, batch: Dict) -> torch.Tensor:
    return self.utr_branch(self._encode_utr(batch['utr'].float()))


class DuetCdsOnlyMultiModel(_RegionAblationBase):
  """CDS only."""

  def _drop_branch(self):
    del self.utr_branch
    del self.utr3_branch

  def _kept_dims(self):
    return [self.cds_branch.output_dim]

  def _features(self, batch: Dict) -> torch.Tensor:
    return self.cds_branch(self._encode_cds(batch['cds'].float()))


class DuetUtr3OnlyMultiModel(_RegionAblationBase):
  """3' UTR only."""

  def _drop_branch(self):
    del self.utr_branch
    del self.cds_branch

  def _kept_dims(self):
    return [self.utr3_branch.output_dim]

  def _features(self, batch: Dict) -> torch.Tensor:
    return self.utr3_branch(self._encode_utr(batch['utr3'].float()))
