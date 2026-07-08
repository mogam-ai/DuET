# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

import torch.nn as nn
from ..configs.config import LossConfig


class PearsonCorrLoss(nn.Module):
    """1 - Pearson correlation coefficient as a loss."""
    def forward(self, yhat, y):
        y_mean = y - y.mean()
        yhat_mean = yhat - yhat.mean()
        cov = (y_mean * yhat_mean).sum()
        std_y = y_mean.pow(2).sum().sqrt()
        std_yhat = yhat_mean.pow(2).sum().sqrt()
        r = cov / (std_y * std_yhat + 1e-8)
        return 1 - r


class CompositeLoss(nn.Module):
    """Weighted combination of a base loss and correlation loss."""
    def __init__(self, base_loss: nn.Module, corr_weight: float = 0.5):
        super().__init__()
        self.base_loss = base_loss
        self.corr_loss = PearsonCorrLoss()
        self.corr_weight = corr_weight

    def forward(self, yhat, y):
        base = self.base_loss(yhat, y)
        corr = self.corr_loss(yhat, y)
        return base + self.corr_weight * corr


def build_loss(loss_cfg: LossConfig) -> nn.Module:
    """Build loss function from config.

    Supported names:
      - mse, mae, huber: standard PyTorch losses
      - huber_corr: Huber + (1 - Pearson R), weighted by loss.param.corr_weight (default 0.5)

    Config example:
      loss:
        name: huber_corr
        param:
          corr_weight: 0.5
    """
    params = loss_cfg.param or {}
    name = loss_cfg.name.lower()

    match name:
        case 'mse': return nn.MSELoss(reduction='mean')
        case 'mae': return nn.L1Loss(reduction='mean')
        case 'huber': return nn.HuberLoss(reduction='mean', delta=params.get('delta', 1.0))
        case 'huber_corr':
            base = nn.HuberLoss(reduction='mean', delta=params.get('delta', 1.0))
            return CompositeLoss(base, corr_weight=params.get('corr_weight', 0.5))
        case _: raise NotImplementedError(f"loss '{loss_cfg.name}' is not implemented")
