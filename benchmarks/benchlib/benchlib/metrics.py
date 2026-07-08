# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""Evaluation metrics."""

import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics (handles NaN)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    pearson, _ = stats.pearsonr(y_true, y_pred)
    spearman, _ = stats.spearmanr(y_true, y_pred)
    return {
        "pearson": float(pearson),
        "spearman": float(spearman),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
