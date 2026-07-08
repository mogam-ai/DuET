# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""Label scaling utilities."""

import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_scaler(train_labels: np.ndarray, method: str):
    """Fit scaler on train labels. Returns (transform_fn, inverse_fn)."""
    if method == "none":
        return lambda x: x, lambda x: x
    if method == "log1p":
        return np.log1p, np.expm1
    if method == "standard":
        scaler = StandardScaler()
        scaler.fit(np.asarray(train_labels).reshape(-1, 1))
        return (
            lambda x: scaler.transform(np.asarray(x).reshape(-1, 1)).flatten(),
            lambda x: scaler.inverse_transform(np.asarray(x).reshape(-1, 1)).flatten(),
        )
    raise ValueError(f"Unknown scaling method: {method}")
