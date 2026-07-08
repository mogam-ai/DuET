# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""KFold split utilities."""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def generate_splits(data_path: str, output_dir: str, k: int = 10, seed: int = 42,
                    label_col: str = None, scaling: str = "none",
                    skip_if_exists: bool = False) -> int:
    """Generate k-fold indices (3-way: test=fold_i, val=fold_(i-1), train=rest).

    If label_col and scaling are provided, fits a scaler on train labels per fold
    and saves it as scaler.joblib.

    If skip_if_exists is True and all fold indices already exist and are valid JSON,
    returns immediately without rewriting (avoids races when many workers call this
    concurrently on the same shared splits dir).

    Writes are atomic (temp file + os.replace) so concurrent readers never observe
    a truncated/empty indices.json.
    """
    output_dir = Path(output_dir)

    if skip_if_exists and _splits_valid(output_dir, k):
        return k

    df = pd.read_csv(data_path, sep='\t')
    n = len(df)
    all_idx = np.arange(n)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    folds = [idx for _, idx in kf.split(all_idx)]

    for i in range(k):
        fold_dir = output_dir / f"fold_{i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        test_idx = folds[i]
        val_idx = folds[(i - 1) % k]
        train_idx = np.concatenate([folds[j] for j in range(k)
                                    if j != i and j != (i - 1) % k])
        _atomic_write_json(
            fold_dir / "indices.json",
            {"train": train_idx.tolist(), "val": val_idx.tolist(),
             "test": test_idx.tolist()})

        # Fit and save scaler
        if label_col and scaling != "none":
            _save_scaler(df[label_col].values[train_idx], scaling, fold_dir)

    return k


def _splits_valid(output_dir: Path, k: int) -> bool:
    """True if every fold's indices.json exists and is valid JSON."""
    for i in range(k):
        p = output_dir / f"fold_{i}" / "indices.json"
        if not p.exists():
            return False
        try:
            with open(p) as f:
                json.load(f)
        except (json.JSONDecodeError, OSError):
            return False
    return True


def _atomic_write_json(path: Path, obj):
    """Write JSON atomically: write to a unique temp file then os.replace."""
    import os
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def generate_rank_split(train_path: str, test_path: str, output_dir: str,
                        val_frac: float = 0.05, label_col: str = None,
                        scaling: str = "none") -> int:
    """Generate indices for pre-split (rank) data. Single fold."""
    train_df = pd.read_csv(train_path, sep='\t')
    n_train = len(train_df)
    fold_dir = Path(output_dir) / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)
    all_train = np.arange(n_train)
    val_mask = rng.rand(n_train) < val_frac
    train_idx = all_train[~val_mask]

    with open(fold_dir / "indices.json", "w") as f:
        json.dump({"train": train_idx.tolist(),
                   "val": all_train[val_mask].tolist()}, f)
    with open(fold_dir / "meta.json", "w") as f:
        json.dump({"train_file": train_path, "test_file": test_path}, f)

    # Fit and save scaler
    if label_col and scaling != "none":
        _save_scaler(train_df[label_col].values[train_idx], scaling, fold_dir)

    return 1


def _save_scaler(train_labels: np.ndarray, method: str, fold_dir: Path):
    """Fit scaler on train labels and save to fold_dir/scaler.joblib."""
    if method == "standard":
        scaler = StandardScaler()
        scaler.fit(train_labels.reshape(-1, 1))
        joblib.dump({"method": method, "scaler": scaler}, fold_dir / "scaler.joblib")
    elif method == "log1p":
        joblib.dump({"method": method}, fold_dir / "scaler.joblib")


def load_indices(fold_dir: str) -> dict:
    """Load indices.json from a fold directory."""
    with open(Path(fold_dir) / "indices.json") as f:
        return json.load(f)


def load_scaler(fold_dir: str):
    """Load scaler from fold directory. Returns (transform_fn, inverse_fn)."""
    scaler_path = Path(fold_dir) / "scaler.joblib"
    if not scaler_path.exists():
        return lambda x: x, lambda x: x
    data = joblib.load(scaler_path)
    if data["method"] == "log1p":
        return np.log1p, np.expm1
    if data["method"] == "standard":
        scaler = data["scaler"]
        return (
            lambda x: scaler.transform(np.asarray(x).reshape(-1, 1)).flatten(),
            lambda x: scaler.inverse_transform(np.asarray(x).reshape(-1, 1)).flatten(),
        )
    return lambda x: x, lambda x: x
