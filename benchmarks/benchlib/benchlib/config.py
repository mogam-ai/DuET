# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""Config loading and dataset resolution."""

import yaml
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """Load benchmark config. Searches up from cwd if not specified."""
    if config_path is None:
        # Search for config.yaml in current dir or parent benchmarks dir
        for candidate in [Path("config.yaml"), Path("../config.yaml"), Path("../../config.yaml")]:
            if candidate.exists():
                config_path = str(candidate)
                break
        else:
            raise FileNotFoundError("config.yaml not found")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_data_path(ds_cfg: dict, name: str) -> str:
    """Get the data file path for a dataset name."""
    base = Path(ds_cfg["base_path"])
    if ds_cfg["split"] == "rank":
        return None  # rank split uses train/test files separately
    return str(base / ds_cfg["data_pattern"].format(name=name))


def get_rank_paths(ds_cfg: dict, name: str) -> tuple:
    """Get (train_path, test_path) for rank split datasets."""
    base = Path(ds_cfg["base_path"])
    train_path = str(base / ds_cfg["train_pattern"].format(name=name))
    test_path = str(base / ds_cfg["test_pattern"].format(name=name))
    return train_path, test_path
