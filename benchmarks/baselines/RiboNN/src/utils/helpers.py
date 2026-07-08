import json
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import yaml
import re


def masked_mse_loss(input, target, masked_value="NA", reduction="mean"):
    if masked_value == "NA":
        mask = target.isnan()
    else:
        mask = target == masked_value
    out = (input[~mask] - target[~mask]) ** 2
    if reduction == "mean":
        if out.numel() == 0:
            return (input * 0).sum()  # zero loss with grad
        return out.mean()
    elif reduction == "None":
        return out


def load_config(config_path: str = "config/data_conf.yml") -> Dict[str, Any]:
    """Load configuration"""
    config_path_obj = Path(config_path)
    return yaml.safe_load(config_path_obj.read_text(encoding="utf-8"))


def extract_config(run_df: pd.DataFrame, run_id: str) -> Dict[str, Any]:
    """Extract the config dict from mlflow run_df filtered by run_id.

    Args:
        run_df (pd.DataFrame): mlflow run_df
        run_id (str): mlflow run_id

    Returns:
        Dict: the config used to train the model associated with run_id
    """
    run_info = (
        run_df.query("run_id == @run_id")
        .filter(regex="^params\.", axis=1)
        .reset_index(drop=True)
    )
    run_info_dict = run_info.loc[0].to_dict()
    new_dict = dict()
    for k, v in run_info_dict.items():
        k = k.replace("params.", "")
        if v == "None":
            v = None
        elif v == "True":
            v = True
        elif v == "False":
            v = False
        elif re.search(
            r"(^\d+$)|(^[\d.]+$)|(^[\d.]+e-?[\d.]+$)|(^\[[\d.,e\- ]+\]$)", str(v)
        ):
            v = json.loads(str(v))
        if k != "betas":
            new_dict[k] = v
        else:
            match = re.match(r"\(([\d.]+),\s*([\d.]+)\)", v)
            if match:
                new_dict['adam_beta1'] = float(match.group(1))
                new_dict['adam_beta2'] = float(match.group(2))

    return new_dict

