#!/usr/bin/env python3
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""Model config loader for duet_bench runners.

The DuET model config is a yaml under model_configs/ (duet.yaml) describing:
  - model_class:   nn class key ('duet')
  - precision:     32-true | 16-mixed (controls AMP in manual training loops)
  - output_names:  {multi, multi_cv, finetune, single} output directory names
  - model_params:  constructor kwargs (e.g. cnn_filters, gru_hidden_dim)
  - train:         training/data params
  - finetune:      finetune-specific params (run_finetune.py only)

Usage:
    from duet_bench.model_config import load_model_config, get_nn_classes
    cfg = load_model_config('duet')            # dict
    MultiCls, SingleCls = get_nn_classes(cfg['model_class'])
"""
import os

import yaml

from duet.models.duet import DuetMultiModel, DuetSingletaskModel

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_DIR = os.path.join(_HERE, 'model_configs')

# nn class key -> (MultiModel, SingleModel)
NN_REGISTRY = {
    'duet': (DuetMultiModel, DuetSingletaskModel),
}

# CLI --model value -> yaml filename stem
MODEL_TAGS = {
    'duet': 'duet',
}


def load_model_config(model_tag: str) -> dict:
    """Load model_configs/<stem>.yaml for the given --model tag."""
    if model_tag not in MODEL_TAGS:
        raise ValueError(f"Unknown model tag '{model_tag}'. Choices: {list(MODEL_TAGS)}")
    path = os.path.join(_CONFIG_DIR, MODEL_TAGS[model_tag] + '.yaml')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault('model_params', {})
    cfg.setdefault('precision', '32-true')
    return cfg


def get_nn_classes(model_class: str):
    """Return (MultiModel, SingleModel) classes for a model_class key."""
    if model_class not in NN_REGISTRY:
        raise ValueError(f"Unknown model_class '{model_class}'. Choices: {list(NN_REGISTRY)}")
    return NN_REGISTRY[model_class]


def amp_enabled(precision: str) -> bool:
    """True if precision requests 16-bit mixed AMP."""
    return str(precision).lower() in ('16-mixed', '16', 'fp16', 'amp')
