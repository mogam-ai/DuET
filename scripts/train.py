#!/usr/bin/env python3
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD
#
# CLI entry point for training. Parses config/override arguments and delegates
# the training loop to duet.train() so the logic lives in one place. Training
# artifacts (model.ckpt, metrics.yaml, indices.json, run_stats.json with peak
# VRAM + runtime) are written by duet.train() under log_dir/exp_name.
#
# Usage:
#   python scripts/train.py --config configs/duet.yaml
#   python scripts/train.py --config configs/duet.yaml \
#       --override-configs exp_name=my_run optimizer.lr=1e-5
#   CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/duet_singletask.yaml

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from duet.configs.config import parse_args
import duet


def main():
    config_list, override_configs = parse_args()
    # duet.train selects the GPU via the device string; honor CUDA_VISIBLE_DEVICES
    # if set (train on the first visible device), else default to 'cuda'.
    device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES', '') != 'cpu' else 'cpu'
    duet.train(config_paths=config_list,
               override_configs=override_configs,
               device=device)


if __name__ == '__main__':
    main()
