#!/usr/bin/env python3
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""Per-celltype plot (+ MEAN): x-axis = key methods, grouped bars = task (single/multi/ft).

Key methods (per-task mapping):
  - DuET ensemble : single has no inner CV so uses the single prediction (v*); multi/ft use v*_top5
  - RiboNN ensemble          : ribonn_top5
  - mix(duet+ribonn) ensemble: single=mix_duet_ribonn_top5, multi/ft=mix_duet_top5_ribonn_top5
spearman mean +/- std over folds, for key celltypes.
Output: collect_all/figs/celltype_<ct>.png  (+ MEAN.png)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchlib import load_config

# Aggregate root = <output_dir>/collect_all (output_dir from config.yaml).
# Override with the BENCH_CONFIG env var to point at a different config.
BASE = os.path.join(load_config(os.environ.get('BENCH_CONFIG'))['output_dir'], 'collect_all')
FIG_DIR = os.path.join(BASE, 'figs'); os.makedirs(FIG_DIR, exist_ok=True)
KEY_CTS = ['gm12878_TE', 'gm12892_TE', 'hek293t_TE', 'hela_TE',
           'imr-90_TE', 'muscle-tissue_TE', 'pc3_TE', 'u-2-os_TE']
METRIC = 'spearman'
TASKS = ['single', 'multi', 'ft']

# x-axis label -> actual per-task method name
#  (single uses the single DuET prediction; multi/ft use the top5 ensemble)
METHOD_LABELS = ['DuET', 'RiboNN', 'mix(DuET+RiboNN)']
METHOD_MAP = {
    'single': ['duet', 'ribonn_top5', 'mix_duet_ribonn_top5'],
    'multi':  ['duet_top5', 'ribonn_top5', 'mix_duet_top5_ribonn_top5'],
    'ft':     ['duet_top5', 'ribonn_top5', 'mix_duet_top5_ribonn_top5'],
}
TASK_COLOR = {'single': '#4C72B0', 'multi': '#DD8452', 'ft': '#55A868'}

# load per-task long df
data = {}
for task in TASKS:
    p = os.path.join(BASE, f'{task}_long.csv')
    if os.path.exists(p):
        data[task] = pd.read_csv(p)


def stats_for(task, method, ct):
    """Method spearman (mean, std) for celltype ct; ct=='MEAN' averages over key celltypes."""
    if task not in data or method is None:
        return np.nan, np.nan
    df = data[task]
    df = df[df['method'] == method]
    if ct != 'MEAN':
        df = df[df['celltype'] == ct]
    else:
        df = df[df['celltype'].isin(KEY_CTS)]
    if df.empty:
        return np.nan, np.nan
    return df[METRIC].mean(), df[METRIC].std()


def make_plot(ct):
    x = np.arange(len(METHOD_LABELS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for ti, task in enumerate(TASKS):
        means, stds = [], []
        for mi in range(len(METHOD_LABELS)):
            m = METHOD_MAP[task][mi]
            mu, sd = stats_for(task, m, ct)
            means.append(mu); stds.append(sd)
        offset = (ti - 1) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=task, color=TASK_COLOR[task], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_LABELS, rotation=20, ha='right')
    ax.set_ylabel(f'{METRIC}')
    ax.set_title(f'{ct} - {METRIC} (mean +/- std over folds)')
    ax.legend(title='task')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f'celltype_{ct}.png')
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


for ct in KEY_CTS + ['MEAN']:
    print('saved', make_plot(ct))
