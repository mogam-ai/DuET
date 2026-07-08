#!/usr/bin/env python3
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""collect_all single/multi/ft long CSVs -> one celltype x (task_method) spearman table.

Rows = celltype, columns = <task>_<method> (e.g. multi_duet_ens), cells = mean (std),
over the 10 folds of key celltypes. The final MEAN row is the celltype average.
Output: collect_all/table_all_spearman.csv
"""
import os
import pandas as pd
from benchlib import load_config

# Aggregate root = <output_dir>/collect_all (output_dir from config.yaml).
# Override with the BENCH_CONFIG env var to point at a different config.
BASE = os.path.join(load_config(os.environ.get('BENCH_CONFIG'))['output_dir'], 'collect_all')
KEY_CTS = ['gm12878_TE', 'gm12892_TE', 'hek293t_TE', 'hela_TE',
           'imr-90_TE', 'muscle-tissue_TE', 'pc3_TE', 'u-2-os_TE']
METRIC = 'spearman'

METHOD_ORDER = [
    'duet', 'duet_common', 'duet_top5', 'duet_top5_common',
    'ribonn', 'ribonn_top5',
    'mix_duet_ribonn', 'mix_duet_top3_ribonn_top3', 'mix_duet_top5_ribonn_top5',
]

frames = []
for task in ['single', 'multi', 'ft']:
    path = os.path.join(BASE, f'{task}_long.csv')
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    df = df[df['celltype'].isin(KEY_CTS)].copy()
    df['col'] = task + '_' + df['method']            # task prefix
    frames.append(df)

alldf = pd.concat(frames, ignore_index=True)

# celltype x col mean(std)
agg = alldf.groupby(['celltype', 'col'])[METRIC].agg(['mean', 'std']).reset_index()
agg['cell'] = agg.apply(lambda r: f"{r['mean']:.3f} ({r['std']:.3f})", axis=1)
pivot = agg.pivot(index='celltype', columns='col', values='cell')

# MEAN row (over all celltype x fold)
mean_agg = alldf.groupby('col')[METRIC].agg(['mean', 'std'])
pivot.loc['MEAN'] = pd.Series(
    {c: f"{mean_agg.loc[c, 'mean']:.3f} ({mean_agg.loc[c, 'std']:.3f})" for c in mean_agg.index})

# column order: by task, then by method
col_order = [f'{task}_{m}' for task in ['single', 'multi', 'ft'] for m in METHOD_ORDER]
cols = [c for c in col_order if c in pivot.columns]
pivot = pivot[cols].reindex(KEY_CTS + ['MEAN'])
pivot.columns.name = None

out = os.path.join(BASE, f'table_all_{METRIC}.csv')
pivot.to_csv(out)
print(f'Saved {out}')
print(pivot.to_string())
