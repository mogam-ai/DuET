# DuET benchmark (duet_bench)

DuET single-task / multitask-CV / multitask-pretrain / finetune benchmark runners.
Model hyperparameters are loaded from `model_configs/duet.yaml`.

Run everything from the repository root (see the top-level `benchmarks/README.md`
for environment setup: conda envs, `LD_LIBRARY_PATH`, `TMPDIR`, thread limits).
The `--config` argument points at `benchmarks/config.yaml`; model hyperparameters
live separately in `model_configs/*.yaml`.

## Model -> output directory names

Output directory names come from `output_names` in `model_configs/duet.yaml`:

| runner       | output dir       |
|--------------|------------------|
| single       | `DuET`           |
| multi_cv     | `DuET_multi_cv`  |
| multitarget  | `DuET_multi`     |
| finetune     | `DuET_ft`        |

Output path patterns:
- single:      `<output_dir>/celltype_te/<celltype>/DuET/fold_N/`
- multi_cv / multitarget: `<output_dir>/celltype_te_multi/multitask/<name>/fold_N/`
- finetune:    `<output_dir>/celltype_te/<celltype>/DuET_ft/fold_N/`

For the ensemble runners each `fold_N/` holds `inner_0..8/` (9 inner folds) plus
`metrics.json`; the reported metric is the top-5 inner ensemble, and a fold with
an existing `metrics.json` is skipped on re-run (resume). The plain-CV runners
(`run_single.py`, `run_multi_cv.py`) write a single `predictions.tsv` per fold.

## 1) Single-task

Independent per-celltype training; jobs auto-distribute across `--gpus`/`--jobs-per-gpu`.

```bash
python benchmarks/duet_bench/run_single.py --config benchmarks/config.yaml --gpus 0,1 --jobs-per-gpu 3 \
  --names celltype_te/hek293t_TE celltype_te/hela_TE
```

- `--names` uses the `celltype_te/<celltype>` form.
- Data: loads `celltype_te/<celltype>.tsv` directly (label_col=`logratio_te`).

## 2) Multitask, plain 10-fold CV

One multi-target model per outer fold, no inner ensemble (the plain multitask
training path). Single job, so distribute with `--gpus`.

```bash
python benchmarks/duet_bench/run_multi_cv.py --config benchmarks/config.yaml --gpus 0,1
```

## 3) Multitask pretrain (multitarget, inner ensemble)

Trains on the multitask TE table (76 targets at once) with an inner 9-fold top-5
ensemble. This is a single job, so `--gpus` does not parallelize folds; split
folds across processes manually.

```bash
python benchmarks/duet_bench/run_multitarget.py --config benchmarks/config.yaml --gpu 0 --folds 0 1 2 3 4 &
python benchmarks/duet_bench/run_multitarget.py --config benchmarks/config.yaml --gpu 1 --folds 5 6 7 8 9 &
```

- Each `inner_<i>/best.pt` is used as the finetune init (below).

## 4) Finetune

Finetune the multitask pretrain checkpoint to a single celltype; per-celltype
jobs auto-distribute across `--gpus`/`--jobs-per-gpu`.

```bash
python benchmarks/duet_bench/run_finetune.py --config benchmarks/config.yaml --gpus 0,1 --jobs-per-gpu 3 \
  --pretrain-dir <output_dir>/celltype_te_multi/multitask/DuET_multi \
  --names celltype_te/hek293t_TE celltype_te/pc3_TE
```

- `--pretrain-dir` must be the multitarget output (`DuET_multi`) for the same model (matching dims).
- finetune inner `i` loads the pretrain `fold_N/inner_i/best.pt` directly.
- te_col mapping: `TE_<celltype without _TE>` (e.g. hek293t_TE -> `TE_hek293t`); celltypes with no such column are skipped.
- The finetune data split shares the multitask outer split, so the finetune test
  never overlaps the pretrain train_val (verified: zero test leakage). Inner-val may
  include data the pretrain saw (transductive), matching RiboNN's original
  `transfer_learning.py` design.
- Omitting `--names` runs all celltypes in the config's `celltype_te` list;
  celltypes without a matching `TE_<celltype>` column (e.g. `all-celltype_TE`) are skipped.

### Stability note (many concurrent finetune workers)

- The DataLoader uses `num_workers=0` with `MultiTargetDataset(pretensorize=True)`.
  With `num_workers>0`, fork+CUDA workers can deadlock at epoch/inner boundaries;
  pre-tensorizing makes `__getitem__` a pure indexing op, keeping GPU util high at
  `num_workers=0`.
- Set `OMP_NUM_THREADS`/`DUET_NUM_THREADS` (default 4) to cap per-process torch
  intra-op threads; otherwise each process spawns as many threads as cores and
  oversubscribes the host.

## Aggregating results

```bash
python benchmarks/duet_bench/collect_all.py --config benchmarks/config.yaml --task single   # or multi / ft
# options: --folds 0 1 2   --celltypes hek293t_TE hela_TE
```

- Inner prediction npz files (`inner_<i>/test_pred.npz`) are pre-saved by each
  runner's `--predict-only`.
