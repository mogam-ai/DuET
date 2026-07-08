# DuET Benchmark Suite

Unified benchmarking framework comparing **DuET** against published models on two tasks:

- **MRL** (mean ribosome load, MPRA 5′UTR): Optimus 5-prime, FramePool, TranslateLSTM, UTR-LM, GEMORNA-5UTR, DuET
- **TE** (translation efficiency, per cell type): Optimus 5-prime, FramePool, TranslateLSTM, UTR-LM, RiboNN, DuET

All models share one framework for data splitting, scaling, and metrics (`benchlib`), so results are directly comparable. Each model's original architecture and core training recipe are preserved; adaptations are minimal (see [Changes from original code](#changes-from-original-code)).

---

## 1. Layout

```text
benchmarks/
├── benchlib/            # pip-installed shared helpers: split, scaling, metrics, config, parallel runner
├── baselines/
│   ├── optimus5p/       # Keras Conv1D            (env: benchmark_tf)
│   ├── framepool/       # Keras FrameSlice        (env: benchmark_tf)
│   ├── translatelstm/   # PyTorch dual encoder    (env: benchmark_torch)
│   ├── UTR-LM/          # ESM2 + CNN              (env: benchmark_torch)
│   ├── GEMORNA/         # MRL only                (env: benchmark_torch)
│   └── RiboNN/          # PyTorch Lightning, full transcript, inner 9-fold ensemble (env: benchmark_torch)
├── duet_bench/          # DuET runners (single / multitarget / finetune) + aggregation
│   ├── run_single.py         # per-celltype from scratch
│   ├── run_multi_cv.py       # multi-target model, plain 10-fold CV (no ensemble)
│   ├── run_multitarget.py    # one multitask model over all celltypes (inner ensemble)
│   ├── run_finetune.py       # multitask-pretrained → per-celltype fine-tune
│   ├── model_config.py       # loads model_configs/*.yaml
│   ├── model_configs/        # duet.yaml
│   ├── collect_all.py        # aggregate single/multi/ft → long CSV (compare-only)
│   ├── method_summary.py     # celltype × method summary table
│   ├── plot_celltype.py      # per-celltype grouped-bar figures
│   └── pooled_rsquared.py    # pooled Pearson R / R² + variance decomposition (reviewer metric)
├── config.yaml          # datasets + models + output_dir + gpus
├── run_all.sh           # full suite: baselines + DuET runners + aggregation
└── envs/                # benchmark_torch.yaml, benchmark_tf.yaml (+ *_full.yaml pinned snapshots)
```

---

## 2. Environment setup

Two conda environments are used:

| env | framework | models |
|-----|-----------|--------|
| `benchmark_torch` | PyTorch 2.5 | DuET, RiboNN, UTR-LM, TranslateLSTM, GEMORNA |
| `benchmark_tf` | TensorFlow / Keras | optimus5p, framepool |

```bash
# create envs
conda env create -f benchmarks/envs/benchmark_torch.yaml -n benchmark_torch
conda env create -f benchmarks/envs/benchmark_tf.yaml  -n benchmark_tf

# install shared helpers into EACH env
conda activate benchmark_torch && pip install -e benchmarks/benchlib --no-deps
conda activate benchmark_tf  && pip install -e benchmarks/benchlib --no-deps

# install DuET package (benchmark_torch env) - run_single/multitarget/finetune import `duet`
conda activate benchmark_torch && pip install -e .
```

**Required at runtime (both envs):**

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"   # else GLIBCXX_3.4.29 errors (matplotlib/torch)
export TMPDIR=/path/to/writable/tmp                          # tf env: XLA/ptxas write here; /tmp may be read-only
```

For the parallel finetune runners, also cap CPU threads to avoid host oversubscription:

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 DUET_NUM_THREADS=4
```

### Pretrained weights (UTR-LM)

UTR-LM requires pretrained ESM2 checkpoints from the original repository
(<https://github.com/a96123155/UTR-LM>). Download the pretrained model files and
place them under `benchmarks/baselines/UTR-LM/params/Model/Pretrained/`:

| task | file |
|------|------|
| TE  | `ESM2SI_3.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_MLMLossMin.pkl` |
| MRL | `ESM2SISS_FS4.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_lr1e-05_supervisedweight1.0_structureweight1.0_MLMLossMin_epoch93.pkl` |

The `params` entry may be a symlink to shared storage:

```bash
ln -s /path/to/utr-lm/model benchmarks/baselines/UTR-LM/params
# expects: benchmarks/baselines/UTR-LM/params/Model/Pretrained/<the .pkl files above>
```

> **Run everything from the repository root** (`python benchmarks/duet_bench/...`). `benchlib` is an editable install under `benchmarks/benchlib/benchlib`; running with the working directory set to `benchmarks/` shadows it with a namespace package and breaks `from benchlib import ...`.

---

## 3. Configuration (`config.yaml`)

- `output_dir`: root for all outputs (predictions, metrics, aggregates).
- `gpus`, `jobs_per_gpu`: default parallelism.
- `datasets`: each entry defines `base_path`, `names`, `data_pattern`, split (`kfold` k/seed or pre-split `rank`), `scaling`, `label_col`, input columns/lengths, and `models` (which models run on it).

| dataset | split | scaling | label | note |
|---------|-------|---------|-------|------|
| `mrl_rank` | pre-split train/test | standard | te | MPRA, 50nt 5′UTR |
| `celltype_te` | kfold 10 | standard | logratio_te | per-celltype TE (single-model baselines + DuET single/finetune) |
| `celltype_te_multi` | kfold 10 | none | `^TE_` | multitask TE (all celltypes in one table; DuET/RiboNN multi) |

Cross-validation is 3-way per outer fold (`test = fold_i`, `val = fold_(i-1)`, `train = rest`); DuET/RiboNN merge train+val and re-split into an **inner 9-fold** ensemble (see below). All models on the same dataset share the same split files, so predictions are aligned by transcript ID.

---

## 4. Running the baselines

Each baseline exposes a `run.py` with a common interface:

```bash
conda activate <env> && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# single GPU, single fold
python benchmarks/baselines/optimus5p/run.py --config benchmarks/config.yaml --gpu 0 --folds 0

# parallel across GPUs
python benchmarks/baselines/optimus5p/run.py --config benchmarks/config.yaml --gpus 0,1,2,3 --jobs-per-gpu 3

# a specific dataset only
python benchmarks/baselines/UTR-LM/run.py --config benchmarks/config.yaml --gpus 0 \
    --names celltype_te/hek293t_TE
```

Common flags: `--folds`, `--force` (ignore cache), `--names` (subset of datasets).

`run_all.sh` runs the full suite end to end — baselines, then the DuET runners,
then aggregation — from the repository root:

```bash
bash benchmarks/run_all.sh [GPUS] [JOBS_PER_GPU] [-- EXTRA_ARGS...]
# e.g. bash benchmarks/run_all.sh 0,1 1 -- --folds 0 1
```

Requires `export CONDA_SH=/path/to/conda/etc/profile.d/conda.sh` (or `conda_sh`
set in `config.yaml`). RiboNN and DuET finetune are commented out by default
(cost / pretrain-dir dependency); uncomment inside the script to include them.

### RiboNN (single / multi / finetune)

RiboNN is cell-type-aware and runs in three modes (PyTorch, `benchmark_torch` env, full transcript, inner 9-fold ensemble):

```bash
# single-celltype
python benchmarks/baselines/RiboNN/run.py --config benchmarks/config.yaml --gpus 0,1 --jobs-per-gpu 3

# multitask (all celltypes at once)
python benchmarks/baselines/RiboNN/run.py --config benchmarks/config.yaml --gpu 0 \
    --names celltype_te_multi/multitask

# finetune from a multitask checkpoint
python benchmarks/baselines/RiboNN/run.py --config benchmarks/config.yaml --gpu 0 \
    --finetune <output_dir>/celltype_te_multi/multitask/RiboNN \
    --names celltype_te/pc3_TE
```

`--predict-only` reruns inference from existing checkpoints (no training) and writes `inner_<i>/test_pred.npz`; it never deletes checkpoints (`--force` does).

---

## 5. DuET benchmarks (`duet_bench/`)

The DuET model config is loaded from `model_configs/duet.yaml` (architecture and
training params). The ensemble runners (`run_multitarget.py`, `run_finetune.py`)
save per-inner test predictions to `inner_<i>/test_pred.npz`; the plain-CV runners
(`run_single.py`, `run_multi_cv.py`) save one `predictions.tsv` per outer fold.

```bash
conda activate benchmark_torch && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# (a) single-celltype (train from scratch, 10-fold)
python benchmarks/duet_bench/run_single.py --config benchmarks/config.yaml --gpus 0,1

# (b) multitask, plain 10-fold CV (one multi-target model per fold, no inner ensemble)
python benchmarks/duet_bench/run_multi_cv.py --config benchmarks/config.yaml --gpus 0,1

# (c) multitask pretrain (one model, all celltypes; inner 9-fold, top-5 ensemble)
python benchmarks/duet_bench/run_multitarget.py --config benchmarks/config.yaml --gpus 0,1 --jobs-per-gpu 1

# (d) finetune per celltype from the multitask checkpoint (two-phase: head, then full)
python benchmarks/duet_bench/run_finetune.py --config benchmarks/config.yaml \
    --gpus 0,1 --jobs-per-gpu 3 \
    --pretrain-dir <output_dir>/celltype_te_multi/multitask/DuET_multi
```

Flags: `--folds`, `--names celltype_te/<ct>_TE`, `--force`, `--predict-only` (reuse checkpoints, only regenerate `test_pred.npz`).

**Ensembling.** For each outer fold, 9 inner CV models are trained; the **top-5 by validation R²** are averaged. `inner_0` alone is the non-ensemble baseline.

---

## 6. Output structure

```text
{output_dir}/
├── celltype_te/<ct>_TE/
│   ├── {optimus5p,framepool,translatelstm,UTR-LM}/fold_<F>/predictions.tsv , metrics.json
│   ├── DuET/fold_<F>/predictions.tsv                       # single
│   ├── DuET_ft/fold_<F>/inner_<I>/test_pred.npz            # finetune
│   └── RiboNN_ft/fold_<F>/inner_<I>/test_pred.npz
├── celltype_te_multi/<name>/
│   ├── splits/fold_<F>/indices.json                           # shared train/val/test
│   ├── DuET_multi/fold_<F>/inner_<I>/test_pred.npz         # multitask (N,75)
│   └── RiboNN/fold_<F>/inner_<I>/test_pred.npz
└── collect_all/                                               # aggregates (see §7)
```

- `predictions.tsv`: `y_true, y_pred` (baselines) or `txID, y_true, y_pred` (DuET/RiboNN), original scale.
- `test_pred.npz`: `pred`, `txid`, `te_cols`, `val_r2`, `val_loss` (ft also stores raw-scale `pred` + `pred_scaled`).
- `metrics.json`: pearson, spearman, r2, rmse (+ timings).

---

## 7. Aggregation & analysis (`duet_bench/`)

These are **compute-free** (no GPU); they read the saved predictions/npz. Run from the repo root.

```bash
# 1) collect predictions → long-format CSV per task
python benchmarks/duet_bench/collect_all.py --task single --config benchmarks/config.yaml
python benchmarks/duet_bench/collect_all.py --task multi  --config benchmarks/config.yaml
python benchmarks/duet_bench/collect_all.py --task ft     --config benchmarks/config.yaml
#   → collect_all/{single,multi,ft}_long.csv
#     columns: task, celltype, outer_fold, method, pearson, spearman, r2, rmse, n

# 2) combined celltype × method summary table
python benchmarks/duet_bench/method_summary.py            # → collect_all/table_all_spearman.csv

# 3) per-celltype grouped-bar figures (needs LD_LIBRARY_PATH)
python benchmarks/duet_bench/plot_celltype.py             # → collect_all/figs/celltype_*.png

# 4) pooled Pearson R / R² + variance decomposition (across genes and cell types)
python benchmarks/duet_bench/pooled_rsquared.py --config benchmarks/config.yaml
#   → collect_all/pooled_rsquared.tsv
```

### Method naming (in `*_long.csv`)

| method | meaning |
|--------|---------|
| `duet` | DuET single model, `inner_0` (no ensemble) |
| `duet_top5` | DuET, top-5 inner ensemble |
| `ribonn`, `ribonn_top5` | RiboNN inner_0 / top-5 ensemble |
| `mix_duet_top5_ribonn_top5` | average of DuET-top5 and RiboNN-top5 (transcript-matched) |
| `mix_duet_top3_ribonn_top3` | same, top-3 each |
| `mix_duet_ribonn_pool_top5` | pool all DuET+RiboNN inners, take top-5 by val R², average |
| `*_common` | DuET restricted to the transcript set common to RiboNN (fair comparison; RiboNN drops extreme-length transcripts) |

### `pooled_rsquared.py` (reviewer metric)

Reports, per model, the **pooled Pearson R** and **R² (fraction of total dataset variance explained across genes and cell types)**, plus a decomposition into between-cell-type vs within-cell-type variance and the R² captured in each. Baselines are pooled from per-celltype `predictions.tsv`; DuET/RiboNN from multitask `inner_0` (non-ensemble). Excludes the `all-celltype` pseudo-column. Restrict cell types with `--celltypes`.

---

## 8. Changes from original code

Kept minimal and documented here for reproducibility. Each baseline's original code is available in its upstream GitHub repository.

**RiboNN** (`baselines/RiboNN/src/`)
- `data.py`: `setup_from_df()` for external DataFrame injection (multitask); CDS `assert` → `WARNING` (Ensembl CDS excludes stop codon).
- `model.py`: `on_validation_epoch_end` handles NaN targets in multi-target mode (per-column masking) and skips NaN losses/empty batches.
- `utils/helpers.py`: `masked_mse_loss` returns a zero-grad loss when a batch has no valid targets.

**UTR-LM**: DDP removed (single-GPU); global state replaced with explicit params; sequences truncated to 100 nt. Import in an isolated subprocess — importing the module runs top-level code (argparse, `os.chdir`) that corrupts global state.

**FramePool**: Keras backend upgraded v2.2.4 → v3.10.0; inputs padded/truncated to 100 nt.

**TranslateLSTM**: ported from Keras to PyTorch; dual encoder (5′UTR + CDS, 100 nt each) joined by `txID`; uses precomputed sequence features.

---

## 9. Notes & gotchas

- **Run from repo root** (benchlib namespace shadowing, §2).
- **Split consistency**: DuET and RiboNN share `celltype_te_multi/<name>/splits`; predictions are matched by `txID`, so exact inner-split alignment is not required for ensembling.
- **Scale**: multitask uses `scaling: none` (predictions and targets are on the log-ratio TE scale); finetune scales per-celltype internally and stores inverse-transformed `pred`.
- **RiboNN cost**: full-transcript + inner 9-fold ≈ 20–50 h per celltype (dataset-dependent). Plan multi/finetune runs accordingly.
- **Resuming**: runners skip folds with an existing `metrics.json`; re-running the same command resumes. If a filesystem glitch kills workers mid-run, just relaunch.