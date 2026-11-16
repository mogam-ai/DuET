#!/bin/bash
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD
#
# Run the full DuET benchmark suite end to end: baselines, DuET runners, then
# aggregation. Each model parallelizes across GPUs internally (via run_parallel).
#
# Run from the repository root (benchlib namespace shadowing otherwise; see README §2).
#
# Usage:
#   bash benchmarks/run_all.sh [GPUS] [JOBS_PER_GPU] [-- EXTRA_ARGS...]
#     GPUS          comma-separated GPU ids       (default 0,1)
#     JOBS_PER_GPU  concurrent jobs per GPU        (default 1)
#     EXTRA_ARGS    passed to every runner         (e.g. --folds 0 1, --force)
#
# Prerequisites (see README §2):
#   - conda envs `utr` and `tf`, each with `pip install -e benchmarks/benchlib`
#   - `pip install -e .` (DuET) in the `benchmark_torch` env
#   - export CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
#
# Cost note: RiboNN (full-transcript, inner 9-fold) is very expensive
# (~20-50 h per cell type). It is included below but commented out by default;
# uncomment to run it.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/config.yaml"

GPUS="${1:-0,1}"
JOBS_PER_GPU="${2:-1}"
shift 2 2>/dev/null || true
[[ "${1:-}" == "--" ]] && shift
EXTRA_ARGS="$@"

# Resolve conda: prefer $CONDA_SH, else the value in config.yaml.
CONDA_SH="${CONDA_SH:-$(grep conda_sh "$CONFIG" | awk '{print $2}')}"
if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
  echo "ERROR: conda profile not found. Set CONDA_SH=/path/to/conda/etc/profile.d/conda.sh"
  exit 1
fi
source "$CONDA_SH"

run() { echo ""; echo "=== $1 ==="; shift; "$@"; }

echo "=== DuET Benchmark Suite ==="
echo "GPUs: $GPUS | Jobs/GPU: $JOBS_PER_GPU | Extra: ${EXTRA_ARGS:-<none>}"

# ---------------------------------------------------------------------------
# Baselines — TensorFlow env
# ---------------------------------------------------------------------------
conda activate benchmark_tf
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
run "optimus5p" python "$SCRIPT_DIR/baselines/optimus5p/run.py" --config "$CONFIG" --gpus "$GPUS" --jobs-per-gpu "$JOBS_PER_GPU" $EXTRA_ARGS
run "framepool" python "$SCRIPT_DIR/baselines/framepool/run.py" --config "$CONFIG" --gpus "$GPUS" --jobs-per-gpu "$JOBS_PER_GPU" $EXTRA_ARGS

# ---------------------------------------------------------------------------
# Baselines + DuET — PyTorch env
# ---------------------------------------------------------------------------
conda activate benchmark_torch
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

run "translatelstm" python "$SCRIPT_DIR/baselines/translatelstm/run.py" --config "$CONFIG" --gpus "$GPUS" --jobs-per-gpu "$JOBS_PER_GPU" $EXTRA_ARGS
run "UTR-LM"        python "$SCRIPT_DIR/baselines/UTR-LM/run.py"        --config "$CONFIG" --gpus "$GPUS" --jobs-per-gpu "$JOBS_PER_GPU" $EXTRA_ARGS
run "GEMORNA"       python "$SCRIPT_DIR/baselines/GEMORNA/run.py"       --config "$CONFIG" --gpus "$GPUS" --jobs-per-gpu "$JOBS_PER_GPU" $EXTRA_ARGS

# RiboNN — very expensive; uncomment to run.
# run "RiboNN"      python "$SCRIPT_DIR/baselines/RiboNN/run.py"        --config "$CONFIG" --gpus "$GPUS" --jobs-per-gpu "$JOBS_PER_GPU" $EXTRA_ARGS

# DuET runners
run "DuET single"    python "$SCRIPT_DIR/duet_bench/run_single.py"      --config "$CONFIG" --gpus "$GPUS" --jobs-per-gpu "$JOBS_PER_GPU" $EXTRA_ARGS
run "DuET multi-cv"  python "$SCRIPT_DIR/duet_bench/run_multi_cv.py"     --config "$CONFIG" --gpus "$GPUS" --jobs-per-gpu "$JOBS_PER_GPU" $EXTRA_ARGS
run "DuET multitask" python "$SCRIPT_DIR/duet_bench/run_multitarget.py"  --config "$CONFIG" --gpus "$GPUS" --jobs-per-gpu "$JOBS_PER_GPU" $EXTRA_ARGS
# DuET finetune needs a multitask pretrain dir; run after run_multitarget:
# run "DuET finetune" python "$SCRIPT_DIR/duet_bench/run_finetune.py"    --config "$CONFIG" --gpus "$GPUS" --jobs-per-gpu "$JOBS_PER_GPU" \
#     --pretrain-dir <output_dir>/celltype_te_multi/multitask/DuET_multi $EXTRA_ARGS

# ---------------------------------------------------------------------------
# Aggregation (no GPU; reads saved predictions)
# ---------------------------------------------------------------------------
run "collect single" python "$SCRIPT_DIR/duet_bench/collect_all.py" --task single --config "$CONFIG"
run "collect multi"  python "$SCRIPT_DIR/duet_bench/collect_all.py" --task multi  --config "$CONFIG"
run "collect ft"     python "$SCRIPT_DIR/duet_bench/collect_all.py" --task ft     --config "$CONFIG"
run "method summary" env BENCH_CONFIG="$CONFIG" python "$SCRIPT_DIR/duet_bench/method_summary.py"
run "pooled R2"      python "$SCRIPT_DIR/duet_bench/pooled_rsquared.py" --config "$CONFIG"

echo ""
echo "=== Done ==="
