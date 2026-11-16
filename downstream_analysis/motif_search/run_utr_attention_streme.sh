#!/usr/bin/env bash
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD
#
# run_utr_attention_streme.sh - STREME on 5'UTR attention local-peak k-mers
# vs a position-matched control set.
#
# Inputs (from attribution_score_and_attention.ipynb attention-export cell):
#   data/utr_attn_peak.fa     primary: attention-spike-centered k-mers
#   data/utr_attn_control.fa  control: position-matched non-peak sites
# Usage:  conda activate motif; bash run_utr_attention_streme.sh
set -euo pipefail

# ---- config ----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="${SCRIPT_DIR}/data"
OUT="${SCRIPT_DIR}/results/utr_attention_streme"
PEAK="${DATA}/utr_attn_peak.fa"
CTRL="${DATA}/utr_attn_control.fa"

# MEME_BIN: path to a local MEME Suite bin/ to prepend to PATH (optional).
# Set this if your MEME build is not already on PATH,
# e.g. export MEME_BIN=/path/to/meme/bin.
# TMPDIR: temp dir for STREME's background estimation (default /tmp). Set to a
# writable path if /tmp is unavailable, e.g. export TMPDIR=/scratch/$USER.
export TMPDIR="${TMPDIR:-/tmp}"
export MEME_TEMP_DIR="${TMPDIR}"
# ----------------------------------------------------------------------------

[[ -n "${MEME_BIN:-}" ]] && export PATH="${MEME_BIN}:${PATH}"
mkdir -p "${TMPDIR}" "${OUT}"
command -v streme >/dev/null || { echo "ERROR: streme not found on PATH (activate the MEME env, e.g. conda activate motif)"; exit 1; }

for f in "${PEAK}" "${CTRL}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing ${f} (run the notebook attention-export cell first)"; exit 1; }
done
echo "STREME | peak=$(grep -c '>' "${PEAK}") control=$(grep -c '>' "${CTRL}")"

# Pre-compute a 0-order background from the control set and pass it via --bfile,
# avoiding STREME's internal /tmp background estimation.
BG="${TMPDIR}/utr_attn_ctrl_bg.txt"
fasta-get-markov -m 0 "${CTRL}" "${BG}"

# short k-mers (7-mer centers): minw 4, maxw 8; control as negative set.
streme --p "${PEAK}" --n "${CTRL}" --bfile "${BG}" --dna --minw 4 --maxw 8 --nmotifs 15 \
       --oc "${OUT}" --verbosity 1
echo "Done -> ${OUT}/streme.html  (motifs: ${OUT}/streme.txt)"
