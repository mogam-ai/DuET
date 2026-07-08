#!/usr/bin/env bash
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD
#
# run_xstreme.sh - XSTREME motif discovery for DuET 5'UTR TE subsets.
#
# For each condition: high TE = primary (--p), low TE = control (--n).
# de novo discovery (STREME + MEME) + optional known-motif enrichment (SEA, --m) + TOMTOM.
#
# Usage:
#   conda activate motif; bash run_xstreme.sh [N_CORES] [FILTER]
#     N_CORES : MEME parallel processes (default 8). Only MEME is MPI-parallel;
#               STREME/SEA/TOMTOM are single-threaded.
#     FILTER  : condition substring filter (default "all"). e.g. te20, 5p50, 3p100
#
# Environment variables:
#   MOTIF_DB : known-motif MEME DB (if set, runs SEA enrichment via --m).
#   ALPH     : alphabet mode. "dna2rna" (default) | "rna" | "dna".
#   MEME_BIN : path to a local MEME Suite bin/ to prepend to PATH (optional).
#              Set this if your MEME build is not already on PATH,
#              e.g. export MEME_BIN=/path/to/meme/bin.
#   TMPDIR   : temp dir for MEME (default /tmp). Set to a writable path if
#              /tmp is unavailable, e.g. export TMPDIR=/scratch/$USER.
#
# Requires the MEME Suite (5.5.9) on PATH - e.g. conda env 'motif'.
set -euo pipefail

# ---- config ----------------------------------------------------------------
N_CORES="${1:-8}"
FILTER="${2:-all}"
ALPH="${ALPH:-dna2rna}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
OUT_ROOT="${SCRIPT_DIR}/results/xstreme"
export TMPDIR="${TMPDIR:-/tmp}"
export MEME_TEMP_DIR="${TMPDIR}"
# ----------------------------------------------------------------------------

[[ -n "${MEME_BIN:-}" ]] && export PATH="${MEME_BIN}:${PATH}"
mkdir -p "${OUT_ROOT}" "${TMPDIR}"
command -v xstreme >/dev/null || { echo "ERROR: xstreme not found on PATH (activate the MEME env, e.g. conda activate motif)"; exit 1; }

# alphabet flag
case "${ALPH}" in
  dna2rna) ALPH_FLAG="--dna2rna" ;;
  rna)     ALPH_FLAG="--rna" ;;
  dna)     ALPH_FLAG="--dna" ;;
  *) echo "ERROR: ALPH must be dna2rna|rna|dna"; exit 1 ;;
esac

# optional known-motif DB
DB_FLAG=()
if [[ -n "${MOTIF_DB:-}" ]]; then
  [[ -f "${MOTIF_DB}" ]] || { echo "ERROR: MOTIF_DB not found: ${MOTIF_DB}"; exit 1; }
  DB_FLAG=(--m "${MOTIF_DB}")
  echo "Known motif DB: ${MOTIF_DB}"
fi

echo "============================================================"
echo "XSTREME run | cores(MEME)=${N_CORES} | alphabet=${ALPH} | filter=${FILTER}"
echo "data=${DATA_DIR}  out=${OUT_ROOT}"
echo "============================================================"

# conditions derived from *_high.fa files
shopt -s nullglob
declare -a CONDS=()
for hf in "${DATA_DIR}"/*_high.fa; do
  cond="$(basename "${hf}" _high.fa)"
  if [[ "${FILTER}" == "all" || "${cond}" == *"${FILTER}"* ]]; then
    CONDS+=("${cond}")
  fi
done
[[ ${#CONDS[@]} -gt 0 ]] || { echo "ERROR: no matching conditions for filter='${FILTER}'"; exit 1; }
echo "Conditions (${#CONDS[@]}): ${CONDS[*]}"
echo

for cond in "${CONDS[@]}"; do
  HIGH="${DATA_DIR}/${cond}_high.fa"
  LOW="${DATA_DIR}/${cond}_low.fa"

  if [[ ! -f "${HIGH}" || ! -f "${LOW}" ]]; then
    echo "[SKIP] ${cond}: missing high/low FASTA"; continue
  fi

  # Run BOTH directions: XSTREME only reports motifs enriched in --p (primary).
  #   <cond>_hi : primary=high, control=low  -> enriched in HIGH TE (activators)
  #   <cond>_lo : primary=low,  control=high -> enriched in LOW  TE (repressors, e.g. uORF/GC)
  for dirpair in "hi:${HIGH}:${LOW}" "lo:${LOW}:${HIGH}"; do
    IFS=':' read tag primary control <<< "${dirpair}"
    OUTDIR_D="${OUT_ROOT}/${cond}_${tag}"
    LOG_D="${OUT_ROOT}/${cond}_${tag}.log"
    if [[ -f "${OUTDIR_D}/xstreme.html" ]]; then
      echo "[DONE] ${cond}_${tag}: xstreme.html exists, skipping"; continue
    fi
    echo "[RUN ] ${cond}_${tag}  (primary=$(grep -c '^>' "${primary}") control=$(grep -c '^>' "${control}"))  -> ${OUTDIR_D}"
    xstreme \
      --oc "${OUTDIR_D}" \
      --p "${primary}" \
      --n "${control}" \
      ${ALPH_FLAG} \
      --minw 6 --maxw 15 \
      --meme-p "${N_CORES}" \
      "${DB_FLAG[@]}" \
      > "${LOG_D}" 2>&1 \
      && echo "       ok -> ${OUTDIR_D}/xstreme.html" \
      || { echo "       FAILED (see ${LOG_D})"; tail -5 "${LOG_D}"; }
  done
done

echo
echo "All done. Reports: ${OUT_ROOT}/<condition>_{hi,lo}/xstreme.html"
echo "  _hi = motifs enriched in HIGH TE (activators); _lo = enriched in LOW TE (repressors)"
