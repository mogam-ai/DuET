#!/usr/bin/env bash
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD
#
# run_modisco.sh - TF-MoDISco-lite de novo motif discovery from DuET attributions.
#
# Input npy files come from the 'TF-MoDISco export' cell of
# attribution_score_and_attention.ipynb:
#   data/modisco_<region>_seq_<method>.npy   (N,4,L) one-hot
#   data/modisco_<region>_attr_<method>.npy  (N,4,L) attribution
# (default region=utr, method=shap; override via arguments.)
#
# Two steps: (1) modisco motifs  (2) modisco report (HTML + TOMTOM matching).
# Both run on CPU (numba). Set thread caps (OMP_NUM_THREADS / NUMBA_NUM_THREADS)
# in your environment if you need to limit CPU usage.
#
# Usage:
#   conda activate motif; bash run_modisco.sh [N_CORES] [REGION] [METHOD] [MAX_SEQLETS]
#     N_CORES     : unused (kept for arg-position compat; set thread caps via env)
#     REGION      : utr (default)  [cds needs a separate 4-channel no-embed attribution]
#     METHOD      : shap (default) | ig
#     MAX_SEQLETS : max seqlets per metacluster (default 50000)
#
# Environment variables:
#   MODISCO_DB : known-motif MEME DB (if set, TOMTOM matching in report, -m).
#   WINDOW     : modisco motifs -w (default 500; matches UTR input length).
#   TMPDIR     : temp dir (default /tmp). Set to a writable path if /tmp is
#                unavailable, e.g. export TMPDIR=/scratch/$USER.
#
# Requires modiscolite (2.4.0) on PATH - e.g. conda env 'motif'.
set -euo pipefail

# ---- config ----------------------------------------------------------------
N_CORES="${1:-8}"
REGION="${2:-utr}"
METHOD="${3:-shap}"
MAX_SEQLETS="${4:-50000}"
WINDOW="${WINDOW:-500}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
OUT_ROOT="${SCRIPT_DIR}/results/modisco"
export TMPDIR="${TMPDIR:-/tmp}"
export MEME_TEMP_DIR="${TMPDIR}"
# ----------------------------------------------------------------------------

mkdir -p "${OUT_ROOT}" "${TMPDIR}"
command -v modisco >/dev/null || { echo "ERROR: modisco not found on PATH (activate the modiscolite env, e.g. conda activate motif)"; exit 1; }

SEQ_NPY="${DATA_DIR}/modisco_${REGION}_seq_${METHOD}.npy"
ATTR_NPY="${DATA_DIR}/modisco_${REGION}_attr_${METHOD}.npy"
TAG="${REGION}_${METHOD}"
H5="${OUT_ROOT}/modisco_${TAG}.h5"
REPORT_DIR="${OUT_ROOT}/report_${TAG}"

echo "============================================================"
echo "TF-MoDISco | region=${REGION} method=${METHOD} | threads=${N_CORES}"
echo "  seq =${SEQ_NPY}"
echo "  attr=${ATTR_NPY}"
echo "  max_seqlets=${MAX_SEQLETS} window=${WINDOW}"
echo "============================================================"

for f in "${SEQ_NPY}" "${ATTR_NPY}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing ${f}"; echo "  -> run the 'TF-MoDISco export' cell in attribution_score_and_attention.ipynb first."; exit 1; }
done

# WINDOW guard: must be a positive int <= sequence length L (npy shape (N,4,L)).
# A bad WINDOW (e.g. 0 from a stray env var) extracts zero seqlets and crashes
# modisco deep inside numpy.percentile (IndexError on empty array). Auto-fix to L.
SEQ_L=$(python -c "import numpy as np; print(np.load('${SEQ_NPY}', mmap_mode='r').shape[-1])")
if ! [[ "${WINDOW}" =~ ^[0-9]+$ ]] || [[ "${WINDOW}" -le 0 ]] || [[ "${WINDOW}" -gt "${SEQ_L}" ]]; then
  echo "WARN: WINDOW='${WINDOW}' invalid (seq length L=${SEQ_L}); resetting to L=${SEQ_L}."
  WINDOW="${SEQ_L}"
fi
echo "  (effective window=${WINDOW}, seq length L=${SEQ_L})"

# --- (1) motif discovery ---
if [[ -f "${H5}" ]]; then
  echo "[DONE] motifs: ${H5} exists, skipping discovery"
else
  echo "[RUN ] modisco motifs -> ${H5}"
  modisco motifs \
    -s "${SEQ_NPY}" \
    -a "${ATTR_NPY}" \
    -n "${MAX_SEQLETS}" \
    -w "${WINDOW}" \
    -o "${H5}" \
    -v
  echo "       ok -> ${H5}"
fi

# --- (2) report (HTML + TOMTOM) ---
DB_FLAG=()
if [[ -n "${MODISCO_DB:-}" ]]; then
  [[ -f "${MODISCO_DB}" ]] || { echo "ERROR: MODISCO_DB not found: ${MODISCO_DB}"; exit 1; }
  # modisco TOMTOM runs in DNA (ACGT) space. A RNA (ACGU) MEME DB is rejected
  # with "uses a 'RNA' alphabet when a 'DNA' alphabet was expected" -> empty
  # tomtom output -> pandas EmptyDataError. If the DB is RNA, switch to its
  # DNA-encoded twin (MEME ships <name>.dna_encoded.meme alongside <name>.meme).
  MD_DB="${MODISCO_DB}"
  if grep -qiE '^ALPHABET[= ].*ACGU|^ALPHABET[= ].*RNA' "${MD_DB}" 2>/dev/null; then
    DNA_TWIN="${MD_DB%.meme}.dna_encoded.meme"
    if [[ -f "${DNA_TWIN}" ]]; then
      echo "NOTE: MODISCO_DB is RNA alphabet; modisco TOMTOM needs DNA. Using ${DNA_TWIN}"
      MD_DB="${DNA_TWIN}"
    else
      echo "WARN: MODISCO_DB is RNA alphabet and no DNA-encoded twin found (${DNA_TWIN})."
      echo "      modisco TOMTOM will likely fail; skipping TOMTOM (-m) for report."
      MD_DB=""
    fi
  fi
  if [[ -n "${MD_DB}" ]]; then
    DB_FLAG=(-m "${MD_DB}" -t)
    echo "Known motif DB (TOMTOM): ${MD_DB}"
  fi
fi

if [[ -f "${REPORT_DIR}/motifs.html" ]]; then
  echo "[DONE] report: ${REPORT_DIR}/motifs.html exists, skipping"
else
  echo "[RUN ] modisco report -> ${REPORT_DIR}"
  mkdir -p "${REPORT_DIR}"
  # -l (tomtom-lite): use modiscolite's built-in matcher instead of the external
  # MEME `tomtom`. The external path builds a temp query MEME from each pattern
  # PPM; padded (all-zero) positions in our 5'UTR one-hot yield NaN/non-summing
  # PPM rows -> "probabilities don't sum to 1" -> empty tomtom output ->
  # pandas EmptyDataError. tomtom-lite avoids that (Euclidean distance, no temp meme).
  modisco report \
    -i "${H5}" \
    -o "${REPORT_DIR}" \
    -s "${REPORT_DIR}/" \
    -l \
    "${DB_FLAG[@]}"
  echo "       ok -> ${REPORT_DIR}/motifs.html"
fi

# --- optional MEME export: discovered motifs for use by other tools (TOMTOM/FIMO) ---
MEME_OUT="${OUT_ROOT}/modisco_${TAG}_PFM.meme"
if [[ ! -f "${MEME_OUT}" ]]; then
  echo "[RUN ] modisco meme (PFM) -> ${MEME_OUT}"
  modisco meme -i "${H5}" -t PFM -o "${MEME_OUT}" -q \
    && echo "       ok -> ${MEME_OUT}" \
    || echo "       (meme export skipped/failed; non-critical)"
fi

echo
echo "Done. h5=${H5}  report=${REPORT_DIR}/motifs.html  meme=${MEME_OUT}"
