#!/bin/bash
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD
#
# Entry point for the preprocessing pipeline. Runs one step over all samples in
# a metadata file. See preprocessing/README.md for the step/metadata workflow.

set -u

STEP="${1:-}"
THREADS="${2:-}"
METADATA_FILE="${3:-}"

if [[ -z "${STEP}" || -z "${THREADS}" || -z "${METADATA_FILE}" ]]; then
    echo "Usage: $0 [step] [threads] [metadata_file]"
    exit 1
fi

if [[ ! -f "$METADATA_FILE" ]]; then
    echo "Error: $METADATA_FILE file not found."
    exit 1
fi

source "$(dirname "$0")/pipeline.sh"

trap 'trap " " SIGINT SIGTERM; cleanup' SIGINT SIGTERM

run_step
