#!/bin/bash
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD
#
# Ribo-seq / RNA-seq preprocessing pipeline (steps 1-5). Sourced by run.sh.
# Paths and tool locations are configurable via environment variables; the
# defaults below assume the tools are on PATH. Set DATA-related variables to
# your own reference/index locations before running.

# ---- paths (override via environment) --------------------------------------
# Working directory holding per-sample <GSE>/<SRR>/ subdirectories.
BASE_DIR="${BASE_DIR:-./data}"
# Directory containing this pipeline's helper scripts/binaries.
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# ---- external tools (override via environment; default: found on PATH) -----
PREFETCH="${PREFETCH:-prefetch}"           # sra-tools
FASTERQ_DUMP="${FASTERQ_DUMP:-fasterq-dump}"  # sra-tools
BOWTIE2="${BOWTIE2:-bowtie2}"
RENAME="${RENAME:-rename.sh}"              # bbmap
CUTADAPT="${CUTADAPT:-cutadapt}"
FASTP="${FASTP:-fastp}"

# ---- reference data (defaults are repo-relative; override via environment) --
# Adapter FASTA for trimming (bundled under indices/).
ADAPTERS="${ADAPTERS:-${SCRIPT_DIR}/indices/total_adapters.fa}"
# bowtie2 index of contaminant sequences (rRNA/tRNA/phiX/lambda), bundled under indices/.
FILTER_INDEX="${FILTER_INDEX:-${SCRIPT_DIR}/indices/human_rtrna_phix_lambda}"
# bowtie2 index of the DuET transcriptome. Build it from the transcriptome FASTA
# in the downloaded datasets (see README): bowtie2-build datasets/transcriptome/
# duet_transcriptome.selected.fa <this prefix>.
BOWTIE2_INDEX="${BOWTIE2_INDEX:-${SCRIPT_DIR}/indices/duet_transcriptome}"
# CDS valid-window intervals (SAF), from the downloaded datasets.
CDS_WINDOW="${CDS_WINDOW:-${SCRIPT_DIR}/../datasets/transcriptome/duet_transcriptome_valid_window.saf}"

# ---- helper scripts/binaries in SCRIPT_DIR ---------------------------------
FASTQ_STAT="${SCRIPT_DIR}/fastq_stats"     # compile fastq_stats.c (see README)
RIBO_QC2="${SCRIPT_DIR}/riboseq_qc_2.R"
QUANTIFY="${SCRIPT_DIR}/quantify"          # compile quantify.cpp (see README)

FASTERQ_TMP="${FASTERQ_TMP:-${TMPDIR:-/tmp}/fasterq-dump_temp}"
STAT_SUFFIX=.read_stat.tsv

log() {
    echo "$(date +%T) - ${CURRENT_ACTION}: $*"
}

sample_fields() {
    local line="$1"
    IFS=$'\t' read -r gse srx srr gsm condition sampletype extra  <<< "$line"
    sampledir="${BASE_DIR}/${gse}/${srr}"
    sampleprefix="${sampledir}/${srr}"
}

alignment_fields() {
    local line="$1"
    IFS=$'\t' read -r gse srx srr gsm files libtype condition sampletype extra  <<< "$line"
    sampledir="${BASE_DIR}/${gse}/${srr}"
    sampleprefix="${sampledir}/${srr}"
	if [[ "${files}" == *";"* ]]; then
		local r1="${files%;*}"
		local r2="${files#*;}"
		inputs="-1 ${r1} -2 ${r2}"
	else
		inputs="-U ${files}"
	fi
	files="${files//[;,]/ }"
}

quantification_fields() {
    local line="$1"
    IFS=$'\t' read -r gse srx srr gsm length condition sampletype extra  <<< "$line"
    sampledir="${BASE_DIR}/${gse}/${srr}"
    sampleprefix="${sampledir}/${srr}"
}

restore_if_exists() {
    local f
    for f in "$@"; do
		[[ -e "$f" ]] && lfs hsm_restore $(realpath "$f")
    done
}

release_if_exists() {
    local f
    for f in "$@"; do
		[[ -e "$f" ]] && lfs hsm_release $(realpath "$f")
    done
}

mark_pair_dummy_if_needed() {
    [[ -f ${sampleprefix}_1.fastq ]] && touch "${sampleprefix}.fastq"
}

with_lock() {
    local outsuffix="$1"
    local action="$2"
	CURRENT_ACTION="$action"
	CURRENT_OUTSUFFIX="$outsuffix"

	mkdir -p "${sampledir}"

    if [[ -e ${sampleprefix}.${outsuffix}.checkin ]]; then
        log "Skipping ${sampledir} - checkin found"
        return 0
    fi

    if [[ -e ${sampleprefix}.${outsuffix}.error ]]; then
        log "Skipping ${sampledir} - error found"
        return 0
    fi

    if [[ -e ${sampleprefix}.${outsuffix} ]]; then
        log "${sampleprefix}.${outsuffix} exists. continuing..."
        return 0
    fi

    log "Trying to lock ${sampledir} ..."
    sleep $((1 + RANDOM % 5))
    if [[ -e ${sampleprefix}.${outsuffix}.checkin ]]; then
        log "${sampledir} locked by other process."
        return 0
    fi

    log "Locking ${sampledir}"
    touch "${sampleprefix}.${outsuffix}.checkin"

    log "Processing ${sampledir} ..."
    "$action"
    local code=$?

    if [[ $code -ne 0 ]]; then
        log "Error processing ${sampleprefix}.${outsuffix}"
        rm -f "${sampleprefix}.${outsuffix}"
        mv -f "${sampleprefix}.${outsuffix}.checkin" "${sampleprefix}.${outsuffix}.error"
        return $code
    fi

    log "Job finished."
    rm -f "${sampleprefix}.${outsuffix}.checkin"
    sleep $((3 + RANDOM % 5))
    return 0
}

process_samples() {
    local outsuffix="$1"
    local action="$2"
    local metadata_file="$3"
	local first=1

    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" ]] && continue
		 if [[ $first -eq 1 ]]; then
            first=0
            continue
        fi
        sample_fields "$line"
        with_lock "$outsuffix" "$action"
    done < "$metadata_file"
}

process_alignments() {
    local outsuffix="$1"
    local action="$2"
    local metadata_file="$3"
	local first=1

    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" ]] && continue
		 if [[ $first -eq 1 ]]; then
            first=0
            continue
        fi
        alignment_fields "$line"
        with_lock "$outsuffix" "$action"
    done < "$metadata_file"
}

process_quantifications() {
    local outsuffix="$1"
    local action="$2"
    local metadata_file="$3"
	local first=1

    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" ]] && continue
		 if [[ $first -eq 1 ]]; then
            first=0
            continue
        fi
        quantification_fields "$line"
        with_lock "$outsuffix" "$action"
    done < "$metadata_file"
}

download_sras() {
    local srr_local
    local gse_local
    srr_local="$(basename "${sampledir}")"
    gse_local="$(basename "$(dirname "${sampledir}")")"

    [[ -e ${sampleprefix}.sra.lock ]] && return 0
    [[ -f ${sampleprefix}.sralite ]] && return 0

    ("$PREFETCH" --resume yes -X u -O "${BASE_DIR}/${gse_local}" "${srr_local}") || rm -f "${sampleprefix}.sra.lock"
	release_if_exists "${sampleprefix}.sra" "${sampleprefix}.sralite"
}

unpack_sras_to_fastq() {
    local code
    if [[ -f ${sampleprefix}_1.fastq && -f ${sampleprefix}_2.fastq ]]; then
        log "Paired end fastq found. Generating dummy single end file..."
        touch "${sampleprefix}.fastq"
        code=0
    else
        restore_if_exists "${sampleprefix}.sra" "${sampleprefix}.sralite" "${sampleprefix}.sra.lock"
        if [[ -f ${sampleprefix}.sralite ]]; then
			"$FASTERQ_DUMP" -e "${THREADS}" -m 32G -M 15 -pvf -t "${FASTERQ_TMP}" -O "${sampledir}" "${sampleprefix}.sralite"
        else
            "$FASTERQ_DUMP" -e "${THREADS}" -m 32G -M 15 -pvf -t "${FASTERQ_TMP}" -O "${sampledir}" "${sampleprefix}.sra"
        fi
        code=$?
        [[ $code -eq 0 ]] && mark_pair_dummy_if_needed
    fi
    sleep 1
    release_if_exists "${sampleprefix}.sra" "${sampleprefix}.sralite" "${sampleprefix}.sra.lock"
    sleep 3
    release_if_exists "${sampleprefix}.fastq" "${sampleprefix}_1.fastq" "${sampleprefix}_2.fastq"
    return $code
}

trim_reads() {
    local code
    local trim_param
    if [[ "${sampletype}" == "rpf" ]]; then
		trim_param="-e 0.10 -O 5 --no-indels --cores=${THREADS}"
	else
		trim_param="--cut_right --cut_window_size 4 --cut_mean_quality 20 -q 20 -u 30 -n 5 -l 35 -w ${THREADS}"
    fi

    if [[ -f ${sampleprefix}_1.fastq ]]; then
        restore_if_exists "${sampleprefix}_1.fastq" "${sampleprefix}_2.fastq"
        "${FASTQ_STAT}" "${sampleprefix}_1.fastq" "${sampleprefix}_2.fastq" > "${sampleprefix}${STAT_SUFFIX}"

        if [[ "${sampletype}" == "rpf" ]]; then
            "${CUTADAPT}" -a "file:${ADAPTERS}" -A "file:${ADAPTERS}" -m 15:15 ${trim_param} \
                -o "${sampleprefix}_trimmed.1.fastq" -p "${sampleprefix}_trimmed.2.fastq" \
                "${sampleprefix}_1.fastq" "${sampleprefix}_2.fastq" \
                &> "${sampleprefix}.trimmed.log"
        else
            "${FASTP}" --adapter_fasta "${ADAPTERS}" ${trim_param} \
                --detect_adapter_for_pe -c \
                -i "${sampleprefix}_1.fastq" -I "${sampleprefix}_2.fastq" \
                -o "${sampleprefix}_trimmed.1.fastq" -O "${sampleprefix}_trimmed.2.fastq" \
                -j "${sampleprefix}.trimmed.json" -h /dev/null 2> "${sampleprefix}.trimmed.log"
        fi
        code=$?
        if [[ $code -eq 0 ]]; then
            mv "${sampleprefix}_trimmed.1.fastq" "${sampleprefix}_1.trimmed.fastq"
            mv "${sampleprefix}_trimmed.2.fastq" "${sampleprefix}_2.trimmed.fastq"
            touch "${sampleprefix}.trimmed.fastq"
            "${FASTQ_STAT}" "${sampleprefix}_1.trimmed.fastq" "${sampleprefix}_2.trimmed.fastq" >> "${sampleprefix}${STAT_SUFFIX}"
        fi
        release_if_exists "${sampleprefix}_1.fastq" "${sampleprefix}_2.fastq"
    else
        restore_if_exists "${sampleprefix}.fastq"
        "${FASTQ_STAT}" "${sampleprefix}.fastq" > "${sampleprefix}${STAT_SUFFIX}"

        if [[ "${sampletype}" == "rpf" ]]; then
            "${CUTADAPT}" -a "file:${ADAPTERS}" -m 15 ${trim_param} \
                -o "${sampleprefix}_trimmed.un.fastq" "${sampleprefix}.fastq" \
                &> "${sampleprefix}.trimmed.log"
        else
            "${FASTP}" --adapter_fasta "${ADAPTERS}" ${trim_param} \
                -i "${sampleprefix}.fastq" -o "${sampleprefix}_trimmed.un.fastq" \
                -j "${sampleprefix}.trimmed.json" -h /dev/null 2> "${sampleprefix}.trimmed.log"
        fi
        code=$?
        if [[ $code -eq 0 ]]; then
            mv "${sampleprefix}_trimmed.un.fastq" "${sampleprefix}.trimmed.fastq"
            "${FASTQ_STAT}" "${sampleprefix}.trimmed.fastq" >> "${sampleprefix}${STAT_SUFFIX}"
        fi
        release_if_exists "${sampleprefix}.fastq"
    fi
    return $code
}

filter_reads() {
    local code
    if [[ -f ${sampleprefix}_1.fastq ]]; then
        restore_if_exists "${sampleprefix}_1.trimmed.fastq" "${sampleprefix}_2.trimmed.fastq"
        "${BOWTIE2}" -x "${FILTER_INDEX}" --un-conc "${sampleprefix}.filtered.fastq" \
            -1 "${sampleprefix}_1.trimmed.fastq" -2 "${sampleprefix}_2.trimmed.fastq" \
            -p "${THREADS}" --time 2> "${sampleprefix}.filtered.log" 1> /dev/null && \
        "${RENAME}" fixsra=t -Xmx64G \
            in="${sampleprefix}.filtered.1.fastq" in2="${sampleprefix}.filtered.2.fastq" \
            out="${sampleprefix}_1.filtered.fastq" out2="${sampleprefix}_2.filtered.fastq" && \
        rm -f "${sampleprefix}.filtered.1.fastq" "${sampleprefix}.filtered.2.fastq" && \
        touch "${sampleprefix}.filtered.fastq"
        code=$?
        release_if_exists "${sampleprefix}_1.trimmed.fastq" "${sampleprefix}_2.trimmed.fastq"
        [[ $code -eq 0 ]] && "${FASTQ_STAT}" "${sampleprefix}_1.filtered.fastq" "${sampleprefix}_2.filtered.fastq" >> "${sampleprefix}${STAT_SUFFIX}"
    else
        restore_if_exists "${sampleprefix}.trimmed.fastq"
        "${BOWTIE2}" -x "${FILTER_INDEX}" -U "${sampleprefix}.trimmed.fastq" \
            --un "${sampleprefix}.filtered.un.fastq" -p "${THREADS}" --time 1> /dev/null \
            2> "${sampleprefix}.filtered.log" \
			&& mv "${sampleprefix}.filtered.un.fastq" "${sampleprefix}.filtered.fastq"
        code=$?
        release_if_exists "${sampleprefix}.trimmed.fastq"
        [[ $code -eq 0 ]] && "${FASTQ_STAT}" "${sampleprefix}.filtered.fastq" >> "${sampleprefix}${STAT_SUFFIX}"
    fi
    return $code
}

align_reads_to_transcriptome() {
    local code
	local align_param
	# variable 'inputs' and 'files' from upper namespace; branching is unnecessary for library type
	if [[ "${sampletype}" == "rpf" ]]; then
		align_param="--very-sensitive -L 15 -N 0 --no-unal"
	else
		align_param="--very-sensitive -L 20 -N 0 --no-mixed --no-discordant --no-unal"
	fi
	restore_if_exists ${files}
	"${BOWTIE2}" -x "${BOWTIE2_INDEX}" ${inputs} \
		${align_param} -p "$((THREADS - THREADS/4))" -S - --time 2> "${sampleprefix}.bowtie2.log" \
		| samtools sort -@ "$((THREADS/4))" -m 4G -O BAM -o "${sampleprefix}.bam" 
	code=$?
	release_if_exists ${files}
	if ! grep -q 'alignment rate' "${sampleprefix}.bowtie2.log"; then
		code=42
		rm -f "${sampleprefix}.bam"
	fi
    return $code
}

riboseq_qc() {
    local code
	local libtype
    if [[ -f ${sampleprefix}_1.fastq ]]; then
		libtype="pe"
	else
		libtype="se"
	fi
	Rscript "${RIBO_QC2}" "${sampleprefix}.bam" "${CDS_WINDOW}" "${libtype}" \
		"${sampleprefix}.qc.2.tsv" "${sampleprefix}.qc.2.length.tsv" &> "${sampleprefix}.qc.2.log"
	code=$?
	return $code
}

quantify_alignments() {
	local code
	restore_if_exists "${sampleprefix}.bam"
    if [[ -f ${sampleprefix}_1.fastq ]]; then
		"${QUANTIFY}" "${sampleprefix}.bam" "pe" "${CDS_WINDOW}" "${length}" "${THREADS}" \
			1> "${sampleprefix}.cpm.tmp.tsv" 2> "${sampleprefix}.quant.log"
		code=$?
	else
		"${QUANTIFY}" "${sampleprefix}.bam" "se" "${CDS_WINDOW}" "${length}" "${THREADS}" \
			1> "${sampleprefix}.cpm.tmp.tsv" 2> "${sampleprefix}.quant.log"
		code=$?
	fi
	if [[ $code -eq 0 ]]; then
		mv -f "${sampleprefix}.cpm.tmp.tsv" "${sampleprefix}.cpm.tsv"
	fi
	release_if_exists "${sampleprefix}.bam"
	return $code
}

run_step() {
    case "$STEP" in
        1|download_unpack)
            process_samples sra download_sras "$METADATA_FILE"
            process_samples fastq unpack_sras_to_fastq "$METADATA_FILE"
            ;;
        2|trim_filter)
            process_samples trimmed.fastq trim_reads "$METADATA_FILE"
            process_samples filtered.fastq filter_reads "$METADATA_FILE"
            ;;
        3|align)
            process_alignments bam align_reads_to_transcriptome "$METADATA_FILE" # sample_to_alignment
            ;;
        4|riboseq_qc)
            process_samples qc.2.tsv riboseq_qc "$METADATA_FILE"
            ;;
        5|quant) 
            process_quantifications cpm.tsv quantify_alignments "$METADATA_FILE"
            ;;
        *)
            echo "Usage: $0 [step] [threads] [metadata_file]"
            echo "step: 1|2|3|4|5 or download_unpack|trim_filter|align|dedup|quant"
            return 1
            ;;
    esac
}

cleanup() {
    log "killing child processes"
    kill -SIGTERM 0 2>/dev/null
    wait
    [[ -n "$sampleprefix" && -n "$CURRENT_OUTSUFFIX" ]] && rm -f "${sampleprefix}.${CURRENT_OUTSUFFIX}.checkin"
    exit 1
}
