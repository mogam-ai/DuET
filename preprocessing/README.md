# Preprocessing: Ribo-seq / RNA-seq to translation efficiency (TE)

Pipeline that turns raw Ribo-seq (RPF) and RNA-seq reads into the per-cell-type
TE tables under `datasets/celltype_te/`. It runs in five staged steps followed
by TE calculation and averaging.

## Workflow

```
metadata (step 1)
   │  run.sh 1  download + unpack SRA -> FASTQ
   │  run.sh 2  trim (cutadapt/fastp) + filter contaminants (bowtie2)
   ▼
summarize_preprocessing.py  ->  metadata (step 2/3, adds files/libraryType)
   │  run.sh 3  align to transcriptome (bowtie2 -> sorted BAM)
   ▼
summarize_alignment.py      ->  metadata (step 3, post-alignment)
   │  run.sh 4  ribo-seq QC (riboseq_qc_2.R)
   ▼
summarize_qc.py             ->  metadata (step 4, adds length)
   │  run.sh 5  quantify (quantify) -> per-sample <SRR>.cpm.tsv
   ▼
calculate_te.py             ->  celltype_te/<celltype>_TE.tsv  (+ QC reports)
   ▼
average_all_celltype_TE.py  ->  all-celltype_TE.tsv
```

`run.sh` runs one step over every sample in a metadata file. Between steps, the
`summarize_*.py` scripts parse logs/QC, write stat reports, and emit the
metadata file for the next step (this is where metadata columns change).

## Usage

```bash
# each step: run.sh <step> <threads> <metadata_file>
bash run.sh 1 8 metadata_step1.tsv     # download + unpack
bash run.sh 2 8 metadata_step1.tsv     # trim + filter
python summarize_preprocessing.py metadata_step1.tsv
bash run.sh 3 8 2.filtered_sample_to_alignment
python summarize_alignment.py
bash run.sh 4 8 3.filtered_sample_after_alignment
python summarize_qc.py
bash run.sh 5 8 4.filtered_sample_to_quantify
# reconstruct gene metadata (txID/utr5/cds/utr3) from the downloaded datasets
python build_gene_metadata.py \
    --fasta ../datasets/transcriptome/duet_transcriptome.selected.fa \
    --features ../datasets/sequence_features.tsv \
    -o gene_metadata.tsv
python calculate_te.py 5.duet_ribonn_merged_samples \
    --data-dir data --output-dir ../datasets/celltype_te \
    --gene-metadata gene_metadata.tsv
python average_all_celltype_TE.py ../datasets/celltype_te/*_TE.tsv \
    -o ../datasets/celltype_te/all-celltype_TE.tsv
```

`run.sh`/`pipeline.sh` are idempotent per sample via `.checkin`/`.error` lock
files, so a step can be re-run to resume.

## Metadata schema (changes per stage)

`example_metadata.tsv` (100 rows) is a step-1 input. Later stages add/rearrange
columns; the `summarize_*.py` scripts produce each stage's file.

| Stage | File | Columns |
|-------|------|---------|
| step 1 | `1.total_sample_to_process` | gse, srx, srr, gsm, condition, type, sampleType, index |
| step 2/3 | `2.filtered_sample_to_alignment` | gse, srx, srr, gsm, files, libraryType, condition, sampletype, celltype, index |
| step 4 | `3.filtered_sample_after_alignment` | gse, srx, srr, gsm, condition, sampletype, celltype, index |
| step 5 / TE | `4.filtered_sample_to_quantify`, `5.duet_ribonn_merged_samples` | gse, srx, srr, gsm, length, condition, sampletype, celltype, index |

`sampletype` is `rpf` (Ribo-seq) or `rna` (RNA-seq); `condition` groups
control/treatment; paired RNA/RPF samples share the same `index`.

## Configuration

Paths and tools in `pipeline.sh` are set via environment variables. Tools
default to `PATH`; reference data must be provided:

```bash
export BASE_DIR=./data                 # per-sample <GSE>/<SRR>/ working dir
export ADAPTERS=/path/to/adapters.fa
export FILTER_INDEX=/path/to/contaminant_bowtie2_index   # rRNA/tRNA/phiX/lambda
export BOWTIE2_INDEX=/path/to/duet_transcriptome_bowtie2_index
export CDS_WINDOW=/path/to/duet_transcriptome_valid_window.saf
# optional tool overrides: PREFETCH, FASTERQ_DUMP, BOWTIE2, CUTADAPT, FASTP, RENAME
```

The transcriptome bowtie2 index and valid-window SAF derive from the DuET
transcriptome (deposited on Zenodo with the datasets).

## Environment

Preprocessing uses a **separate conda env** from the main `duet` env (it needs
sequencing tools and R/Bioconductor, which should not be mixed with the
pytorch-cuda stack). Create it from the bundled spec, then install riboWaltz
(not available on bioconda) from R:

```bash
conda env create -f preprocessing/environment_preprocessing.yml
conda activate duet-preprocessing
R -e 'devtools::install_github("LabTranslationalArchitectomics/riboWaltz", dependencies = TRUE)'
```

Pinned versions (see `environment_preprocessing.yml`):

- **Sequencing tools**: sra-tools 3.1.1, bowtie2 2.5.4, cutadapt 5.2,
  fastp 1.3.1, samtools 1.21, bbmap 39.19 (rename.sh)
- **Python**: numpy 1.26.3, pandas 2.1.4, scipy 1.10.1, pyarrow, tqdm
- **R 4.5.2**: data.table 1.18.2.1, Rsamtools 2.26.0, GenomicAlignments 1.46.0,
  riboWaltz 2.0 (installed via devtools, above)

### Compiling the helper binaries

`pipeline.sh` expects two compiled binaries in this directory (the env provides
the compilers and htslib):

```bash
# fastq_stats: standard C
cc -O2 -o fastq_stats fastq_stats.c -lm

# quantify: C++, requires htslib
c++ -O2 -std=c++17 -o quantify quantify.cpp -lhts
```
