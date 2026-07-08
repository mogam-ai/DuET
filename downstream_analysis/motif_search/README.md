# motif_search

Motif discovery on DuET 5'UTR / CDS from three angles, then a single notebook
that parses and compares all results.

- **XSTREME**   — de novo motif discovery on high- vs low-TE 5'UTR subsets
- **TF-MoDISco** — motif discovery from DuET attribution scores
- **Attention STREME** — motifs at DuET attention-peak positions (5'UTR + CDS)

This directory holds **code only**; input data, intermediate artifacts, and tool
results are not committed (see `DATA_LOCATION.md`).

The motif tools (MEME Suite 5.5.9, modiscolite 2.4.0) live in a **separate `motif`
env**, kept apart from the main `duet` env because MEME Suite is a Perl/C CLI
toolkit (not a Python package) with its own bioconda dependency tree. Create it once:
```bash
conda env create -f downstream_analysis/motif_search/environment_motif.yml
```
Set `MEME_BIN` to your MEME `bin/` if it is not already on `PATH`, and `TMPDIR` to
a writable path if `/tmp` is unavailable.

Attribution/attention inputs are exported from the `duet` env (the attribution
notebook); the discovery scripts below then run in the `motif` env.

Run everything from the repository root.

## 1. Prepare inputs

**XSTREME inputs** — high/low-TE 5'UTR FASTA from the TE table:
```bash
python downstream_analysis/motif_search/make_datasets.py
# options: --te-cutoff 10 [20 ...]   --windows 100 [50 ...]   --tol 10
# -> motif_search/data/te{pct}_{full_lenmatch,5pW,3pW}_{high,low}.fa
```

**TF-MoDISco / Attention inputs** — attribution npy and attention-peak FASTA are
exported by the downstream attribution notebook
(`../attribution_score_and_attention.ipynb`); CDS attention FASTA by its CDS cell:
```
data/modisco_utr_{seq,attr}_shap*.npy      # TF-MoDISco
data/utr_attn_{peak,control}.fa            # UTR attention STREME
data/cds_attn_{peak,control}.fa            # CDS attention STREME
```

## 2. Run motif discovery

```bash
conda activate motif

# (a) XSTREME on high vs low TE 5'UTR subsets
bash downstream_analysis/motif_search/run_xstreme.sh
#   -> results/xstreme/<cond>_{hi,lo}/xstreme.html

# (b) TF-MoDISco from attribution scores
bash downstream_analysis/motif_search/run_modisco.sh
#   -> results/modisco/modisco_utr_shap.h5 + report_utr_shap/ + *_PFM.meme

# (c) Attention-peak STREME (5'UTR and CDS)
bash downstream_analysis/motif_search/run_utr_attention_streme.sh
bash downstream_analysis/motif_search/run_cds_attention_streme.sh
#   -> results/{utr,cds}_attention_streme/streme.{html,txt}
```

Optional known-motif enrichment/matching: set `MOTIF_DB` (XSTREME `--m` SEA) or
`MODISCO_DB` (TF-MoDISco TOMTOM) to a MEME-format database.

## 3. Parse and compare

`analyze_motifs_finalized.ipynb` reads the tool outputs above (no GPU) and
produces the unified motif tables, seqlogos, and figures.
