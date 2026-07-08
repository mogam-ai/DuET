#!/usr/bin/env python3
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD
import argparse
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

THRIFT_LIMIT = 1_000_000_000
METADATA_COLS = ["pairIndex", "geoStudy", "cellType", "sampleType", "sampleID"]
MERGED_RNA_PATH = f"RNA_count_table_merged.parquet"
MERGED_RPF_PATH = f"RPF_count_table_merged.parquet"
# Transcriptome metadata (index=txID, columns utr5/cds/utr3), produced by
# build_gene_metadata.py from the transcriptome FASTA + sequence_features.tsv.
# polyA-lacking transcript list is bundled under indices/.
# Override via --gene-metadata / --polyA-lacking-ref.
DEFAULT_GENE_METADATA = "gene_metadata.tsv"
DEFAULT_POLYA_LACKING_TXIDS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "indices", "polyA_lacking_human_genes_transcriptIdRef.tsv",
)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def slugify_local(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "na"


def normalize_condition(value: str) -> str:
    value = value.strip().lower()
    mapping = {
        "all": "all",
        "total": "all",
        "control": "control",
        "untreated": "control",
        "treatment": "treatment",
        "treated": "treatment",
        "case": "treatment",
    }
    if value not in mapping:
        raise ValueError(f"Unsupported condition: {value}")
    return mapping[value]


def normalize_metadata_condition(value: str) -> str:
    return normalize_condition(value)


def normalize_sampletype(value: str) -> str:
    value = value.strip().lower()
    if "rpf" in value or "ribo" in value or "footprint" in value:
        return "RPF"
    if "rna" in value:
        return "RNA"
    raise ValueError(f"Cannot infer assay type from sampletype={value!r}")


def resolve_input_path(data_dir: Path, gse: str, srr: str, suffix: str) -> Path:
    primary = data_dir / gse / srr / f"{srr}.{suffix}"
    if primary.exists():
        return primary

    fallback1 = data_dir / gse / srr / suffix
    if fallback1.exists():
        return fallback1

    matches = list((data_dir / gse / srr).glob(f"{srr}*.{suffix}"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Missing input file for {gse}/{srr} with suffix '{suffix}'")


def read_sample_metadata(metadata_file: str) -> Dict[str, List[dict]]:
    print(f"[1/4] Reading sample metadata for full-sample merge: {metadata_file}")
    groups = {"RNA": [], "RPF": []}
    total = 0
    kept = 0

    with open(metadata_file) as handle:
        next(handle)
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts[:1] == ["gse"] or parts[:1] == ["GSE"]:
                continue
            if len(parts) != 9:
                eprint(f"Warning: malformed metadata row skipped: {line}")
                continue

            gse, srx, srr, gsm, length, condition, sampletype, celltype, pair_index = parts
            total += 1
            assay = normalize_sampletype(sampletype)
            sample_group = normalize_metadata_condition(condition)

            groups[assay].append(
                {
                    "geoStudy": gse,
                    "sampleID": srx,
                    "srr": srr,
                    "cellType": celltype,
                    "pairIndex": str(pair_index),
                    "sampleType": sample_group,
                }
            )
            kept += 1

    print(
        f"    metadata rows kept for merge: {kept}/{total} | "
        f"RNA={len(groups['RNA'])}, RPF={len(groups['RPF'])}"
    )
    return groups


def load_expression_table(
    samples: List[dict],
    data_dir: str,
    suffix: str,
    value_column: int,
    output_path: str,
    label: str,
) -> pd.DataFrame:
    os.makedirs(Path(output_path).parent, exist_ok=True)
    data_dir_path = Path(data_dir)

    if not samples:
        raise RuntimeError(f"No {label} samples matched the metadata filter.")

    sample_series = []
    sample_meta_rows = []
    missing_files = []

    print(f"[2/4] Merging {label} files into one full-sample matrix")
    for item in tqdm(samples, desc=f"Loading {label}", unit="sample"):
        try:
            input_path = resolve_input_path(data_dir_path, item["geoStudy"], item["srr"], suffix)
            df = pd.read_csv(
                input_path,
                sep="\t",
                usecols=[0, value_column],
                header=0,
                low_memory=False,
            )
            gene_col = df.columns[0]
            value_col = df.columns[1]
            df = df.rename(columns={gene_col: "gene", value_col: "value"})
            df["gene"] = df["gene"].astype(str).str.split("|", n=1).str[0]
            df = df.drop_duplicates(subset="gene", keep="first")
            series = pd.Series(df["value"].to_numpy(dtype=np.float32), index=df["gene"], name=item["sampleID"])
            sample_series.append(series)

            sample_meta_rows.append(
                {
                    "sampleID": item["sampleID"],
                    "pairIndex": item["pairIndex"],
                    "geoStudy": item["geoStudy"],
                    "cellType": item["cellType"],
                    "sampleType": item["sampleType"],
                }
            )
        except FileNotFoundError as exc:
            missing_files.append(str(exc))
        except ValueError as exc:
            eprint(f"Warning: failed to parse {item['srr']} ({label}): {exc}")
        except Exception as exc:
            eprint(f"Warning: unexpected error while reading {item['srr']} ({label}): {exc}")

    if missing_files:
        eprint(f"Warning: {len(missing_files)} {label} files were missing.")
        for msg in missing_files[:10]:
            eprint("   ", msg)
        if len(missing_files) > 10:
            eprint(f"    ... {len(missing_files) - 10} more missing files omitted")

    if not sample_series:
        raise RuntimeError(f"No valid {label} files could be loaded.")

    expr = pd.concat(sample_series, axis=1, sort=True).fillna(0.0)
    expr = expr.T
    expr.index.name = "sampleID"
    expr = expr.astype(np.float32)

    meta = pd.DataFrame(sample_meta_rows).drop_duplicates(subset="sampleID").set_index("sampleID")
    merged = meta.join(expr, how="inner")
    merged.to_parquet(output_path, engine="pyarrow", compression="gzip", index=True)

    print(
        f"    {label} merged matrix saved: {output_path} | "
        f"samples={merged.shape[0]}, genes={merged.shape[1] - len(meta.columns)}"
    )
    return merged


def read_merged_table(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow", thrift_string_size_limit=THRIFT_LIMIT)


def filter_merged_table_by_group(df: pd.DataFrame, selected_group: str, label: str) -> pd.DataFrame:
    if "sampleType" not in df.columns:
        raise KeyError(
            f"Merged {label} table does not contain 'sampleType'. "
            f"Rebuild {path if 'path' in locals() else label} merged parquet with this script."
        )

    out = df.copy()
    out["sampleType"] = out["sampleType"].astype(str).map(normalize_metadata_condition)

    available_groups = sorted(out["sampleType"].dropna().unique().tolist())
    print(
        f"[2/4] Applying dynamic group filter to merged {label}: "
        f"selected={selected_group}, available={available_groups}"
    )

    if selected_group == "all":
        print(f"    {label} rows kept after group filter: {len(out)}/{len(df)}")
        return out

    filtered = out[out["sampleType"] == selected_group].copy()
    print(f"    {label} rows kept after group filter: {len(filtered)}/{len(df)}")

    if filtered.empty:
        raise RuntimeError(
            f"No {label} samples remain after filtering merged table by --type={selected_group}. "
            f"If these parquet files were created by an older filtered merge, rebuild without --no-merge."
        )

    return filtered


def split_meta_expr(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.reset_index()
    meta = df[METADATA_COLS].copy()
    expr_cols = df.columns.difference(METADATA_COLS)
    expr = df[expr_cols].copy()
    meta = meta.set_index("sampleID")
    expr.index = meta.index
    return meta, expr


def normalize_expr(expr: pd.DataFrame) -> pd.DataFrame:
    expr = expr.apply(pd.to_numeric, errors="coerce").astype(np.float64)
    expr = expr.replace([np.inf, -np.inf], np.nan)
    expr = expr.where(expr >= 0)
    return expr


def apply_cutoff(expr: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    expr = expr.astype(np.float32)
    if cutoff <= 0:
        expr = expr.mask(expr == 0)
    else:
        expr = expr.mask(expr < cutoff)
    return expr


def apply_threshold_row(row: pd.Series, threshold: float) -> pd.Series:
    x = pd.to_numeric(row, errors="coerce").astype(float)
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.where(x >= 0)
    x = x.where(x >= threshold)
    return x


def load_gene_metadata(metadata_path: str) -> pd.DataFrame:
    print(f"[3/4] Loading gene metadata: {metadata_path}")
    metadata = pd.read_csv(
        metadata_path,
        index_col=0,
        header=0,
        sep="\t",
        keep_default_na=False,
        na_values=[
            "",
            "#N/A",
            "#N/A N/A",
            "#NA",
            "-1.#IND",
            "-1.#QNAN",
            "-NaN",
            "-nan",
            "1.#IND",
            "1.#QNAN",
            "<NA>",
            "N/A",
            "NA",
            "NULL",
            "NaN",
            "n/a",
            "nan",
            "null",
        ],
        low_memory=False,
    )
    metadata["txID"] = metadata.index.astype(str)
    for col in ("utr5", "cds", "utr3"):
        if col not in metadata.columns:
            raise KeyError(f"Missing required column in gene metadata: {col}")
        metadata[col] = metadata[col].fillna("").astype(str)
    return metadata


def load_polya_lacking_txids(path: str) -> pd.Index:
    print(f"[3/4] Loading polyA-lacking transcript reference: {path}")
    ref = pd.read_csv(path, sep="\t", header=0, low_memory=False)
    if "txID" not in ref.columns:
        raise KeyError("polyA-lacking reference must contain a 'txID' column")
    txids = pd.Index(ref["txID"].astype(str).dropna().unique())
    print(f"    polyA-lacking transcripts loaded: {len(txids)}")
    return txids


def collapse_nonpolya_to_others(expr: pd.DataFrame, excluded_txids: pd.Index) -> pd.DataFrame:
    excluded_cols = expr.columns.intersection(pd.Index(excluded_txids))
    kept_cols = expr.columns.difference(excluded_cols)
    out = expr.loc[:, kept_cols].copy()
    if len(excluded_cols) > 0:
        others = expr.loc[:, excluded_cols].sum(axis=1, min_count=1)
        out["others"] = others
    return out


def filter_genes_before_te(
    rna_expr: pd.DataFrame,
    rpf_expr: pd.DataFrame,
    gene_metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shared = rna_expr.columns.intersection(rpf_expr.columns)
    shared = shared.intersection(gene_metadata.index)

    if shared.empty:
        raise RuntimeError("No overlapping genes between RNA, RPF, and gene metadata.")

    seq_meta = gene_metadata.loc[shared, ["utr5", "cds", "utr3"]]
    seq_mask = (
        seq_meta["utr5"].str.len().between(30, 1000)
        & seq_meta["cds"].str.len().between(100, 7000)
        & (seq_meta["utr3"].str.len() > 0)
    )
    filtered_genes = seq_meta.index[seq_mask]

    if filtered_genes.empty:
        raise RuntimeError("No genes survived the sequence-based pre-filter.")

    print(
        f"    gene filtering: shared={len(shared)} -> kept={len(filtered_genes)} "
        f"(sequence-ready transcripts)"
    )
    return rna_expr.loc[:, filtered_genes], rpf_expr.loc[:, filtered_genes]


def pair_samples(
    rna_meta: pd.DataFrame,
    rpf_meta: pd.DataFrame,
    rna_expr: pd.DataFrame,
    rpf_expr: pd.DataFrame,
) -> pd.DataFrame:
    rna_pairs = rna_meta.reset_index()[["sampleID", "pairIndex", "geoStudy", "cellType", "sampleType"]]
    rna_pairs = rna_pairs.rename(columns={"sampleID": "rnaSample"})
    rpf_pairs = rpf_meta.reset_index()[["sampleID", "pairIndex"]].rename(columns={"sampleID": "rpfSample"})

    paired = rna_pairs.merge(rpf_pairs, on="pairIndex", how="inner")
    paired = paired[
        paired["rnaSample"].isin(rna_expr.index) & paired["rpfSample"].isin(rpf_expr.index)
    ].copy()

    if paired.empty:
        raise RuntimeError("No paired RNA/RPF samples remained after matching by pair index.")

    paired = paired.drop_duplicates(subset=["pairIndex", "rnaSample", "rpfSample"])
    print(f"    paired samples available for TE: {len(paired)}")
    return paired


def corr_pair(x, y):
    z = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(z) < 2:
        return np.nan, np.nan
    if z["x"].nunique() < 2 or z["y"].nunique() < 2:
        return np.nan, np.nan
    return pearsonr(z["x"], z["y"])[0], spearmanr(z["x"], z["y"])[0]


def propr_clr_matrix(df):
    x = df.to_numpy(dtype=float, copy=True)
    valid = np.isfinite(x) & (x > 0)
    if not valid.any():
        return np.full_like(x, np.nan, dtype=float)
    min_nonzero = np.nanmin(x[valid])
    x[~valid] = min_nonzero
    log_x = np.log(x)
    return log_x - log_x.mean(axis=1, keepdims=True)


def ilr_basis(d):
    return helmert(d, full=False).T


def clr_to_ilr(clr_mat, basis):
    return clr_mat @ basis


def ilr_to_clr(ilr_mat, basis):
    return ilr_mat @ basis.T


def compositional_te(rpf_df, rna_df):
    common_cols = rpf_df.columns.intersection(rna_df.columns)
    rpf_df = rpf_df[common_cols]
    rna_df = rna_df[common_cols]

    valid_mask = rpf_df.notna() & rna_df.notna()

    rpf_clr = propr_clr_matrix(rpf_df)
    rna_clr = propr_clr_matrix(rna_df)
    basis = ilr_basis(rpf_clr.shape[1])

    rpf_ilr = clr_to_ilr(rpf_clr, basis)
    rna_ilr = clr_to_ilr(rna_clr, basis)

    x = rna_ilr.T
    y = rpf_ilr.T

    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_centered = x - x_mean
    denom = np.sum(x_centered ** 2, axis=0)

    slope = np.full(x.shape[1], np.nan, dtype=float)
    nz = denom > 0
    slope[nz] = np.sum(x_centered[:, nz] * (y[:, nz] - y_mean[nz]), axis=0) / denom[nz]
    intercept = y_mean - slope * x_mean
    resid_ilr = y - (intercept + slope * x)

    bad = ~np.isfinite(resid_ilr).all(axis=0)
    resid_ilr[:, bad] = np.nan

    te_clr = ilr_to_clr(resid_ilr.T, basis)
    te = pd.DataFrame(te_clr, index=rpf_df.index, columns=common_cols)
    te = te.where(valid_mask)
    te = te.dropna(axis=1, how="all")
    return te


def compositional_te_with_nonpolya_others(
    rpf_df: pd.DataFrame,
    rna_df: pd.DataFrame,
    excluded_txids: pd.Index,
) -> pd.DataFrame:
    common_cols = rpf_df.columns.intersection(rna_df.columns)
    excluded_common = common_cols.intersection(pd.Index(excluded_txids))

    rpf_comp = collapse_nonpolya_to_others(rpf_df[common_cols], excluded_common)
    rna_comp = collapse_nonpolya_to_others(rna_df[common_cols], excluded_common)

    te = compositional_te(rpf_comp, rna_comp)
    if "others" in te.columns:
        te = te.drop(columns=["others"])
    allowed_cols = common_cols.difference(excluded_common)
    te = te.reindex(columns=allowed_cols)
    te = te.dropna(axis=1, how="all")
    return te



def calculate_logratio_te(rpf_expr: pd.DataFrame, rna_expr: pd.DataFrame) -> pd.DataFrame:
    common_cols = rpf_expr.columns.intersection(rna_expr.columns)
    rpf_expr = rpf_expr[common_cols]
    rna_expr = rna_expr[common_cols]
    te = np.log2(rpf_expr) - np.log2(rna_expr)
    te = te.replace([np.inf, -np.inf], np.nan)
    te = te.dropna(axis=1, how="all")
    return te


def pair_spearman_from_cpm(rpf_row: pd.Series, rna_row: pd.Series, pseudocount: float) -> float:
    common_cols = rpf_row.index.intersection(rna_row.index)
    x = pd.to_numeric(rpf_row.loc[common_cols], errors="coerce").astype(float)
    y = pd.to_numeric(rna_row.loc[common_cols], errors="coerce").astype(float)

    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return np.nan

    x = np.log2(x[valid].to_numpy() + pseudocount)
    y = np.log2(y[valid].to_numpy() + pseudocount)

    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan

    return float(spearmanr(x, y).statistic)


def max_logratio_te(rpf_row: pd.Series, rna_row: pd.Series, pseudocount: float):
    common_cols = rpf_row.index.intersection(rna_row.index)
    x = pd.to_numeric(rpf_row.loc[common_cols], errors="coerce").astype(float)
    y = pd.to_numeric(rna_row.loc[common_cols], errors="coerce").astype(float)

    valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if valid.sum() == 0:
        return np.nan, np.nan

    logratio = np.log2(x[valid] + pseudocount) - np.log2(y[valid] + pseudocount)
    if logratio.empty:
        return np.nan, np.nan

    max_idx = logratio.idxmax()
    max_val = logratio.loc[max_idx]

    if pd.isna(max_val):
        return np.nan, np.nan

    return float(max_val), str(max_idx)


def _top_fraction(x: pd.Series, k: int) -> float:
    total = x.sum(skipna=True)
    if pd.isna(total) or total <= 0:
        return np.nan

    valid = x.dropna()
    if valid.empty:
        return np.nan

    top_sum = float(valid.nlargest(min(k, len(valid))).sum())
    return float(top_sum / total)


def row_stats(row: pd.Series):
    x = pd.to_numeric(row, errors="coerce").astype(float)
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.where(x >= 0)

    valid_positive = x[x > 0]
    detected = int(valid_positive.shape[0])

    median_cpm = float(x.median(skipna=True)) if x.notna().any() else np.nan

    if x.notna().any():
        max_txid = x.idxmax(skipna=True)
        max_cpm = x.loc[max_txid]
        max_cpm = float(max_cpm) if pd.notna(max_cpm) else np.nan
        max_txid = str(max_txid) if pd.notna(max_cpm) else np.nan
    else:
        max_cpm = np.nan
        max_txid = np.nan

    return {
        "median_cpm": median_cpm,
        "max_cpm": max_cpm,
        "max_txid": max_txid,
        "top1_fraction": _top_fraction(x, 1),
        "top3_fraction": _top_fraction(x, 3),
        "top5_fraction": _top_fraction(x, 5),
        "top10_fraction": _top_fraction(x, 10),
        "detected_genes": detected,
    }


def build_pairwise_status_report(
    paired: pd.DataFrame,
    rna_expr: pd.DataFrame,
    rpf_expr: pd.DataFrame,
    pseudocount: float,
    report_path: str,
) -> pd.DataFrame:
    rows = []
    for rec in paired.itertuples(index=False):
        pair_index = str(rec.pairIndex)
        geo_study = rec.geoStudy
        cell_type = rec.cellType
        sample_type = rec.sampleType
        rpf_sample_id = rec.rpfSample
        rna_sample_id = rec.rnaSample

        if rpf_sample_id not in rpf_expr.index or rna_sample_id not in rna_expr.index:
            continue

        rpf_row = apply_threshold_row(rpf_expr.loc[rpf_sample_id], 0.0)
        rna_row = apply_threshold_row(rna_expr.loc[rna_sample_id], 0.0)

        max_logratio_te_value, max_te_txid = max_logratio_te(rpf_row, rna_row, pseudocount)
        rpf_stats = row_stats(rpf_row)
        rna_stats = row_stats(rna_row)
        pair_spearman = pair_spearman_from_cpm(rpf_row, rna_row, pseudocount)

        rows.append(
            {
                "pairIndex": pair_index,
                "geoStudy": geo_study,
                "cellType": cell_type,
                "sampleType": sample_type,
                "rpf_sampleID": rpf_sample_id,
                "rna_sampleID": rna_sample_id,
                "max_logratio_TE": max_logratio_te_value,
                "max_TE_txID": max_te_txid,
                "rpf_median_cpm": rpf_stats["median_cpm"],
                "rpf_max_cpm": rpf_stats["max_cpm"],
                "rpf_max_txID": rpf_stats["max_txid"],
                "rpf_top1_fraction": rpf_stats["top1_fraction"],
                "rpf_top3_fraction": rpf_stats["top3_fraction"],
                "rpf_top5_fraction": rpf_stats["top5_fraction"],
                "rpf_top10_fraction": rpf_stats["top10_fraction"],
                "rpf_detected_genes": rpf_stats["detected_genes"],
                "rna_median_cpm": rna_stats["median_cpm"],
                "rna_max_cpm": rna_stats["max_cpm"],
                "rna_max_txID": rna_stats["max_txid"],
                "rna_top1_fraction": rna_stats["top1_fraction"],
                "rna_top3_fraction": rna_stats["top3_fraction"],
                "rna_top5_fraction": rna_stats["top5_fraction"],
                "rna_top10_fraction": rna_stats["top10_fraction"],
                "rna_detected_genes": rna_stats["detected_genes"],
                "pair_spearman": pair_spearman,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No valid paired rows were produced for te_pairwise_status_report.tsv.")
    out.to_csv(report_path, sep="\t", index=False, na_rep="NaN")
    return out


def apply_pair_filters(
    pair_report: pd.DataFrame,
    spearman_hard_cutoff: float,
    spearman_soft_cutoff: float,
    rpf_top5_soft_cutoff: float,
) -> pd.DataFrame:
    spearman = pd.to_numeric(pair_report["pair_spearman"], errors="coerce")
    top5 = pd.to_numeric(pair_report["rpf_top5_fraction"], errors="coerce")

    hard_fail = spearman < spearman_hard_cutoff
    soft_fail = (spearman < spearman_soft_cutoff) & (top5 > rpf_top5_soft_cutoff)
    keep_mask = ~(hard_fail | soft_fail)

    pair_report = pair_report.copy()
    pair_report["filter_hard_spearman"] = hard_fail.fillna(False)
    pair_report["filter_soft_spearman_top5"] = soft_fail.fillna(False)
    pair_report["pair_removed"] = (~keep_mask).fillna(False)
    return pair_report


def calculate_te_and_reports(
    rna_df: pd.DataFrame,
    rpf_df: pd.DataFrame,
    gene_metadata: pd.DataFrame,
    polya_lacking_txids: pd.Index,
    rna_cutoff: float,
    rpf_cutoff: float,
    sparsity_cutoff: float,
    pair_metric_pseudocount: float,
    spearman_hard_cutoff: float,
    spearman_soft_cutoff: float,
    rpf_top5_soft_cutoff: float,
    output_dir: str,
):
    print("[4/4] Filtering genes/samples and calculating TE")

    rna_meta, rna_expr = split_meta_expr(rna_df)
    rpf_meta, rpf_expr = split_meta_expr(rpf_df)

    rna_expr = apply_cutoff(rna_expr, rna_cutoff)
    rpf_expr = apply_cutoff(rpf_expr, rpf_cutoff)
    rna_expr, rpf_expr = filter_genes_before_te(rna_expr, rpf_expr, gene_metadata)

    paired = pair_samples(rna_meta, rpf_meta, rna_expr, rpf_expr)

    pair_report_path = "te_pairwise_status_report.tsv"
    pair_report = build_pairwise_status_report(
        paired=paired,
        rna_expr=rna_expr,
        rpf_expr=rpf_expr,
        pseudocount=pair_metric_pseudocount,
        report_path=pair_report_path,
    )
    pair_report = apply_pair_filters(
        pair_report=pair_report,
        spearman_hard_cutoff=spearman_hard_cutoff,
        spearman_soft_cutoff=spearman_soft_cutoff,
        rpf_top5_soft_cutoff=rpf_top5_soft_cutoff,
    )
    pair_report.to_csv(pair_report_path, sep="\t", index=False, na_rep="NaN")

    keep_pair_ids = pair_report.loc[~pair_report["pair_removed"], "pairIndex"].astype(str).tolist()
    dropped_by_pair_filter = int(pair_report["pair_removed"].sum())
    print(
        f"    pairwise filter: total={len(pair_report)}, kept={len(keep_pair_ids)}, "
        f"dropped={dropped_by_pair_filter}"
    )

    paired = paired[paired["pairIndex"].astype(str).isin(keep_pair_ids)].copy()
    if paired.empty:
        raise RuntimeError("All RNA/RPF pairs were removed by pairwise QC filters.")

    rna_aligned = rna_expr.loc[paired["rnaSample"]].copy()
    rpf_aligned = rpf_expr.loc[paired["rpfSample"]].copy()
    pair_index = paired["pairIndex"].astype(str).to_numpy()

    rna_aligned.index = pair_index
    rpf_aligned.index = pair_index

    common_genes = rna_aligned.columns.intersection(rpf_aligned.columns)
    excluded_txids = common_genes.intersection(pd.Index(polya_lacking_txids))
    allowed_genes = common_genes.difference(excluded_txids)

    if len(allowed_genes) == 0:
        raise RuntimeError("No transcripts remained after excluding polyA-lacking transcripts.")

    if len(excluded_txids) > 0:
        print(
            f"    polyA filter: excluded={len(excluded_txids)} of {len(common_genes)} common transcripts "
            f"(logratio removed; residual collapsed into 'others')"
        )

    rna_aligned = rna_aligned[common_genes].astype(np.float32)
    rpf_aligned = rpf_aligned[common_genes].astype(np.float32)

    logratio_te = calculate_logratio_te(
        rpf_aligned.loc[:, allowed_genes],
        rna_aligned.loc[:, allowed_genes],
    )
    residual_te = compositional_te_with_nonpolya_others(
        rpf_aligned,
        rna_aligned,
        excluded_txids,
    )

    metric_cols = (
        logratio_te.columns
        .union(residual_te.columns)
        .intersection(rna_aligned.columns)
        .intersection(rpf_aligned.columns)
    )

    if metric_cols.empty:
        raise RuntimeError("No genes remained after TE metric calculation.")

    logratio_te = logratio_te.reindex(columns=metric_cols)
    residual_te = residual_te.reindex(columns=metric_cols)
    rna_aligned = rna_aligned[metric_cols]
    rpf_aligned = rpf_aligned[metric_cols]

    both_te_all_nan = logratio_te.isna().all(axis=0) & residual_te.isna().all(axis=0)
    metric_cols = metric_cols[~both_te_all_nan]
    if len(metric_cols) == 0:
        raise RuntimeError("No genes remained after removing genes with both TE metrics fully missing.")

    logratio_te = logratio_te[metric_cols]
    residual_te = residual_te[metric_cols]
    rna_aligned = rna_aligned[metric_cols]
    rpf_aligned = rpf_aligned[metric_cols]

    if logratio_te.empty:
        raise RuntimeError("Log-ratio TE matrix is empty after filtering.")

    te_concat = pd.concat(
        [
            paired.set_index("pairIndex")[["geoStudy", "cellType", "sampleType", "rnaSample", "rpfSample"]],
            logratio_te,
        ],
        axis=1,
    )

    sample_null_prop = logratio_te.isnull().mean(axis=1)
    sample_status = te_concat[["geoStudy", "cellType", "sampleType", "rnaSample", "rpfSample"]].copy()
    sample_status["logratio_te_max"] = logratio_te.max(axis=1, skipna=True).round(6)
    sample_status["logratio_te_min"] = logratio_te.min(axis=1, skipna=True).round(6)
    sample_status["logratio_te_mean"] = logratio_te.mean(axis=1, skipna=True).round(6)
    sample_status["residual_te_max"] = residual_te.max(axis=1, skipna=True).round(6)
    sample_status["residual_te_min"] = residual_te.min(axis=1, skipna=True).round(6)
    sample_status["residual_te_mean"] = residual_te.mean(axis=1, skipna=True).round(6)
    sample_status["sample_na_percent"] = sample_null_prop.round(4)

    pearson_vals = []
    spearman_vals = []
    for idx in logratio_te.index:
        pearson, spearman = corr_pair(rna_aligned.loc[idx], rpf_aligned.loc[idx])
        pearson_vals.append(pearson)
        spearman_vals.append(spearman)
    sample_status["rna_rpf_pearson"] = np.round(pearson_vals, 6)
    sample_status["rna_rpf_spearman"] = np.round(spearman_vals, 6)
    sample_status.to_csv("te_samplewise_status_report.tsv", sep="\t", index=True)

    keep_mask = sample_null_prop <= sparsity_cutoff
    dropped = int((~keep_mask).sum())

    paired_kept = paired.set_index("pairIndex").loc[keep_mask]
    logratio_te = logratio_te.loc[keep_mask]
    residual_te = residual_te.loc[keep_mask]
    rna_aligned = rna_aligned.loc[keep_mask]
    rpf_aligned = rpf_aligned.loc[keep_mask]

    print(
        f"    TE matrix: pairs={len(keep_mask)}, kept={int(keep_mask.sum())}, dropped={dropped}, "
        f"genes={logratio_te.shape[1]}"
    )

    if logratio_te.empty:
        raise RuntimeError("All TE sample pairs were removed by the sparsity cutoff.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = "te_celltypewise_status_report.tsv"

    relevant_metadata = gene_metadata.loc[metric_cols, ["txID", "utr5", "cds", "utr3"]].copy()
    relevant_metadata["full_seq"] = (
        relevant_metadata["utr5"] + relevant_metadata["cds"] + relevant_metadata["utr3"]
    )

    skipped = []
    with open(report_path, "w") as report:
        print(
            "groupKey",
            "safeGroupKey",
            "sampleSize",
            "txSize",
            "minLogratioTE",
            "maxLogratioTE",
            "meanLogratioTE",
            "sdLogratioTE",
            "minResidualTE",
            "maxResidualTE",
            "meanResidualTE",
            "sdResidualTE",
            sep="\t",
            file=report,
        )

        grouped = paired_kept.groupby("cellType", sort=True)
        for cell_type, group_df in tqdm(grouped, desc="Writing TE_by_celltype", unit="group"):
            pair_ids = group_df.index.astype(str)

            logratio_mean = logratio_te.loc[pair_ids].mean(axis=0, skipna=True)
            residual_mean = residual_te.loc[pair_ids].mean(axis=0, skipna=True)
            rna_mean = rna_aligned.loc[pair_ids].mean(axis=0, skipna=True)
            rpf_mean = rpf_aligned.loc[pair_ids].mean(axis=0, skipna=True)

            keep_out_mask = ~(logratio_mean.isna() & residual_mean.isna())
            common_out_cols = logratio_mean.index[keep_out_mask]

            if len(common_out_cols) == 0:
                skipped.append(f"Skipped {cell_type}: no valid output values")
                print(
                    cell_type, "na", len(group_df), 0,
                    "NaN", "NaN", "NaN", "NaN",
                    "NaN", "NaN", "NaN", "NaN",
                    sep="\t", file=report
                )
                continue

            out = relevant_metadata.loc[common_out_cols].copy()
            out["logratio_te"] = logratio_mean.loc[common_out_cols].astype(np.float32)
            out["residual_te"] = residual_mean.loc[common_out_cols].astype(np.float32)
            out["rna"] = rna_mean.loc[common_out_cols].astype(np.float32)
            out["rpf"] = rpf_mean.loc[common_out_cols].astype(np.float32)
            out = out[["txID", "utr5", "cds", "utr3", "full_seq", "logratio_te", "residual_te", "rna", "rpf"]]

            safe_key = slugify_local(cell_type)
            out_path = output_dir / f"{safe_key}_TE.tsv"
            out.to_csv(out_path, sep="\t", index=False, na_rep="NaN")

            logratio_vals = out["logratio_te"]
            residual_vals = out["residual_te"]
            print(
                cell_type,
                safe_key,
                len(group_df),
                len(out),
                round(float(logratio_vals.min(skipna=True)), 8),
                round(float(logratio_vals.max(skipna=True)), 8),
                round(float(logratio_vals.mean(skipna=True)), 8),
                round(float(logratio_vals.std(skipna=True)), 8) if len(logratio_vals) > 1 else "NaN",
                round(float(residual_vals.min(skipna=True)), 8),
                round(float(residual_vals.max(skipna=True)), 8),
                round(float(residual_vals.mean(skipna=True)), 8),
                round(float(residual_vals.std(skipna=True)), 8) if len(residual_vals) > 1 else "NaN",
                sep="\t",
                file=report,
            )

    print("Step summary")
    print(f"    merged RNA matrix   : {MERGED_RNA_PATH}")
    print(f"    merged RPF matrix   : {MERGED_RPF_PATH}")
    print(f"    pairwise report     : {pair_report_path}")
    print(f"    samplewise report   : te_samplewise_status_report.tsv")
    print(f"    celltype report     : {report_path}")
    print(f"    TE output directory : {output_dir}")
    if skipped:
        print(f"    skipped groups      : {len(skipped)}")
        for msg in skipped[:10]:
            print(f"      - {msg}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Merge RNA/RPF count tables, filter genes, calculate paired translation efficiency metrics, "
            "write pairwise QC report, apply pairwise filters, and export cell type-wise TE outputs. "
            "Merged parquet files are always built from the full sample set; -t/--type is applied dynamically after loading merged tables."
        )
    )
    p.add_argument("metadata_file", help="Tab-delimited sample metadata file")
    p.add_argument(
        "--data-dir",
        default="data",
        help="Root directory containing <GSE>/<SRR>/<SRR>.<suffix> files (default: %(default)s)",
    )
    p.add_argument(
        "--output-dir",
        default="../celltype_te",
        help="Output directory to write TE files. (default: %(default)s)",
    )
    p.add_argument(
        "--gene-metadata",
        default=DEFAULT_GENE_METADATA,
        help=f"Transcript metadata table for TE output generation (default: {DEFAULT_GENE_METADATA})",
    )
    p.add_argument(
        "--polyA-lacking-ref",
        default=DEFAULT_POLYA_LACKING_TXIDS,
        help=(
            "Tab-delimited reference of polyA-lacking transcripts with a txID column. "
            f"These transcripts are removed from final outputs; for residual TE they are collapsed into 'others' during CLR composition (default: {DEFAULT_POLYA_LACKING_TXIDS})"
        ),
    )
    p.add_argument(
        "--suffix",
        default="cpm.tsv",
        help="Input file suffix appended after '<SRR>.' (default: %(default)s)",
    )
    p.add_argument(
        "--column",
        type=int,
        default=2,
        help="Column index(0-based) to read from each input table (default: %(default)s)",
    )
    p.add_argument(
        "-t",
        "--type",
        default="all",
        choices=["control", "treatment", "all"],
        help=(
            "Sample group to analyze after loading merged parquet. "
            "Merge output itself is always full-sample (default: %(default)s)"
        ),
    )
    p.add_argument(
        "-r",
        "--rpf-cutoff",
        type=float,
        default=1.0,
        help="RPF cutoff applied before TE calculation (default: %(default)f)",
    )
    p.add_argument(
        "-R",
        "--rna-cutoff",
        type=float,
        default=1.0,
        help="RNA cutoff applied before TE calculation (default: %(default)f)",
    )
    p.add_argument(
        "-s",
        "--sparsity-cutoff",
        type=float,
        default=0.8,
        help="Maximum allowed per-sample log-ratio TE NaN ratio (default: %(default)f)",
    )
    p.add_argument(
        "--pair-metric-pseudocount",
        type=float,
        default=0.01,
        help="Pseudocount used for pairwise QC metrics such as max logratio TE and pair Spearman (default: %(default)f)",
    )
    p.add_argument(
        "--pair-filter-spearman-hard",
        type=float,
        default=0.60,
        help="Drop pair if pair_spearman is below this value (default: %(default)f)",
    )
    p.add_argument(
        "--pair-filter-spearman-soft",
        type=float,
        default=0.70,
        help="Apply additional RPF top5 filter when pair_spearman is below this value (default: %(default)f)",
    )
    p.add_argument(
        "--pair-filter-rpf-top5",
        type=float,
        default=0.10,
        help="Drop pair when pair_spearman is below --pair-filter-spearman-soft and rpf_top5_fraction exceeds this value (default: %(default)f)",
    )
    p.add_argument(
        "--no-merge",
        action="store_true",
        help=(
            "Skip input-table merge and reuse existing merged parquet files at "
            f"{MERGED_RNA_PATH} and {MERGED_RPF_PATH}. "
            "These merged parquet files must have been built from the full sample set."
        ),
    )
    return p

def main():
    args = build_argparser().parse_args()
    selected_group = normalize_condition(args.type)

    if args.no_merge:
        print("[1/4] --no-merge enabled: reusing existing full-sample merged parquet files")
        if not os.path.exists(MERGED_RNA_PATH):
            raise FileNotFoundError(f"Missing merged RNA file: {MERGED_RNA_PATH}")
        if not os.path.exists(MERGED_RPF_PATH):
            raise FileNotFoundError(f"Missing merged RPF file: {MERGED_RPF_PATH}")

        print(f"[2/4] Loading merged RNA table: {MERGED_RNA_PATH}")
        rna_df = read_merged_table(MERGED_RNA_PATH)
        print(f"[2/4] Loading merged RPF table: {MERGED_RPF_PATH}")
        rpf_df = read_merged_table(MERGED_RPF_PATH)

    else:
        groups = read_sample_metadata(args.metadata_file)

        rna_df = load_expression_table(
            groups["RNA"],
            data_dir=args.data_dir,
            suffix=args.suffix,
            value_column=args.column,
            output_path=MERGED_RNA_PATH,
            label="RNA",
        )
        rpf_df = load_expression_table(
            groups["RPF"],
            data_dir=args.data_dir,
            suffix=args.suffix,
            value_column=args.column,
            output_path=MERGED_RPF_PATH,
            label="RPF",
        )

    rna_df = filter_merged_table_by_group(rna_df, selected_group, "RNA")
    rpf_df = filter_merged_table_by_group(rpf_df, selected_group, "RPF")

    gene_metadata = load_gene_metadata(args.gene_metadata)
    polya_lacking_txids = load_polya_lacking_txids(args.polyA_lacking_ref)

    calculate_te_and_reports(
        rna_df=rna_df,
        rpf_df=rpf_df,
        gene_metadata=gene_metadata,
        polya_lacking_txids=polya_lacking_txids,
        rna_cutoff=args.rna_cutoff,
        rpf_cutoff=args.rpf_cutoff,
        sparsity_cutoff=args.sparsity_cutoff,
        pair_metric_pseudocount=args.pair_metric_pseudocount,
        spearman_hard_cutoff=args.pair_filter_spearman_hard,
        spearman_soft_cutoff=args.pair_filter_spearman_soft,
        rpf_top5_soft_cutoff=args.pair_filter_rpf_top5,
        output_dir=args.output_dir,
    )

if __name__ == "__main__":
    main()
