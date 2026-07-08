#!/usr/bin/env python3
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD


import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def progress(message, current=None, total=None, width=30):
    if current is None or total is None or total == 0:
        print(f"[INFO] {message}", file=sys.stderr)
        return

    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[INFO] {message}: [{bar}] {current}/{total}", end="", file=sys.stderr)
    if current == total:
        print("", file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Update sample status report using ribo-seq QC reports."
    )
    parser.add_argument(
        "metadata",
        nargs="?",
        default="./3.filtered_sample_after_alignment",
        help="Input metadata file (default: ./3.filtered_sample_after_alignment)",
    )
    parser.add_argument(
        "status_report",
        nargs="?",
        default="./2.alignment_stat_report.tsv",
        help="Existing status report (default: ./2.alignment_stat_report.tsv)",
    )
    parser.add_argument(
        "suffix",
        nargs="?",
        default="qc.1.tsv",
        help="Suffix of QC reports (default: qc.1.tsv)",
    )
    return parser.parse_args()


def read_tsv(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [dict(row) for row in reader]
        return reader.fieldnames, rows


def write_tsv(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def build_metadata_maps(metadata_rows):
    uniqueid_to_row = {}
    member_to_uniqueid = {}
    index_to_uniqueids = defaultdict(list)

    total = len(metadata_rows)
    progress("Parsing metadata entries", 0, total)
    for i, row in enumerate(metadata_rows, 1):
        required = ["gse", "srx", "srr", "gsm", "condition", "sampletype", "celltype", "index"]
        missing = [col for col in required if col not in row]
        if missing:
            raise KeyError(f"Missing required metadata columns: {', '.join(missing)}")

        uniqueid = f'{row["index"]}_{row["sampletype"]}'
        if uniqueid in uniqueid_to_row:
            raise RuntimeError(f"Duplicated unique biological entry in metadata: {uniqueid}")

        member_key = (row["gse"].strip(), row["srr"].strip())
        if member_key in member_to_uniqueid:
            raise RuntimeError(f"Sample mapped to multiple metadata entries: {member_key}")

        uniqueid_to_row[uniqueid] = row
        member_to_uniqueid[member_key] = uniqueid
        index_to_uniqueids[row["index"]].append(uniqueid)

        progress("Parsing metadata entries", i, total)

    return uniqueid_to_row, member_to_uniqueid, index_to_uniqueids


def is_rpf_sampletype(sampletype):
    value = sampletype.strip().lower()
    return value == "rpf" or "rpf" in value


def is_rna_sampletype(sampletype):
    value = sampletype.strip().lower()
    return value == "rna" or "rna" in value


def parse_qc_report(path):
    data = {}

    with open(path, newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            value = row[1].strip() if len(row) > 1 else ""
            if key:
                data[key] = value

    return data


def read_qc_reports():
    reports = {}
    #qc_paths = sorted(Path("data").glob(f"*/*/*{args.suffix}"))
    qc_paths = sorted(Path("data").glob("*/*/*qc.2.tsv"))

    total = len(qc_paths)
    progress("Reading QC reports", 0, total)
    for i, path in enumerate(qc_paths, 1):
        rel = path.as_posix()
        parts = path.parts
        if len(parts) < 4:
            raise RuntimeError(f"Unexpected QC report path: {rel}")

        gse = parts[1]
        sample_dir = parts[2]
        report = parse_qc_report(path)

        sample_id = report.get("sample_id", "").strip()
        if sample_id and sample_id != sample_dir:
            raise RuntimeError(f"QC report sample_id mismatch: {rel} ({sample_id} != {sample_dir})")

        key = (gse, sample_dir)
        if key in reports:
            raise RuntimeError(f"Duplicated QC report for sample: {key}")

        reports[key] = report
        progress("Reading QC reports", i, total)

    return reports


def find_pair_rna_uniqueid(index, index_to_uniqueids, uniqueid_to_row):
    for uniqueid in index_to_uniqueids.get(index, []):
        row = uniqueid_to_row[uniqueid]
        if is_rna_sampletype(row["sampletype"]):
            return uniqueid
    return None


def normalize_pass(value):
    value = value.strip()
    if value == "":
        return None
    if value not in {"0", "1"}:
        raise RuntimeError(f"Unexpected pass value: {value}")
    return int(value)


def evaluate_entries(uniqueid_to_row, member_to_uniqueid, index_to_uniqueids, qc_reports):
    qc_by_uniqueid = {}
    discard_reason = {}
    targeted_uniqueids = set()

    total = len(qc_reports)
    progress("Evaluating QC decisions", 0, total)
    for i, (member_key, report) in enumerate(qc_reports.items(), 1):
        uniqueid = member_to_uniqueid.get(member_key)
        if uniqueid is None:
            #raise RuntimeError(f"QC report does not map to metadata entry: {member_key}")
            continue

        row = uniqueid_to_row[uniqueid]
        if not is_rpf_sampletype(row["sampletype"]):
            raise RuntimeError(f"QC report mapped to non-RPF sample: {member_key} -> {uniqueid}")

        qc_by_uniqueid[uniqueid] = report
        targeted_uniqueids.add(uniqueid)

        final_pass = normalize_pass(report.get("final_pass", ""))
        final_reason = report.get("final_reason", "").strip() or "qc_failed"

        if final_pass != 1:
            discard_reason[uniqueid] = final_reason

        pair_uniqueid = find_pair_rna_uniqueid(row["index"], index_to_uniqueids, uniqueid_to_row)
        if pair_uniqueid is not None:
            targeted_uniqueids.add(pair_uniqueid)
            if final_pass != 1:
                discard_reason[pair_uniqueid] = "pair_discarded"

        progress("Evaluating QC decisions", i, total)

    return qc_by_uniqueid, discard_reason, targeted_uniqueids


def format_float(value):
    if value is None:
        return "na"
    value = value.strip()
    if value == "":
        return "na"
    return f"{float(value):.4f}"


def format_text(value):
    if value is None:
        return "na"
    value = value.strip()
    return value if value else "na"


def update_status_report(
    status_rows,
    qc_by_uniqueid,
    discard_reason,
    targeted_uniqueids,
    member_to_uniqueid,
):
    updated_rows = []

    total = len(status_rows)
    progress("Updating status report", 0, total)
    for i, row in enumerate(status_rows, 1):
        row = dict(row)
        if "sraAcc" not in row or "geoStudy" not in row:
            raise KeyError("status report must contain geoStudy and sraAcc columns")

        original_status = row.get("status", "").strip().lower()
        preserve_existing_discard = original_status == "discard"

        member_key = (row["geoStudy"].strip(), row["sraAcc"].strip())
        uniqueid = member_to_uniqueid.get(member_key)

        row["cds.ratio"] = "na"
        row["tx.coverage"] = "na"
        row["frame0.pct"] = "na"
        row["selected.lengths"] = "na"

        if uniqueid is not None and uniqueid in qc_by_uniqueid:
            report = qc_by_uniqueid[uniqueid]
            row["cds.ratio"] = format_float(report.get("cds_ratio"))
            row["tx.coverage"] = format_float(report.get("coverage_tx"))
            row["frame0.pct"] = format_float(report.get("frame0_pct"))
            row["selected.lengths"] = format_text(report.get("selected_lengths"))

        if uniqueid is None or uniqueid not in targeted_uniqueids:
            updated_rows.append(row)
            progress("Updating status report", i, total)
            continue

        if uniqueid in discard_reason:
            if not preserve_existing_discard:
                row["status"] = "discard"
                row["reason"] = discard_reason[uniqueid]
        else:
            if not preserve_existing_discard:
                row["status"] = "keep"
                row["reason"] = "included"

        updated_rows.append(row)
        progress("Updating status report", i, total)

    return updated_rows


def make_status_header(status_header):
    header = list(status_header)
    for col in ["cds.ratio", "tx.coverage", "frame0.pct", "selected.lengths"]:
        if col in header:
            header.remove(col)

    if "status" not in header:
        raise KeyError("status report must contain status column")

    status_idx = header.index("status")
    header.insert(status_idx, "cds.ratio")
    header.insert(status_idx + 1, "tx.coverage")
    header.insert(status_idx + 2, "frame0.pct")
    header.insert(status_idx + 3, "selected.lengths")
    return header


def build_output_metadata(uniqueid_to_row, qc_by_uniqueid, discard_reason):
    header = ["gse", "srx", "srr", "gsm", "length", "condition", "sampletype", "celltype", "index"]
    rows = []

    ordered_uniqueids = list(uniqueid_to_row.keys())
    total = len(ordered_uniqueids)
    progress("Writing filtered metadata entries", 0, total)
    for i, uniqueid in enumerate(ordered_uniqueids, 1):
        if uniqueid in discard_reason:
            progress("Writing filtered metadata entries", i, total)
            continue

        row = dict(uniqueid_to_row[uniqueid])
        selected_lengths = "na"
        if uniqueid in qc_by_uniqueid:
            selected_lengths = format_text(qc_by_uniqueid[uniqueid].get("selected_lengths"))
        row["length"] = "-" if selected_lengths == "na" else selected_lengths

        rows.append({key: row.get(key, "") for key in header})
        progress("Writing filtered metadata entries", i, total)

    return header, rows


def main():
    args = parse_args()

    metadata_header, metadata_rows = read_tsv(args.metadata)
    status_header, status_rows = read_tsv(args.status_report)

    uniqueid_to_row, member_to_uniqueid, index_to_uniqueids = build_metadata_maps(metadata_rows)
    qc_reports = read_qc_reports()
    qc_by_uniqueid, discard_reason, targeted_uniqueids = evaluate_entries(
        uniqueid_to_row=uniqueid_to_row,
        member_to_uniqueid=member_to_uniqueid,
        index_to_uniqueids=index_to_uniqueids,
        qc_reports=qc_reports,
    )

    output_metadata_header, output_metadata_rows = build_output_metadata(uniqueid_to_row, qc_by_uniqueid, discard_reason)
    updated_status_rows = update_status_report(
        status_rows=status_rows,
        qc_by_uniqueid=qc_by_uniqueid,
        discard_reason=discard_reason,
        targeted_uniqueids=targeted_uniqueids,
        member_to_uniqueid=member_to_uniqueid,
    )
    updated_status_header = make_status_header(status_header)

    metadata_out = "4.filtered_sample_to_quantify"
    report_out = "3.riboseq_qc_stat_report.tsv"

    progress(f"Writing {metadata_out}")
    write_tsv(metadata_out, output_metadata_header, output_metadata_rows)

    progress(f"Writing {report_out}")
    write_tsv(report_out, updated_status_header, updated_status_rows)

    n_total = len(uniqueid_to_row)
    n_discard = len(discard_reason)
    n_keep = n_total - n_discard
    progress(
        f"Done. kept={n_keep}, discarded={n_discard}, "
        f"metadata={metadata_out}, report={report_out}, qc_reports={len(qc_reports)}"
    )


if __name__ == "__main__":
    main()



