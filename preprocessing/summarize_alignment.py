#!/usr/bin/env python3
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD


import argparse
import csv
import re
import sys
import os
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
        description="Update sample status report using bowtie2 alignment logs."
    )
    parser.add_argument(
        "metadata",
        nargs="?",
        default="./2.filtered_sample_to_alignment",
        help="Input metadata file (default: ./2.filtered_sample_to_alignment)",
    )
    parser.add_argument(
        "status_report",
        nargs="?",
        default="./1.fastq_stat_report.tsv",
        help="Existing status report (default: ./1.fastq_stat_report.tsv)",
    )
    parser.add_argument(
        "cutoff",
        nargs="?",
        type=float,
        default=0.1,
        help="Alignment-rate cutoff in [0.0, 1.0] (default: 0.1)",
    )
    parser.add_argument(
        "--mapped-count-cutoff",
        type=int,
        default=3000000,
        help="Mapped read/read-pair cutoff (default: 3000000)",
    )
    return parser.parse_args()


def validate_args(cutoff, mapped_count_cutoff):
    if not (0.0 <= cutoff <= 1.0):
        raise ValueError(f"cutoff must be between 0.0 and 1.0: {cutoff}")
    if mapped_count_cutoff < 0:
        raise ValueError(f"mapped-count-cutoff must be >= 0: {mapped_count_cutoff}")


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


FASTQ_SINGLE_RE = re.compile(r"^data/([^/]+)/([^/]+)/((?!.*_(?:1|2)\.filtered\.fastq$)[^/]+)\.filtered\.fastq$")
FASTQ_PAIRED_RE = re.compile(r"^data/([^/]+)/([^/]+)/([^/]+)_(1|2)\.filtered\.fastq$")

TOTAL_READS_RE = re.compile(r"^\s*(\d+)\s+reads;\s+of\s+these:\s*$")
UNPAIRED_HEADER_RE = re.compile(r"^\s*(\d+)\s+\([^)]+\)\s+were unpaired;\s+of\s+these:\s*$")
UNPAIRED_ZERO_RE = re.compile(r"^\s*(\d+)\s+\([^)]+\)\s+aligned 0 times\s*$")
UNPAIRED_EXACT_RE = re.compile(r"^\s*(\d+)\s+\([^)]+\)\s+aligned exactly 1 time\s*$")
UNPAIRED_MULTI_RE = re.compile(r"^\s*(\d+)\s+\([^)]+\)\s+aligned >1 times\s*$")

PAIRED_HEADER_RE = re.compile(r"^\s*(\d+)\s+\([^)]+\)\s+were paired;\s+of\s+these:\s*$")
PAIRED_CONC_ZERO_RE = re.compile(r"^\s*(\d+)\s+\([^)]+\)\s+aligned concordantly 0 times\s*$")
PAIRED_CONC_EXACT_RE = re.compile(r"^\s*(\d+)\s+\([^)]+\)\s+aligned concordantly exactly 1 time\s*$")
PAIRED_CONC_MULTI_RE = re.compile(r"^\s*(\d+)\s+\([^)]+\)\s+aligned concordantly >1 times\s*$")
PAIRED_DISC_RE = re.compile(r"^\s*(\d+)\s+\([^)]+\)\s+aligned discordantly 1 time\s*$")
PAIRED_ZERO_FINAL_RE = re.compile(r"^\s*(\d+)\s+pairs aligned 0 times concordantly or discordantly;\s+of\s+these:\s*$")

ALIGNMENT_RATE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)%\s+overall alignment rate\s*$")


def nonempty_isfile(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


def normalize_sample_key(path):
    path = path.strip()

    match_paired = FASTQ_PAIRED_RE.match(path)
    if match_paired:
        gse, sample_dir, sample_base, mate = match_paired.groups()
        if sample_dir != sample_base:
            raise ValueError(f"Malformed filtered fastq path: {path}")
        return (gse, sample_base)

    match_single = FASTQ_SINGLE_RE.match(path)
    if match_single:
        gse, sample_dir, sample_base = match_single.groups()
        if sample_dir != sample_base:
            raise ValueError(f"Malformed filtered fastq path: {path}")
        return (gse, sample_base)

    raise ValueError(f"Unexpected filtered fastq path format: {path}")


def parse_files_field(files, library_type):
    files = files.strip()
    library_type = library_type.strip().lower()

    if library_type == "single":
        members = [x.strip() for x in files.split(",") if x.strip()]
        if not members:
            raise ValueError(f"Empty files field for single-end sample: {files}")
        for member in members:
            normalize_sample_key(member)
        return members

    if library_type == "paired":
        try:
            mates1, mates2 = files.split(";")
        except ValueError as exc:
            raise ValueError(f"Paired files field must contain ';': {files}") from exc

        files1 = [x.strip() for x in mates1.split(",") if x.strip()]
        files2 = [x.strip() for x in mates2.split(",") if x.strip()]
        if len(files1) != len(files2) or not files1:
            raise ValueError(f"Unbalanced paired files field: {files}")

        members = []
        for f1, f2 in zip(files1, files2):
            key1 = normalize_sample_key(f1)
            key2 = normalize_sample_key(f2)
            if key1 != key2:
                raise ValueError(f"Paired files do not match: {f1} <-> {f2}")
            members.append((f1, f2))
        return members

    raise ValueError(f"Unsupported libraryType: {library_type}")


def get_representative_sample(row):
    gse = row["gse"].strip()
    srr = row["srr"].strip()
    return gse, srr


def get_log_path(sample_key):
    gse, sample = sample_key
    return Path(f"data/{gse}/{sample}/{sample}.bowtie2.log")


def parse_bowtie2_log(log_path):
    if not nonempty_isfile(log_path):
        return {"alignment_rate": None, "mapped_count": None, "library_kind": None}

    total_reads = None
    alignment_rate = None
    library_kind = None

    unpaired_zero = None
    unpaired_exact = None
    unpaired_multi = None

    paired_total = None
    paired_conc_zero = None
    paired_conc_exact = None
    paired_conc_multi = None
    paired_disc = 0
    paired_zero_final = None

    with open(log_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()

            match = TOTAL_READS_RE.match(line)
            if match:
                total_reads = int(match.group(1))
                continue

            match = ALIGNMENT_RATE_RE.match(line)
            if match:
                alignment_rate = round(float(match.group(1)) / 100.0, 4)
                continue

            if UNPAIRED_HEADER_RE.match(line):
                library_kind = "single"
                continue

            if PAIRED_HEADER_RE.match(line):
                library_kind = "paired"
                paired_total = int(PAIRED_HEADER_RE.match(line).group(1))
                continue

            match = UNPAIRED_ZERO_RE.match(line)
            if match:
                unpaired_zero = int(match.group(1))
                continue

            match = UNPAIRED_EXACT_RE.match(line)
            if match and library_kind == "single":
                unpaired_exact = int(match.group(1))
                continue

            match = UNPAIRED_MULTI_RE.match(line)
            if match and library_kind == "single":
                unpaired_multi = int(match.group(1))
                continue

            match = PAIRED_CONC_ZERO_RE.match(line)
            if match:
                paired_conc_zero = int(match.group(1))
                continue

            match = PAIRED_CONC_EXACT_RE.match(line)
            if match:
                paired_conc_exact = int(match.group(1))
                continue

            match = PAIRED_CONC_MULTI_RE.match(line)
            if match:
                paired_conc_multi = int(match.group(1))
                continue

            match = PAIRED_DISC_RE.match(line)
            if match:
                paired_disc = int(match.group(1))
                continue

            match = PAIRED_ZERO_FINAL_RE.match(line)
            if match:
                paired_zero_final = int(match.group(1))
                continue

    if alignment_rate is None:
        raise RuntimeError(f"overall alignment rate not found: {log_path}")

    if library_kind == "single":
        values = [unpaired_zero, unpaired_exact, unpaired_multi]
        if any(v is None for v in values):
            raise RuntimeError(f"Incomplete unpaired bowtie2 summary: {log_path}")
        mapped_count = unpaired_exact + unpaired_multi
        if total_reads is not None and total_reads != sum(values):
            raise RuntimeError(f"Inconsistent unpaired bowtie2 summary: {log_path}")
        return {
            "alignment_rate": alignment_rate,
            "mapped_count": mapped_count,
            "library_kind": library_kind,
        }

    if library_kind == "paired":
        required = [paired_total, paired_conc_zero, paired_conc_exact, paired_conc_multi]
        if any(v is None for v in required):
            raise RuntimeError(f"Incomplete paired bowtie2 summary: {log_path}")

        if paired_zero_final is not None:
            mapped_count = paired_total - paired_zero_final
        else:
            mapped_count = paired_conc_exact + paired_conc_multi + paired_disc

        return {
            "alignment_rate": alignment_rate,
            "mapped_count": mapped_count,
            "library_kind": library_kind,
        }

    raise RuntimeError(f"Unable to determine library type from bowtie2 summary: {log_path}")


def build_metadata_maps(metadata_rows):
    uniqueid_to_row = {}
    uniqueid_to_members = {}
    member_to_uniqueid = {}
    index_to_uniqueids = defaultdict(list)

    total = len(metadata_rows)
    progress("Parsing metadata entries", 0, total)
    for i, row in enumerate(metadata_rows, 1):
        required = ["gse", "srx", "srr", "gsm", "files", "libraryType", "condition", "sampletype", "celltype", "index"]
        missing = [col for col in required if col not in row]
        if missing:
            raise KeyError(f"Missing required metadata columns: {', '.join(missing)}")

        uniqueid = f'{row["index"]}_{row["sampletype"]}'
        if uniqueid in uniqueid_to_row:
            raise RuntimeError(f"Duplicated unique biological entry in metadata: {uniqueid}")

        members = parse_files_field(row["files"], row["libraryType"])
        member_keys = []
        if row["libraryType"].strip().lower() == "single":
            for path in members:
                member_keys.append(normalize_sample_key(path))
        else:
            for path1, path2 in members:
                member_keys.append(normalize_sample_key(path1))

        uniqueid_to_row[uniqueid] = row
        uniqueid_to_members[uniqueid] = member_keys
        index_to_uniqueids[row["index"]].append(uniqueid)

        for member_key in member_keys:
            if member_key in member_to_uniqueid:
                raise RuntimeError(f"Sample mapped to multiple metadata entries: {member_key}")
            member_to_uniqueid[member_key] = uniqueid

        progress("Parsing metadata entries", i, total)

    return uniqueid_to_row, uniqueid_to_members, member_to_uniqueid, index_to_uniqueids


def evaluate_entries(uniqueid_to_row):
    alignment_rate = {}
    mapped_count = {}

    total = len(uniqueid_to_row)
    progress("Reading bowtie2 logs", 0, total)
    for i, (uniqueid, row) in enumerate(uniqueid_to_row.items(), 1):
        representative = get_representative_sample(row)
        log_path = get_log_path(representative)
        stats = parse_bowtie2_log(log_path)
        alignment_rate[uniqueid] = stats["alignment_rate"]
        mapped_count[uniqueid] = stats["mapped_count"]
        progress("Reading bowtie2 logs", i, total)

    return alignment_rate, mapped_count


def decide_keep_discard(uniqueid_to_row, alignment_rate, mapped_count, index_to_uniqueids, cutoff, mapped_count_cutoff):
    discard_reason = {}

    for uniqueid in uniqueid_to_row:
        rate = alignment_rate.get(uniqueid)
        count = mapped_count.get(uniqueid)

        if rate is None or count is None:
            discard_reason[uniqueid] = "poor_alignment"
            continue

        if rate < cutoff and count < mapped_count_cutoff:
            discard_reason[uniqueid] = "poor_alignment"

    for index, uniqueids in index_to_uniqueids.items():
        if any(uid in discard_reason for uid in uniqueids):
            for uid in uniqueids:
                discard_reason.setdefault(uid, "pair_discarded")

    return discard_reason


def format_rate(value):
    if value is None:
        return "na"
    return f"{value:.4f}"


def format_count(value):
    if value is None:
        return "na"
    return str(value)


def update_status_report(status_rows, alignment_rate, mapped_count, discard_reason, member_to_uniqueid, uniqueid_to_members):
    updated_rows = []

    total = len(status_rows)
    progress("Updating status report", 0, total)
    for i, row in enumerate(status_rows, 1):
        row = dict(row)
        if "sraAcc" not in row or "geoStudy" not in row:
            raise KeyError("status report must contain geoStudy and sraAcc columns")

        original_status = row.get("status", "").strip().lower()
        preserve_reason = original_status == "discard"

        member_key = (row["geoStudy"].strip(), row["sraAcc"].strip())
        uniqueid = member_to_uniqueid.get(member_key)

        if uniqueid is None:
            row["alignment.rate"] = "na"
            row["mapped.count"] = "na"
            row["status"] = "discard"
            if not preserve_reason:
                row["reason"] = "na"
        else:
            row["alignment.rate"] = format_rate(alignment_rate.get(uniqueid))
            row["mapped.count"] = format_count(mapped_count.get(uniqueid))

            members = uniqueid_to_members[uniqueid]
            representative = members[0]

            if member_key != representative:
                row["status"] = "keep"
                if not preserve_reason:
                    row["reason"] = "merged"
            elif uniqueid in discard_reason:
                row["status"] = "discard"
                if not preserve_reason:
                    row["reason"] = discard_reason[uniqueid]
            else:
                row["status"] = "keep"
                if not preserve_reason:
                    row["reason"] = "included"

        updated_rows.append(row)
        progress("Updating status report", i, total)

    return updated_rows


def make_status_header(status_header):
    header = list(status_header)
    for col in ["alignment.rate", "mapped.count"]:
        if col in header:
            header.remove(col)

    if "status" not in header:
        raise KeyError("status report must contain status column")

    status_idx = header.index("status")
    header.insert(status_idx, "alignment.rate")
    header.insert(status_idx + 1, "mapped.count")
    return header


def build_output_metadata(uniqueid_to_row, discard_reason):
    header = ["gse", "srx", "srr", "gsm", "condition", "sampletype", "celltype", "index"]
    rows = []

    ordered_uniqueids = list(uniqueid_to_row.keys())
    total = len(ordered_uniqueids)
    progress("Writing filtered metadata entries", 0, total)
    for i, uniqueid in enumerate(ordered_uniqueids, 1):
        if uniqueid in discard_reason:
            progress("Writing filtered metadata entries", i, total)
            continue

        row = uniqueid_to_row[uniqueid]
        rows.append({key: row[key] for key in header})
        progress("Writing filtered metadata entries", i, total)

    return header, rows


def main():
    args = parse_args()
    validate_args(args.cutoff, args.mapped_count_cutoff)

    metadata_header, metadata_rows = read_tsv(args.metadata)
    status_header, status_rows = read_tsv(args.status_report)

    uniqueid_to_row, uniqueid_to_members, member_to_uniqueid, index_to_uniqueids = build_metadata_maps(metadata_rows)
    alignment_rate, mapped_count = evaluate_entries(uniqueid_to_row)
    discard_reason = decide_keep_discard(
        uniqueid_to_row=uniqueid_to_row,
        alignment_rate=alignment_rate,
        mapped_count=mapped_count,
        index_to_uniqueids=index_to_uniqueids,
        cutoff=args.cutoff,
        mapped_count_cutoff=args.mapped_count_cutoff,
    )

    output_metadata_header, output_metadata_rows = build_output_metadata(uniqueid_to_row, discard_reason)
    updated_status_rows = update_status_report(
        status_rows=status_rows,
        alignment_rate=alignment_rate,
        mapped_count=mapped_count,
        discard_reason=discard_reason,
        member_to_uniqueid=member_to_uniqueid,
        uniqueid_to_members=uniqueid_to_members,
    )
    updated_status_header = make_status_header(status_header)

    metadata_out = "3.filtered_sample_after_alignment"
    report_out = "2.alignment_stat_report.tsv"

    progress(f"Writing {metadata_out}")
    write_tsv(metadata_out, output_metadata_header, output_metadata_rows)
    os.system(f"head -n1 {metadata_out} > {metadata_out}_for_qc; grep 'rpf' {metadata_out} >> {metadata_out}_for_qc")

    progress(f"Writing {report_out}")
    write_tsv(report_out, updated_status_header, updated_status_rows)

    n_total = len(uniqueid_to_row)
    n_discard = len(discard_reason)
    n_keep = n_total - n_discard
    progress(
        f"Done. cutoff={args.cutoff:.4f}, mapped_count_cutoff={args.mapped_count_cutoff}, "
        f"kept={n_keep}, discarded={n_discard}, metadata={metadata_out}, report={report_out}"
    )


if __name__ == "__main__":
    main()

