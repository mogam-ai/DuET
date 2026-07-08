#!/usr/bin/env python3
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD


import sys
import os
from pathlib import Path

def exists(file_path: str | Path, suffix_list: list[str]=[""]) -> bool:

    results = []
    for suffix in suffix_list:
        path = Path(file_path + suffix)
        results.append(path.is_file() and path.stat().st_size > 0)

    return any(results)

def cast(x):

    return int(x) if x.is_integer() else round(x, 4)

def isnan(x):
    return not (float(x) <= 0 or float(x) > 0)

def nonempty_isfile(fname):

    return (Path(fname).exists() or os.path.isfile(fname)) and Path(fname).stat().st_size > 0

def progress(message, current=None, total=None, width=30):

    if current is None or total is None or total == 0:
        print(f"[INFO] {message}", file=sys.stderr)
        return None

    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[INFO] {message}: [{bar}] {current}/{total}", end="", file=sys.stderr)
    if current == total:
        print("", file=sys.stderr)

    return None

def get_fastq_status(line):

    gse, srx, srr, gsm, condition, sampletype, celltype, index = line.strip().split("\t")
    sample = f"data/{gse}/{srr}/{srr}"

    values = {"fastq":{}, "trimmed":{}, "filtered":{}}
    library = "paired" if Path(sample + "_1.fastq").exists() else "single"
    with open(sample + ".read_stat.tsv") as fqstat:
        lines = set()
        for line in fqstat:
            if line in lines:
                continue
            else:
                lines.add(line)
            fname, *stats = line.strip().split("\t")
            count, lenavg, lensd = list(map(float, stats))
            if isnan(lenavg): # 0 -nan -nan (no reads survived)
                count, lenavg, lensd = 1, 1, 1
            values[fname.split(".")[1]].setdefault("count", []).append(count)
            values[fname.split(".")[1]].setdefault("lenavg", []).append(lenavg)
            values[fname.split(".")[1]].setdefault("lensd", []).append(lensd)

    to_print = [cast(sum(v)/len(v)) for k1 in values for k2,v in values[k1].items()]
    try:
        fastq_count = to_print[0]
        trim_count = to_print[3]
        filter_count = to_print[6]
    except:
        print(f"\n\nmissing line: {sample}", file=sys.stderr)
        raise AssertionError

    try:
        ratios = [
                round(trim_count / fastq_count, 4),
                round(filter_count / trim_count, 4),
                round(filter_count / fastq_count, 4),
                ]
    except ZeroDivisionError:
        print(gse, srx, srr, filter_count, trim_count, fastq_count, file=sys.stderr)
        exit(1)

    return {
            "gse": gse,
            "srx": srx,
            "srr": srr,
            "gsm": gsm,
            "condition": condition,
            "sampletype": sampletype,
            "celltype": celltype,
            "index": index,
            "library": library,
            "stats": to_print,
            "fastq_count": fastq_count,
            "trim_count": trim_count,
            "filter_count": filter_count,
            "ratios": ratios,
            }

def write_merge_outputs(index_to_srr, srr_to_line, discard_uniqueids):

    #progress("Writing 2.filtered_sample_to_process")
    #fhandle = open("2.filtered_sample_to_process", "w")
    header = ["gse", "srx", "srr", "gsm", "condition", "sampletype", "celltype", "index"]
    #print("\t".join(header), file=fhandle)
    progress("Writing 2.filtered_sample_to_alignment")
    fhandle_bowtie = open("2.filtered_sample_to_alignment", "w")
    print("\t".join(header[:4]), "files", "libraryType", "\t".join(header[4:]), sep="\t", file=fhandle_bowtie)

    total = len(index_to_srr)
    written = 0
    for uniqueid, srrs in index_to_srr.items():
        if uniqueid in discard_uniqueids:
            written += 1
            progress("Preparing merged sample outputs", written, total)
            continue

        srrs = sorted(srrs, key=lambda x:x[1])
        srr = srrs[0][0] # get representative for the srx using index
        (gse, srx, gsm), (cond, exp, cell, index) = srr_to_line[srr]

        repr_srr = f"data/{gse}/{srr}/{srr}.filtered.fastq"
        repr_srr1 = f"data/{gse}/{srr}/{srr}_1.filtered.fastq"
        repr_srr2 = f"data/{gse}/{srr}/{srr}_2.filtered.fastq"

        if nonempty_isfile(repr_srr1) and nonempty_isfile(repr_srr2): # paired end
            library = "paired"
            files_1 = [repr_srr1]
            files_2 = [repr_srr2]
            if len(srrs) != 1:
                other_srrs1 = [f"data/{gse}/{srr_}/{srr_}_1.filtered.fastq" for srr_, index in srrs[1:]]
                other_srrs2 = [f"data/{gse}/{srr_}/{srr_}_2.filtered.fastq" for srr_, index in srrs[1:]]

                validity = {k:nonempty_isfile(k) for k in other_srrs1}
                validity.update({k:nonempty_isfile(k) for k in other_srrs2})
                if not all(validity.values()):
                    print(f"File not exists: {', '.join(k for k,v in validity.items() if not v)}", file=sys.stderr)
                    exit(1)

                files_1.extend(other_srrs1)
                files_2.extend(other_srrs2)

            files = f"{','.join(files_1)};{','.join(files_2)}"

        elif nonempty_isfile(repr_srr): # single end
            library = "single"
            files = [repr_srr]
            if len(srrs) != 1:
                other_srrs = [f"data/{gse}/{srr_}/{srr_}.filtered.fastq" for srr_, index in srrs[1:]]

                validity = {k:nonempty_isfile(k) for k in other_srrs}
                if not all(validity.values()):
                    print(f"File not exists: {', '.join(k for k,v in validity.items() if not v)}", file=sys.stderr)
                    exit(1)

                files.extend(other_srrs)

            files = ",".join(files)

        else:
            raise RuntimeError(f"data/{gse}/{srr}")

        #print(gse, srx, srr, gsm, cond, exp, cell, index, sep="\t", file=fhandle)
        print(gse, srx, srr, gsm, files, library, cond, exp, cell, index, sep="\t", file=fhandle_bowtie)

        written += 1
        progress("Preparing merged sample outputs", written, total)

    #fhandle.close()
    fhandle_bowtie.close()

    return None

if len(sys.argv) not in [2, 3]:
    print(f"usage: {sys.argv[0]} [metadata.tsv] [filtered_read_count_cutoff]", file=sys.stderr)
    exit(1)

metadata_fname = sys.argv[1] if len(sys.argv) >= 2 else "./1.total_sample_to_process"
cutoff = float(sys.argv[2]) if len(sys.argv) == 3 else 1000000.0
report_fname = "1.fastq_stat_report.tsv"

progress("Loading metadata")
with open(metadata_fname) as metadata:
    lines = metadata.readlines()

header = lines[0]
metadata_lines = [line.strip() for line in lines[1:] if line.strip()]

records = []
report_by_srr = {}
index_to_srr = {} # this also merges technical repllicates (multiple srx per biological sample)
srr_to_line = {}

progress("Collecting fastq status reports", 0, len(metadata_lines))
for i, line in enumerate(metadata_lines, 1):
    record = get_fastq_status(line)
    records.append(record)
    report_by_srr[record["srr"]] = record

    uniqueid = f'{record["index"]}_{record["sampletype"]}'
    index_to_srr.setdefault(uniqueid, []).append((record["srr"], record["index"]))
    srr_to_line[record["srr"]] = [[record["gse"], record["srx"], record["gsm"]], [record["condition"], record["sampletype"], record["celltype"], record["index"]]]
    progress("Collecting fastq status reports", i, len(metadata_lines))

progress("Evaluating merged filtered read counts")
merged_filter_count = {}
index_to_uniqueids = {}
for uniqueid, srrs in index_to_srr.items():
    merged_filter_count[uniqueid] = sum(report_by_srr[srr]["filter_count"] for srr, index in srrs)
    index = uniqueid.rsplit("_", 1)[0]
    index_to_uniqueids.setdefault(index, []).append(uniqueid)

discard_reason = {}
for uniqueid, filter_count in merged_filter_count.items():
    if filter_count < cutoff:
        discard_reason[uniqueid] = "insufficient_reads"

for index, uniqueids in index_to_uniqueids.items():
    if any(uniqueid in discard_reason for uniqueid in uniqueids):
        for uniqueid in uniqueids:
            discard_reason.setdefault(uniqueid, "pair_discarded")

write_merge_outputs(index_to_srr, srr_to_line, set(discard_reason))

progress("Writing fastq status report")
with open(report_fname, "w") as fout:
    print("geoStudy", "sraExp", "sraAcc", "expType", "sampleType", "cellType", "libraryType", "index",
            "fastq.count", "fastq.lenavg", "fastq.lensd",
            "trimmed.count", "trimmed.lenavg", "trimmed.lensd",
            "filtered.count", "filtered.lenavg", "filtered.lensd",
            "trimmed.left.pct", "filtered.left.pct", "total.left.pct",
            "status", "reason",
            sep="\t", file=fout)
    total = len(records)
    progress("Writing fastq status report", 0, total)
    for i, record in enumerate(records, 1):
        uniqueid = f'{record["index"]}_{record["sampletype"]}'
        reason = discard_reason.get(uniqueid, "included")
        status = "discard" if uniqueid in discard_reason else "keep"
        print(record["gse"], record["srx"], record["srr"], record["condition"], record["sampletype"], record["celltype"], record["library"], record["index"],
                *record["stats"], *record["ratios"], status, reason,
                sep="\t", file=fout)
        progress("Writing fastq status report", i, total)

n_total = len(index_to_srr)
n_discard = len(discard_reason)
n_keep = n_total - n_discard
progress(f"Done. cutoff={int(cutoff) if cutoff.is_integer() else cutoff}, kept={n_keep}, discarded={n_discard}, report={report_fname}")

