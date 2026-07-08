#!/usr/bin/env python3
# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD
"""Build the gene-metadata table required by calculate_te.py.

Reconstructs per-transcript 5'UTR / CDS / 3'UTR sequences by slicing the DuET
transcriptome FASTA with the region lengths in sequence_features.tsv, so the
90 MB full-sequence metadata table does not need to be distributed.

Region convention (verified against GENCODE v47 selected transcripts):
    seqLen == utr5Len + cdsLen + utr3Len == len(transcript)
    utr5 = seq[0 : utr5Len]
    cds  = seq[utr5Len : utr5Len + cdsLen]     # includes start (ATG) and stop codon
    utr3 = seq[utr5Len + cdsLen : ]

Output columns: txID, utr5, cds, utr3 (tab-separated; txID is the first column,
read as the index by calculate_te.py).
"""
import argparse
import sys


def read_fasta(path):
    """Yield (id, sequence). Record id is the first whitespace-delimited token."""
    seq_id, chunks = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, "".join(chunks)
                seq_id, chunks = line[1:].split(None, 1)[0], []
            else:
                chunks.append(line.strip())
    if seq_id is not None:
        yield seq_id, "".join(chunks)


def load_lengths(path):
    """Read txID -> (utr5Len, cdsLen, utr3Len, seqLen) from sequence_features.tsv."""
    import csv
    lengths = {}
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        idx = {c: i for i, c in enumerate(header)}
        for col in ("txID", "utr5Len", "cdsLen", "utr3Len", "seqLen"):
            if col not in idx:
                sys.exit(f"ERROR: column '{col}' missing from {path}")
        for row in reader:
            lengths[row[idx["txID"]]] = (
                int(row[idx["utr5Len"]]),
                int(row[idx["cdsLen"]]),
                int(row[idx["utr3Len"]]),
                int(row[idx["seqLen"]]),
            )
    return lengths


STOP_CODONS = {"TAA", "TAG", "TGA"}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasta", default="../datasets/transcriptome/duet_transcriptome.selected.fa",
                   help="DuET transcriptome FASTA (default: %(default)s)")
    p.add_argument("--features", default="../datasets/sequence_features.tsv",
                   help="Table with txID/utr5Len/cdsLen/utr3Len/seqLen (default: %(default)s)")
    p.add_argument("-o", "--output", default="gene_metadata.tsv",
                   help="Output gene-metadata TSV (default: %(default)s)")
    p.add_argument("--strict", action="store_true",
                   help="Exit on the first length/ATG/stop inconsistency instead of warning")
    args = p.parse_args()

    lengths = load_lengths(args.features)
    print(f"loaded lengths for {len(lengths)} transcripts")

    written = 0
    skipped = 0
    n_len_mismatch = n_no_atg = n_no_stop = 0

    with open(args.output, "w") as out:
        out.write("txID\tutr5\tcds\tutr3\n")
        for tx, seq in read_fasta(args.fasta):
            seq = seq.upper().replace("U", "T")
            if tx not in lengths:
                skipped += 1
                continue
            u5, cd, u3, sl = lengths[tx]

            if len(seq) != sl or (u5 + cd + u3) != sl:
                n_len_mismatch += 1
                msg = f"length mismatch {tx}: len(seq)={len(seq)} seqLen={sl} u5+cd+u3={u5+cd+u3}"
                if args.strict:
                    sys.exit("ERROR: " + msg)
                print("WARNING:", msg, file=sys.stderr)
                skipped += 1
                continue

            utr5 = seq[:u5]
            cds = seq[u5:u5 + cd]
            utr3 = seq[u5 + cd:]

            # Sanity checks on the CDS boundary (start/stop codon included in cdsLen).
            if cd >= 3 and cds[:3] != "ATG":
                n_no_atg += 1
                if args.strict:
                    sys.exit(f"ERROR: CDS of {tx} does not start with ATG")
            if cd >= 3 and cds[-3:] not in STOP_CODONS:
                n_no_stop += 1
                if args.strict:
                    sys.exit(f"ERROR: CDS of {tx} does not end with a stop codon")

            out.write(f"{tx}\t{utr5}\t{cds}\t{utr3}\n")
            written += 1

    print(f"wrote {written} transcripts to {args.output} (skipped {skipped})")
    if n_len_mismatch or n_no_atg or n_no_stop:
        print(f"  warnings: length_mismatch={n_len_mismatch}, "
              f"no_ATG_start={n_no_atg}, no_stop_end={n_no_stop}", file=sys.stderr)


if __name__ == "__main__":
    main()
