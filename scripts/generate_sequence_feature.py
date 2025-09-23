#!/usr/bin/env python3
# Program:     DuET v1.0.0
# Author:      Sungho Lee, Jae-Won Lee, Jinhee Hong
# Affiliation: MOGAM Institute for Biomedical Research
# Contact:     sungho.lee@mogam.re.kr; jaewon.lee@mogam.re.kr; jinhee.hong@mogam.re.kr
# Citation:    TBD


import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import RNA
import builtins
import re
import signal

import pandas as pd
import numpy as np
import subprocess as sp

from os import popen
from functools import partial, singledispatch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from Bio.SeqUtils import gc_fraction
from itertools import product
from collections import Counter
from typing import Callable, Final, Optional
from pathlib import Path
from copy import deepcopy

from data.utils import CODON_TO_AA, codon_weight


# Define global variables
URIDINE_CHAR: str = "T"
BASES: list[str] = ["A", "C", "G", URIDINE_CHAR]
START_CODONS: Final[set[str]] = {f"A{u}G" for u in "TUMP1"} # pU, m1pU

CODON_TABLE: dict[str, str] = CODON_TO_AA
CODON_WEIGHTS: dict[str, float] = codon_weight
CODONS: list[str] = sorted(CODON_TABLE.keys())
AMINO_ACIDS: Final[list[str]] = sorted(set(CODON_TABLE.values()))

PRECISION: int = 4
ASSERTION: bool = False  # set to True for debugging purposes

PARAM_PSEUDO_U: Final = RNA.sc_mod_read_from_jsonfile("misc/rna_mod_pseudouridine_parameters.json")
PARAM_M1PSEUDO_U: Final = RNA.sc_mod_read_from_jsonfile("misc/rna_mod_n1methylpseudouridine_parameters.json")

# (gcc)gccRccAUGG(cg) in human; gccRccAUGG for consensus score calculation
KOZAK_PSSM: Final = np.array([
    [-0.25153876, -0.39592867, -0.05889368, 0.87970577, 0.21412481, -0.39592867, -0.18442456],
    [-0.12029423, 0.35614381, 0.60407133, -1.32192808, 0.60407133, 0.84799691, -0.73696558],
    [0.64154603, 0.26303441, 0.05658353, 0.56559718, -0.32192809, 0.16349874, 0.97085366],
    [-0.55639334, -0.39592867, -1.05889368, -1.83650125, -0.94341646, -1.64385617, -0.83650126]
    ]) # A, C, G, T


# Define subroutines
def print(*objects, **kwargs):
    """Overrides builit-in print function to instantly print messages."""
    
    kwargs["flush"] = True
    return builtins.print(*objects, **kwargs)


def get_number_uAUGs(seq: str, is_seq_frame_aligned: bool=True) -> tuple[int, int, int]:
    """Returns number of in-frame, out-frame, total uAUGs from the input sequence.
    Set is_seq_frame_aligned=True for CDS and 3' UTR; False for 5' UTR and full sequence."""

    frame_basis = 0 if is_seq_frame_aligned else len(seq) % 3
    inframe, outframe = 0, 0

    s = seq.upper()

    for i in range(len(s)-2):
        if s[i:i+3] in START_CODONS:
            if (len(s) - i) % 3 == frame_basis:
                inframe += 1
            else:
                outframe += 1

    return inframe, outframe, inframe + outframe


def get_number_ORFs(seq: str, is_seq_frame_aligned: bool=True, URIDINE_CHAR: str="T"): 
    """Returns number of in-frame, out-frame, total ORFs from the input sequence.
    Set is_seq_frame_aligned=True for CDS and 3' UTR; False for 5' UTR and full sequence."""
    
    frame_basis = 0 if is_seq_frame_aligned else len(seq) % 3
    inframe, outframe = 0, 0
    
    orf = re.compile("(?=((AXG)([AXGC]{3})+(XAA|XAG|XGA)))".replace("X", URIDINE_CHAR))
    for start_idx in [m.start() for m in re.finditer(orf, seq)]:
        if start_idx % 3 == frame_basis:
            inframe += 1
        else:
            outframe += 1
    
    return inframe, outframe, inframe + outframe


def gc_ratio(seq: str, **kwargs) -> float:
    """Returns GC ratio of the input sequence."""
    
    return round(gc_fraction(seq, **kwargs), PRECISION)


def fold(seq: str, unpaired_idx: list[int]=[], paired_idx: list[int]=[]) -> float:
    """Returns (SS string, MFE,) length normalized MFE for the input sequence.
    Applies hard constraints with base indices in unpaired_idx, paired_idx (0-based)."""

    if ASSERTION:
        assert not (shared := set(unpaired_idx) & set(paired_idx)), \
            f"unpaired_idx and paired_idx shares element {shared}"
            
        assert len(seq) <= 32767, "Sequence length exceeds RNAfold limit (32767, SHRT_MAX in C)."

    seq = seq[:32767]  # truncate the sequence to avoid RNAfold limit
    if seq == "":
        return np.nan

    if "P" in seq:
        param = PARAM_PSEUDO_U
        seq = seq.replace("P","U")
    elif "1" in seq:
        param = PARAM_M1PSEUDO_U
        seq = seq.replace("1","U")
    else:
        param = None  # no modification

    fc = RNA.fold_compound(seq)
    if param is not None:
        fc.sc_mod(param, list(range(1, len(seq)+1)))

    if unpaired_idx:
        for idx in unpaired_idx:
            fc.hc_add_up(idx+1)

    if paired_idx:
        for idx in paired_idx:
            fc.hc_add_bp_nonspecific(idx+1)

    ss, mfe = fc.mfe()  # mfe is in kcal/mol, ss is dot-bracket notation

    return round(mfe / len(seq), PRECISION)


def fold_LP(seq: str) -> tuple[str, float]:
    """Returns (SS string, MFE,) length normalized MFE for the input sequence.
    Uses LinearPartition instead of RNAfold API.
    Cannot process sequence modification nor hard constraints."""
    
    if seq == "":
        return 0.0
    
    command = (f"echo {seq} | "
               "/fsx/s3/public_data/program/mRNA/LinearPartition/linearpartition "
               "-V -c 0.000001 1> /dev/null")
    process = sp.Popen(command, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    _, err = process.communicate()
    
    mfe = float(err.decode("utf-8").strip().split(' ')[4])
    
    return round(mfe / len(seq), PRECISION)  # length normalized MFE


def get_AUG_context(utr5: str, radius: int) -> tuple[int, int]:
    """Returns start and end index of AUG context of given radius.
    Actual 'radius' is the twice of give radius plus 3, capped at 5' prime."""
    
    utr5_length = len(utr5)
    
    return (max(utr5_length - radius, 0), utr5_length + radius + 3) # 3bp for AUG codon 


def get_Kozak_similarity(seq: str, pssm: np.array=KOZAK_PSSM, 
                         base_xref: dict[str, int]={'A':0, 'C':1, 'G':2, 'T':3, 'U':3}) -> float:
    """Returns Kozak similarity score of the input sequence.
    Requires last 9bp of 5' UTR and first 3bp of CDS after AUG, concatenated."""

    if len(seq) < len(pssm[0]):
        return 0.0 # TODO: is 0.0 correct for default value?

    score = sum([pssm[base_xref.get(base), idx] for idx, base in enumerate(seq.upper())])
    
    return float(score)


def get_AUG_context_stats(seqs: str, radius: int=30):
    """Returns start context norm. MFE, GC ratio and delta-delta-G of the input sequence.
    Requires [utr5];[fullseq] format."""
    
    utr5, seq = seqs.split(";")
    
    aug_context_radius = get_AUG_context(utr5, radius)
    aug_context_idx = list(range(*aug_context_radius))
    
    aug_context_gc = gc_ratio(seq[slice(*aug_context_radius)], ambiguous="ignore")
    aug_context_mfe = fold(seq[slice(*aug_context_radius)])
    aug_context_kozak = get_Kozak_similarity(utr5[-6:] + seq[3])
    
    # skip ddg calculation
    #unconstrained_mfe = fold(seq) # normalized one
    #constrained_mfe = fold(seq, unpaired_idx=aug_context_idx)

    return aug_context_mfe, aug_context_gc, round(aug_context_kozak, PRECISION)


def get_frequency(pool: list, counts: Counter) -> dict[str, float]:
    """Returns frequency of each items in the input sequence for each item in a pool."""
    
    total_counts = sum(counts.values())
    if total_counts == 0:  # counts were based on an empty string
        total_counts = 1
        
    if outlier := counts.keys() - set(pool):
        raise RuntimeError(f"Counts contains items not in the pool: {outlier}")

    return tuple(round(counts[item]/total_counts, PRECISION) for item in pool)


def get_kmer_frequency(seq: str, k: int) -> dict[str, float]:
    """Returns k-mer frequency of the input sequence."""
    
    seq = seq.upper()
    kmers = ["".join(i) for i in product(BASES, repeat=k)]
    kmers_in_seq = Counter([seq[i:i+k] for i in range(0, len(seq) - k + 1)])
    
    return get_frequency(kmers, kmers_in_seq)


def get_base_frequency(seq: str) -> dict[str, float]:
    """Returns base frequency of the input sequence."""
    
    seq = seq.upper()
    bases_in_seq = Counter(list(seq))
    
    return get_frequency(BASES, bases_in_seq)


def get_wobble_base_frequency(cds: str) -> dict[str, float]:
    """Returns wobble position base frequency of the input CDS.
    Input sequence must be a CDS whose length is a multiple of 3."""

    if ASSERTION:
        assert len(cds) % 3 == 0, "CDS length is not in triplet"
            
    cds = cds.upper()[:len(cds) - (len(cds) % 3)]
    
    bases_in_wobble = Counter([cds[k] for k in range(2, len(cds), 3)])
    
    return get_frequency(BASES, bases_in_wobble)


def get_codon_frequency(cds: str) -> dict[str, float]:
    """Returns codon frequency of the input CDS.
    Input sequence must be a CDS whose length is a multiple of 3."""
    
    if ASSERTION:
        assert len(cds) % 3 == 0, "CDS length is not in triplet"
            
    cds = cds.upper()[:len(cds) - (len(cds) % 3)]
    
    codons_in_cds = Counter([cds[k:k+3] for k in range(0, len(cds), 3)])
    
    return get_frequency(CODONS, codons_in_cds)


def get_AA_frequency(cds: str) -> dict[str, float]:
    """Returns amino acid frequency of the input CDS.
    Input sequence must be a CDS whose length is a multiple of 3."""
    
    if ASSERTION:
        assert len(cds) % 3 == 0, "CDS length is not in triplet"
            
    cds = cds.upper()[:len(cds) - (len(cds) % 3)]
    
    if ASSERTION:
        AAs_in_cds = Counter([CODON_TABLE[cds[k:k+3]] for k in range(0, len(cds), 3)])
    else:
        AAs_in_cds = Counter([CODON_TABLE.get(cds[k:k+3], "NA") for k in range(0, len(cds), 3)])

    return get_frequency(AMINO_ACIDS, AAs_in_cds)


def get_cai(cds: str, ramp_length: Optional[int]=None) -> float:
    """Returns codon adaptation index (CAI) of the input CDS.
    Input sequence must be a CDS whose length is a multiple of 3."""
    
    if ASSERTION:
        assert len(cds) % 3 == 0, "CDS length is not in triplet"
            
    cds = cds.upper()[:len(cds) - (len(cds) % 3)]
    if ramp_length is not None:
        assert ramp_length % 3 == 0, "ramp_length is not in triplet"
        cds = cds[:ramp_length]
    
    if ASSERTION:
        weights = [CODON_WEIGHTS[cds[k:k+3]] for k in range(0, len(cds), 3)]
    else:
        weights = [CODON_WEIGHTS.get(cds[k:k+3], np.nan) for k in range(0, len(cds), 3)]
        
    cai = np.exp(np.nanmean(np.log(weights)))
    
    return round(cai, PRECISION)


def get_log_length(seq: str, base: int=10, nan_default: float=-1.0) -> float:
    """Returns log length of the input sequence with given base.
    Returns nan_default(-1.0 by default) for a sequence of length 0."""

    if len(seq) == 0:
        return nan_default
    else:
        return round(np.emath.logn(base, len(seq)), PRECISION)


# metadata items naming convention: either 'name' or ('group_name', 'name1', 'name2', ...)
uAUG_colnames = ("uAUG", "numInFrameAUG", "numOutFrameAUG", "numAUG")
uORF_colnames = ("uORF", "numInFrameORF", "numOutFrameORF", "numORF")
baseFreq_colnames = ("baseFreq", *(f"baseFreq{k}" for k in BASES))
codonFreq_colnames = ("codonFreq", *(f"codonFreq{k}" for k in CODONS))
aaFreq_colnames = ("aaFreq",  *(f"aaFreq{k}" for k in AMINO_ACIDS))
startContext_colnames = ("startContext", "normMFE", "gcRatio", "kozakScore")#, "ddG")

parallel_keywords = ("normMFE", "ddG", "AUG", "ORF",)

# this dictionary controls the behavior of the program
metadata_items = {
    "base": {
        "logLength": get_log_length,
        "gcRatio": partial(gc_ratio, ambiguous="ignore"),
        "normMFE": fold_LP,  # use LinearPartition for fast calculation
        uAUG_colnames: get_number_uAUGs,
        uORF_colnames: get_number_ORFs,
        baseFreq_colnames: get_base_frequency,
    },
    "utr5": {
        uAUG_colnames: partial(get_number_uAUGs, is_seq_frame_aligned=False),
        uORF_colnames: partial(get_number_ORFs, is_seq_frame_aligned=False),
    },
    "cds": {
        codonFreq_colnames: get_codon_frequency,
        aaFreq_colnames: get_AA_frequency,
        "CAI": get_cai,
        "rampCAI": partial(get_cai, ramp_length=30),
    },
    "utr3": {},
    "fullseq": {
        uAUG_colnames: partial(get_number_uAUGs, is_seq_frame_aligned=False),
        uORF_colnames: partial(get_number_ORFs, is_seq_frame_aligned=False),
    },
    "startcontext": {
        startContext_colnames: get_AUG_context_stats,
    },
}


# Define business logics
@singledispatch
def calculate_metadata():
    '''Generic wrapper function of metadata calculation.'''
    ...


@calculate_metadata.register(pd.DataFrame)
def calculate_metadata_dataframe(data: pd.DataFrame, 
                                 region: str,
                                 colname: str, 
                                 metadata_items: dict[str, Callable],
                                 index_is_colname=False, 
                                 threads=-1,
                                 cols_to_parallel=parallel_keywords) -> pd.DataFrame:
    '''Calculate metadata for a dataframe and returns metadata dataframe of same row size.'''

    if index_is_colname:
        result = pd.DataFrame(data.index, columns=[colname])
    else:
        result = pd.DataFrame(data[colname])

    results = [result]
    for metadata_item, func in metadata_items.items():

        # define is_multiple_columns, metadata_name and do_parallel
        if is_multiple_columns := (not isinstance(metadata_item, str)):
            metadata_name, *metadata_item = metadata_item
            metadata_name += f" ({len(metadata_item)} items)"
            do_parallel = any(k in m for k in cols_to_parallel for m in metadata_item)
            metadata_item = [f"{region}_{item}" for item in metadata_item]
        else:
            metadata_name = metadata_item
            do_parallel = any(k in metadata_item for k in cols_to_parallel)
            metadata_item = f"{region}_{metadata_item}"

        if do_parallel:
            def signal_handler(sig, frame):
                print(f"Received signal {sig}. Terminating...")
                os.killpg(0, signal.SIGKILL)
                exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            temp = pd.DataFrame(
                process_map(func, data[colname], 
                            chunksize=100, max_workers=threads,
                            desc=f"Calculating {metadata_name}..",),
                columns=metadata_item if is_multiple_columns else [metadata_item])
            results.append(temp)
        else:
            tqdm.pandas(desc=f"Calculating {metadata_name}..")
            if is_multiple_columns:
                results.append(pd.DataFrame(result[colname].progress_apply(func).tolist(),
                                            columns=metadata_item))
            else:
                results.append(result[colname].progress_apply(func).to_frame(metadata_item))

    return pd.concat(results, axis=1)


@calculate_metadata.register(str)
def calculate_metadata_row(seq: str) -> list:
    '''Calculate metadata for a single sequence and returns list of metadata values.'''

    result = [seq]
    for metadata_item, func in metadata_items.items():
        if isinstance(metadata_item, str):
            result.append(func(seq))
        else:
            result.extend(func(seq))

    return result


if __name__ == "__main__":
    
    import argparse
    from argparse import RawTextHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    
    parser.add_argument("data", metavar="FNAME", type=str, nargs="+",
                        help="File(s) to calculate metadata. Column names must match between files.")
    parser.add_argument("-5", "--utr5", metavar="COLNAME", type=str,
                        help="Column name of 5' UTR sequences.")
    parser.add_argument("-c", "--cds", metavar="COLNAME", type=str,
                        help="Column name of CDS sequences.")
    parser.add_argument("-3", "--utr3", metavar="COLNAME", type=str,
                        help="Column name of 3' UTR sequences.")
    parser.add_argument("-f", "--full-seq", metavar="COLNAME", type=str,
                        help="Column name of full sequences.")
    parser.add_argument("--infer-fullseq", action="store_true",
                        help="Generate full_seq column from the data.\nNote: This only works when all of -5, -c and -3 specified.")
    parser.add_argument("--infer-start-context", action="store_true",
                        help="Generate start_context column from the data.\nNote: This only works when -f and -5 specified.")
    parser.add_argument("-s", "--start-context-radius", type=int, default=15,
                        help="""radius of AUG context to calculate start context metadata. Default: %(default)s.
Note: Actual start context span will be the twice of the given radius plus 3, capped at 5' prime.""")
    parser.add_argument("-d", "--delim", type=str,
                        help="Universal file delimeter. Default: auto-detect tab or comma from the file.")
    parser.add_argument("-u", "--uridine", type=str, default=URIDINE_CHAR,
                        help="Uridine character to use. Default: %(default)r")
    parser.add_argument("-t", "--threads", type=int, default=-1,
                        help="Threads to use for multiprocessed calculation. Default: %(default)s (all cores)")
    parser.add_argument("-p", "--precision", type=int, default=PRECISION,
                        help="Precision of the calculated values. Default: %(default)s.")
    parser.add_argument("--separate-output", action="store_true",
                        help="Save metadata files separately for each input column per file.")
    parser.add_argument("-o", "--overwrite", action="store_true",
                        help="For separate output mode, overwrite existing output files if they exist.")
    parser.add_argument("--enable-assertion", action="store_true",
                        help="Enable assertions for debugging purposes. Use with caution.")
    
    args = parser.parse_args() 
    
    if args.enable_assertion:
        ASSERTION = True
        print("Assertions are enabled. Use with caution.")
    
    stage_separator = "\n" + "-" * 80 + "\n"
    print()
    
    if args.delim and args.delim == "\\t":
        args.delim = "\t"
    
    if not any([args.utr5, args.cds, args.utr3]):
        parser.error("At least one column name must be specified.")

    columns = {
        "utr5": args.utr5,
        "cds": args.cds,
        "utr3": args.utr3,
        "fullseq": args.full_seq,
    }
    columns = {k:v for k,v in columns.items() if v}  # remove None values
    
    if args.infer_fullseq:
        if all([args.utr5, args.cds, args.utr3]):
            print(f"Full sequence column will be inferred from {', '.join([args.utr5, args.cds, args.utr3])}.")
            columns["fullseq"] = "fullseq"
        else:
            parser.error("--infer-fullseq requires all of -5, -c and -3 to be specified.")

    if args.infer_start_context:
        if args.utr5 and "fullseq" in columns:
            print(f"Start context column will be inferred from {args.utr5} and {columns['fullseq']}.")
            columns["startcontext"] = "start"
        else:
            parser.error("--infer-startcontext requires full sequence and -5 to be specified.")

    if args.uridine != URIDINE_CHAR:  # update URIDINE_CHAR and BASES
        print(f"Setting '{URIDINE_CHAR}' as default uridine character.")
        
        del metadata_items["base"][baseFreq_colnames]
        del metadata_items["cds"][codonFreq_colnames]
        
        CODON_TABLE = {k.replace(URIDINE_CHAR, args.uridine):v for k,v in CODON_TABLE.items()}
        CODONS: list[str] = sorted(CODON_TABLE.keys())
        URIDINE_CHAR = args.uridine
        BASES = ["A", "C", "G", args.uridine]
        
        baseFreq_colnames = ("baseFreq", *(f"baseFreq{k}" for k in BASES))
        metadata_items["base"][baseFreq_colnames] = get_base_frequency
        
        codonFreq_colnames = ("codonFreq", *(f"codonFreq{k}" for k in CODONS))
        metadata_items["cds"][codonFreq_colnames] = get_codon_frequency
        
    if args.precision != PRECISION:
        print(f"Setting precision to {args.precision}.")
        PRECISION = args.precision
        
    if args.threads == -1:
        args.threads = int(popen("nproc").read().strip())
        print(f"Using all available cores: {args.threads}.")
    else:
        print(f"Using {args.threads} threads for parallel calculation of {', '.join(parallel_keywords)}.")
    
    for fname in args.data:
        print(stage_separator)
        if not args.delim:
            delim = "\t" if "\t" in open(fname).readline() else ","
        else:
            delim = args.delim
        suffix = ".tsv" if delim == "\t" else ".csv"

        print(f"Processing file: {fname} with delimiter {delim!r}...", end="\t")
        df = pd.read_csv(fname, sep=delim, header=0, index_col=None)
        print(f"Found {len(df):,} entries.")
        
        if args.infer_fullseq:
            df[columns["fullseq"]] = df[columns["utr5"]] + df[columns["cds"]] + df[columns["utr3"]]
        
        if args.infer_start_context:
            df[columns["startcontext"]] = df.apply(
                lambda row: row[columns["utr5"]] + ";" + row[columns["fullseq"]], 
                axis=1)
        
        outputs = []
        for region, colname in columns.items():
            print(stage_separator)
            
            current_output_fname = fname.replace(suffix, f"_metadata_{region}{suffix}")
            if args.separate_output:
                if not args.overwrite:
                    if Path(current_output_fname).exists():
                        print(f"Output file for {region} already exists. Skipping...")
                        continue
            
            df[colname] = df[colname].fillna("").astype(str)
            
            if region == "startcontext":
                region_metadata_items = metadata_items["startcontext"]
            else:
                region_metadata_items = deepcopy(metadata_items["base"])
                region_metadata_items.update(metadata_items[region])
            
            print(f"For {colname!r} column as {region}, following metadata columns will be calculated:")
            formatted_keys = [key if isinstance(key, str) else ", ".join(key[1:]) for key in region_metadata_items.keys()]
            print(", ".join(formatted_keys)); print()
            data = calculate_metadata(df, region, colname, region_metadata_items, threads=args.threads).drop(columns=colname)

            if not args.separate_output:
                outputs.append(data)
            else:
                data.to_csv(current_output_fname, sep=delim, index=False)

        if not args.separate_output:
            pd.concat(outputs, axis=1).to_csv(fname.replace(suffix, "_metadata.tsv"), sep=delim, index=False)
