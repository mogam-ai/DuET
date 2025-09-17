# Program:     DuET v1.0.0
# Author:      Sungho Lee, Jae-Won Lee, Jinhee Hong
# Affiliation: MOGAM Institute for Biomedical Research
# Contact:     sungho.lee@mogam.re.kr; jaewon.lee@mogam.re.kr; jinhee.hong@mogam.re.kr
# Citation:    TBD


import sys
import os
import h5py
import subprocess as sp
import pandas as pd

from pathlib import Path


project_path = Path("../attribution_score_results")
input_dir = project_path / "saliency_dedup_500+1500"
modisco_result = input_dir / "2.modisco_result"
modisco_report = input_dir / "3.modisco_report"
meme_result = input_dir / "4.meme_result"
tomtom_result = input_dir / "5.tomtom_result"
reference_file = Path("../misc/Ray2013_rbp_Homo_sapiens.dna_encoded.meme")

os.makedirs(modisco_result, exist_ok=True)
os.makedirs(modisco_report, exist_ok=True)
os.makedirs(meme_result, exist_ok=True)
os.makedirs(tomtom_result, exist_ok=True)
os.makedirs("modisco_tmp", exist_ok=True)


def run_modisco(name, window_size=7, max_seqlets=4000, motif_width=400, use_normalized=True):
    cmd = (f"modisco motifs -s {input_dir}/{name}/utr5_500+1500_input.npz "
           f"-a {input_dir}/{name}/utr5_500+1500_attr{'_norm' if use_normalized else ''}.npz "
           f"-z {window_size} -n {max_seqlets} "
           f"-o {modisco_result}/{name}_5UTR.h5 -w {motif_width}")

    result = sp.run(cmd, shell=True, capture_output=True)
    print(result.stderr)
    read_modisco_result(name)


def read_modisco_result(fname):
    fname = f"{modisco_result}/{fname}_utr5.h5"
    
    with h5py.File(fname, "r") as f:
        print("Keys in the H5 file:")
        for key in f.values():
            print(key)


def run_modisco_report(celltype):
    cmd = (f"modisco report -t "
           f"-i {modisco_result}/{celltype}_utr5.h5 "
           f"-o {modisco_report}/{celltype}_utr5")
    
    result = sp.run(cmd, shell=True, capture_output=True)
    print(result.stderr)


def compare_known_motif(celltype):
    cmd = (f"modisco report -t "
           f"-i {modisco_result}/{celltype}_utr5.h5 "
           f"-o {meme_result}/{celltype}_utr5 "
           f"-s {modisco_result}/{celltype}_utr5 "
           f"-m {reference_file}")
    
    result = sp.run(cmd, shell=True, capture_output=True)
    print(result.stderr)


def merge_filter_tomtom_files(celltype, p_cutoff=1.0, q_cutoff=1.0):
    tomtom_result = Path(f"{meme_result}/{celltype}_utr5/tomtom")
    try:
        tomtom_result_files = os.listdir(tomtom_result)
    except FileNotFoundError:
        print("TOMTOM file not found.")
        return None
    
    df_list = []
    for result_file in tomtom_result_files: 
        try:
            df = pd.read_csv(tomtom_result.joinpath(result_file), sep="\t").dropna()
            df["name"] = result_file.split(".tomtom.tsv")[0] #tomtom result 이름 열 추가
            df_list.append(df)
        except pd.errors.EmptyDataError:
            print("Dataframe is empty. Skipping this file.")
            
    if len(df_list) >= 1: 
        df = pd.concat(df_list).reset_index(drop=True)
        df = df[(df["p-value"] <= p_cutoff) & (df["q-value"] <= q_cutoff)]
        df = df.groupby("name").apply(lambda x: x.iloc[0]).reset_index(drop=True)
        
        if p_cutoff == 1.0 and q_cutoff == 1.0: 
            df.to_csv(project_path.joinpath(f"5.tomtom_result/{celltype}_motif.csv"), sep="\t", index=False)
        else: 
            df.to_csv(project_path.joinpath(f"5.tomtom_result/{celltype}_filtered_motif.csv"), sep="\t", index=False)
        
        return df
    else:
        return None


def process_file(celltype, window_size=10, use_normalized=True):
    try:
        run_modisco(celltype, window_size=window_size, use_normalized=use_normalized)
    except Exception as e: 
        print(f"Error {type(e).__name__} occured while running TF-modisco in {celltype}. Aborting...")
        return
    
    run_modisco_report(celltype)
    
    compare_known_motif(celltype)
    
    merge_filter_tomtom_files(celltype)
    merge_filter_tomtom_files(celltype, p_cutoff=0.05, q_cutoff=0.2)


def run_sbatch(celltype_file):
    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name={celltype_file}_modisco
#SBATCH --output=tmp/modisco_output_{celltype_file}.log
#SBATCH --error=tmp/modisco_error_{celltype_file}.log
#SBATCH --partition=cpu-r6-4x
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

source ~/.bashrc
conda activate MTtrans

python 2.run_modisco_sbatch.py {celltype_file}
"""
    script_filename = f"./tmp/sbatch_modisco_{celltype_file}.sh"
    with open(script_filename, "w") as f:
        f.write(sbatch_script)
    sp.run(f"sbatch {script_filename}", shell=True)
    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        celltype_file = sys.argv[1]
        process_file(celltype_file, window_size=7, norm=True)
    else:
        celltype_files = os.listdir(input_dir)
        
        for celltype_file in celltype_files:
            run_sbatch(celltype_file)

