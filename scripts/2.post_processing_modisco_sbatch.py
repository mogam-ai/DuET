#%% 
# modisco를 run하기 위한 목적
import numpy as np
import subprocess as sp
import h5py, os, sys
from IPython.display import HTML
from pathlib import Path

import pandas as pd

# must run with conda activate MTtrans due to revised code

os.makedirs('tmp', exist_ok=True) # tmp file for log and sh script file

# modisco에서 나온 중간 파일의 directory 생성하기
project_path = Path('/fsx/home/jhhong/2024_project/3.RNA_UTR/results_attribution_score')
input_dir = project_path.joinpath('saliency_wo_dedup_UTR5_500')
modisco_result = input_dir.joinpath('2.modisco_result') # 
modisco_report = input_dir.joinpath('3.modisco_report')
meme_result = input_dir.joinpath('4.meme_result')
tomtom_result = input_dir.joinpath('5.tomtom_result')
reference_file = Path('/fsx/s3/project/P240017_mRNA_UTR/motif_explain/motif_databases/RNA/Ray2013_rbp_Homo_sapiens.dna_encoded.meme')

os.makedirs(modisco_result, exist_ok=True)
os.makedirs(modisco_report, exist_ok=True)
os.makedirs(meme_result, exist_ok=True)
os.makedirs(tomtom_result, exist_ok=True)
# assert os.path.isdir(tomtom_result), f'there is no {tomtom_result}'


def run_modisco(name, window_size = 7, norm=True):
    """[1] Run modisco and get pattern
        * input : deepSHAP [ N * one-hot encoding * length ]
        * modisco parameter options: 
            - window = 500 (read all 5'UTR)
            - window_size = 15
            - num_seqlet = 2000
        * return: 
            - h5 file
    """
    print(f'name:{name}') #check input data is made
    input_name = '5UTR_500input.npz'
    if norm:
        attr_name = '5UTR_500attr_norm.npz'
    else:
        attr_name = '5UTR_500attr.npz'
    
    # cmd = f'modisco motifs -s {input_dir}/{name}/5UTR_500input.npz -a {input_dir}/{name}/5UTR_500attr.npz -z {window_size} -n 4000 \
    # -o {modisco_result}/{name}_5UTR.h5 -w 500 -v'
    # cmd = f'modisco motifs -s {input_dir}/{name}/5UTR_500input.npz -a {input_dir}/{name}/5UTR_500attr.npz \
    # -o {modisco_result}/{name}_5UTR.h5 -n 4000 -w 100 -z 7 -t 10 -f 10 -g 5 -v'
    cmd = f'modisco motifs \
        -s {input_dir}/{name}/{input_name} \
        -a {input_dir}/{name}/{attr_name} \
        -z {window_size} -n 4000 \
        -o {modisco_result}/{name}_5UTR.h5 -w 400'
    """modisco run option
    -n (--max_seqlets): metacluster당 최대 seqlet 개수
    -l (--n_leiden): 수행할 Leiden clustering 횟수
    -w (--window): 피크 중심을 둘러싼 영역으로, motif 발견을 위해 고려할 window 크기
    """
    result = sp.run(cmd, shell=True, capture_output=True)
    read_modisco_result(name)

def read_modisco_result(filename):
    # Motif가 몇개 나왔는지 count
    filename = f'{modisco_result}/{filename}_5UTR.h5'
    with h5py.File(filename, 'r') as h5_file:
        # 파일 내의 모든 키 출력
        print("Keys in the H5 file:")
        for key in h5_file.values():
            print(key)

def run_modisco_report(name):
    """ Report modisco
    * input: modisco result
    * output: modisco report
    """
    cmd = f'modisco report -t \
    -i {modisco_result}/{name}_5UTR.h5 \
    -o {modisco_report}/{name}_5UTR'
    result = sp.run(cmd, shell=True, capture_output=True)
    print('2번 run modisco report')
    print(result.stderr)
    return 

def compare_known_motif(name):
    """modisco안의 meme 기능 활용해서 비교하기
    환경 MTtrans에서만 되는 듯?
    """ 
    cmd = f'modisco report -t \
        -i {modisco_result}/{name}_5UTR.h5 \
        -o {meme_result}/{name}_5UTR \
        -s {modisco_result}/{name}_5UTR \
        -m {reference_file}'
    print(cmd)
    result = sp.run(cmd, shell=True, capture_output=True)
    print(result.stderr)
    return 

def filtering_df(df, p_cutoff, q_cutoff):
    """  결과 파일의 후보군 추출 (pvalue<=0.05, qvalue<=0.1)   """
    filtered_df = df[(df['p-value'] <= p_cutoff) & (df['q-value'] <= q_cutoff)]
    return filtered_df


def merge_filter_tomtom_files(name, p_cutoff, q_cutoff, filter=True):
    '''
    pos/neg individual files merge & filter => DF
    '''
    tomtom_result = Path(f'{meme_result}/{name}_5UTR/tomtom') #tomtom으로 known motif와 비교
    try:
        tomtom_result_files = os.listdir(tomtom_result)
    except FileNotFoundError as e:
        print('no tomtom file')
        return None
    
    df_list = []
    
    for result_file in tomtom_result_files: 
        try:
            df = pd.read_csv(tomtom_result.joinpath(result_file), sep='\t').dropna()
            df['name'] = result_file.split('.tomtom.tsv')[0] #tomtom result 이름 열 추가
            df_list.append(df)
        except pd.errors.EmptyDataError:
            print('no dataframe')
    if len(df_list) >= 1: 
        DF = pd.concat(df_list).reset_index(drop=True)
        filtered_DF = filtering_df(DF, p_cutoff, q_cutoff)
        print(filtered_DF)
        grouped_DF = filtered_DF.groupby('name').apply(lambda x: x.iloc[0]).reset_index(drop=True)
        print('grouped_DF ', grouped_DF)
        # print(tomtom_result / f'{name}_filtered_motif.csv')
        if filter == False: 
            grouped_DF.to_csv(input_dir.joinpath(f'5.tomtom_result/{name}_motif.csv'), sep='\t', index=False)
        else: 
            grouped_DF.to_csv(input_dir.joinpath(f'5.tomtom_result/{name}_filtered_motif.csv'), sep='\t', index=False)
        return grouped_DF

def process_file(celltype_file, window_size=10, norm=True):
    try:
            # 1) modisco run
        run_modisco(celltype_file, window_size=window_size, norm=norm) # window size = 10
    except Exception as e: 
        # run_modisco(celltype_file, window_size=10)
        print(f'Error occured in {celltype_file} running modisco')
        return
    # 2) modisco report
    run_modisco_report(celltype_file)
    # 3) modisco compare with MEME 
    compare_known_motif(celltype_file)
    # 4) Filtering 
    merge_filter_tomtom_files(celltype_file, 1.0, 1.0, filter=False)
    merge_filter_tomtom_files(celltype_file, 0.05, 0.2, filter=True)

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

