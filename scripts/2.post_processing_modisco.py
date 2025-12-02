#%% 
# modisco를 run하기 위한 목적
import numpy as np
import subprocess as sp
import h5py, os, sys
from IPython.display import HTML
from pathlib import Path

import pandas as pd
import multiprocess as mp

# must run with conda activate MTtrans due to revised code
# modisco에서 나오는 중간 파일들 
project_path = Path('/fsx/home/jhhong/2024_project/3.RNA_UTR/results_attribution_score')
input_dir = project_path.joinpath('saliency_dedup_UTR5_500')
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


def run_modisco(name, window_size = 7):
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
    input = np.load(f'{input_dir}/{name}/5UTR_500input.npz')['arr_0']
    attr = np.load(f'{input_dir}/{name}/5UTR_500attr.npz')['arr_0']
    print(f'Shape of input: {input.shape}')
    # print(f'Shape of input: {attr.shape}')
    input_has_nan = np.isnan(input).any() # check matrix에 nan이 있는지
    attr_has_nan = np.isnan(attr).any()
    # print(f"NaN values in input data: {input_has_nan}")
    # print(f"NaN values in attr data: {attr_has_nan}")
    
    # cmd = f'modisco motifs -s {input_dir}/{name}/5UTR_500input.npz -a {input_dir}/{name}/5UTR_500attr.npz -z {window_size} -n 4000 \
    # -o {modisco_result}/{name}_5UTR.h5 -w 500 -v'
    # cmd = f'modisco motifs -s {input_dir}/{name}/5UTR_500input.npz -a {input_dir}/{name}/5UTR_500attr.npz \
    # -o {modisco_result}/{name}_5UTR.h5 -n 4000 -w 100 -z 7 -t 10 -f 10 -g 5 -v'
    cmd = f'modisco motifs -s {input_dir}/{name}/5UTR_500input.npz -a {input_dir}/{name}/5UTR_500attr.npz -z {window_size} -n 4000\
    -o {modisco_result}/{name}_5UTR.h5 -w 400'
    
    """modisco run option
    -n (--max_seqlets): metacluster당 최대 seqlet 개수
    -l (--n_leiden): 수행할 Leiden clustering 횟수
    -w (--window): 피크 중심을 둘러싼 영역으로, motif 발견을 위해 고려할 window 크기
    """
    result = sp.run(cmd, shell=True, capture_output=True)
    # print(result.stderr)
    read_modisco_result(name)

def read_modisco_result(filename):
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
            grouped_DF.to_csv(project_path.joinpath(f'5.tomtom_result/{name}_motif.csv'), sep='\t', index=False)
        else: 
            grouped_DF.to_csv(project_path.joinpath(f'5.tomtom_result/{name}_filtered_motif.csv'), sep='\t', index=False)
        return grouped_DF

def process_file(celltype_file):
    try:
            # 1) modisco run
        run_modisco(celltype_file, window_size=10) # window size = 10
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


def run_sbatch():
    sbatch_script = """
    #!/bin/bash
    #SBATCH --job-name=modisco_job
    #SBATCH --output=modisco_output.log
    #SBATCH --error=modisco_error.log
    #SBATCH --time=01:00:00
    #SBATCH --partition=standard
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=8G

    module load python/3.8
    python script/2.run_modisco.py
    """
    with open("sbatch_modisco.sh", "w") as f:
        f.write(sbatch_script)
    sp.run("sbatch sbatch_modisco.sh", shell=True)


#%% 1) Get pattern from modisco
if __name__ == "__main__":
    celltype_files = os.listdir(input_dir)
    done_files = os.listdir(meme_result)
    if len(done_files) >= 1:
        done_files = [file.replace('_5UTR','') for file in done_files]
    
        left_files = list(set(celltype_files) - set(done_files))
        print(f'# of left files: {len(left_files)}')
        print(left_files)
        with mp.Pool(processes=1) as pool: 
            pool.map(process_file, left_files)
    else:
        with mp.Pool(processes=1) as pool: 
            pool.map(process_file, celltype_files)
   
sys.exit(1)
#%% 2) modisco result 
read_modisco_result(celltype_file)
#%% 3) modisco report
celltype_file = 'A2780'
run_modisco_report(celltype_file)
#%% 4) modisco compare with MEME 
compare_known_motif(celltype_file)
# 5) Filtering 
grouped_DF = merge_filter_tomtom_files(celltype_file, 0.05, 0.2)