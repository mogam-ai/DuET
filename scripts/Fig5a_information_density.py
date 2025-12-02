#%%
import numpy as np
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from pathlib import Path

from scipy.signal import savgol_filter

result_dir = 'result_csv'
os.makedirs(result_dir, exist_ok=True)


input_path = '/fsx/s3/project/P240017_mRNA_UTR/motif_explain/analysis_dataset/251201_importance_result/'
celltype='All_celltypes'

C2AA = {'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'TCT': 'S', 'TCC': 'S', 'TCA': 'S',
    'TCG': 'S', 'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*', 'TGT': 'C', 'TGC': 'C',
    'TGA': '*', 'TGG': 'W', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L', 'CCT': 'P',
    'CCC': 'P', 'CCA': 'P', 'CCG': 'P', 'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'ATT': 'I', 'ATC': 'I', 'ATA': 'I',
    'ATG': 'M', 'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'AAT': 'N', 'AAC': 'N',
    'AAA': 'K', 'AAG': 'K', 'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R', 'GTT': 'V',
    'GTC': 'V', 'GTA': 'V', 'GTG': 'V', 'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E', 'GGT': 'G', 'GGC': 'G', 'GGA': 'G',
    'GGG': 'G'} # from MGC-D2

def load_numpy_data(file_path: Path, utr5_len=100, cds_len=1500):
    # [1] 데이터 로드
    # - 입력 one-hot, attribution score, prediction 값 로드
    # ----- input data ---
    utr5_input_np = np.load(file_path / f'5UTR_{utr5_len}input.npz')['arr_0']
    cds_input_np = np.load(file_path / f'CDS_{cds_len}input.npz')['arr_0']

    # ----- prediction ---
    y_value = np.load(file_path / 'y_value.npz')['arr_0']
    utr5_pred = y_value
    cds_pred = y_value  

    # --- attribution data ---
    utr5_attr_np = np.load(file_path / f'5UTR_{utr5_len}attr.npz')['arr_0']
    cds_attr_np = np.load(file_path / f'CDS_{cds_len}attr.npz')['arr_0']

    return utr5_input_np, utr5_attr_np, utr5_pred, cds_input_np, cds_attr_np, cds_pred


def non_zero_values(data: np.ndarray):
    """============================================
    [2] Attribution → 각 position 별 zero padding 고려 mean 계산
    ============================================
     
    - 벡터화(vectorized) 계산으로 대폭 속도 향상
    - zero padding 위치를 자동으로 제외하며 mean 계산
    
    input: sum_data_abs (N × L)
    output: (L,)    """

    mask = data != 0                   # padding=0 위치 제외
    sum_vals = (data * mask).sum(axis=0)
    count_vals = mask.sum(axis=0)

    # count=0이면 0으로 처리
    return np.where(count_vals > 0, sum_vals / count_vals, 0)

def change_df(one_hot_encoding_data: np.ndarray, importance_score_matrix: np.ndarray):
    """
     ============================================
     [3] one-hot × attribution → nucleotide contribution → position-wise mean
     ============================================
      ① actual contribution = one-hot × attribution
      ② absolute value
      ③ nucleotide-wise sum → (N × L)
      ④ non-zero padding mean 계산
    """

    # 1. 실제 뉴클레오타이드 기여도 계산
    actual_nucleotide_contributions = importance_score_matrix * one_hot_encoding_data

    # 2. 절댓값
    data_abs = np.abs(actual_nucleotide_contributions)

    # 3. 뉴클레오타이드 채널 합산 (axis=1)
    sum_data_abs = np.sum(data_abs, axis=1)

    # 4. zero padding 고려 mean 계산
    mean_vec = non_zero_values(sum_data_abs)  # shape: (L,)

    # DataFrame 형태로 반환 (원본 코드 유지)
    # 변경 사항: mean_vec은 (L,)이므로 DF로 만들 때 1×L 구조로 변환
    df = pd.DataFrame(mean_vec.reshape(1, -1))

    # --- 원본 코드에서 *100 scaling 적용 이유 불명 → 그대로 유지하되 주석 설명 ---
    df = df * 100  

    return df

def barplot_utr5_cds(utr5_df, cds_df):
   
    plt.figure(figsize=(6, 4))

    # --- 길이 계산 ---
    utr5_len = utr5_df.shape[1]
    cds_len = cds_df.shape[1]

    # --- 데이터 추출 (DataFrame row가 1개라고 가정) ---
    # 변경: for-loop 제거 → row 1개이므로 iloc[0]으로 바로 access
    utr5_values = utr5_df.iloc[0].values
    cds_values = cds_df.iloc[0].values

    # --- barplot for UTR5 ---
    plt.bar(x=utr5_df.columns, height=utr5_values,
        color='#BD3D30', alpha=0.8, width=1.0, label="5'UTR" )

    # --- barplot for CDS ---
    plt.bar(x=cds_df.columns, height=cds_values,
        color='#1E346A', alpha=0.8, width=1.0,label="CDS")

    # --- UTR/CDS 경계를 위한 vertical line ---
    plt.axvline( x=utr5_len, color='black', linestyle='--', linewidth=1.0, alpha=0.7)

    # --- x, y axis 설정 ---
    plt.xlim([0, utr5_len + cds_len])
    plt.ylabel(r'Attribution score × $10^{-2}$', fontsize=11)

    # --- y-axis grid ---
    plt.grid(True, color='lightgray', linestyle='-', linewidth=0.9, alpha=0.7, axis='y' )

    # --- legend ---
    plt.legend(frameon=False, loc='upper right', fontsize=10)

    # --- plot ---
    plt.tight_layout()
    plt.show()
    plt.close()


def calculate_contribution(utr5_df, cds_df):
    """Calculate intensity of information 
    & global intensity of information"""
    # 1. utr5 contribution
    utr5_contribution = utr5_df.values.sum()
    # 2. cds contribution
    cds_contribution = cds_df.values.sum()
    
    # 3. region별 contribution (길이에 대해서 나눈경우)
    region_utr5 = (utr5_contribution*5)/(utr5_contribution*5 + cds_contribution) * 100
    region_cds = (cds_contribution)/(utr5_contribution*5 + cds_contribution) * 100

    # 3. global contribution (전체를 합친 경우)
    global_utr5 = (utr5_contribution / (utr5_contribution + cds_contribution)) * 100
    global_cds = (cds_contribution / (utr5_contribution + cds_contribution)) * 100
    
    # Intensity of utr5 and cds
    return region_utr5, region_cds, global_utr5, global_cds


####### 실행 단계 ######
# 파일 경로 설정
celltype_importance = Path(f"{input_path}/{celltype}")

# [1] 데이터 로드
utr5_input_np, utr5_attr_np, utr5_pred, cds_input_np, cds_attr_np, cds_pred = \
    load_numpy_data(celltype_importance, utr5_len=100, cds_len=1500)

# [2] attribution → information density 변환
utr5_nonpadding_df = change_df(utr5_input_np, utr5_attr_np)
cds_nonpadding_df = change_df(cds_input_np, cds_attr_np)

# [3] column index offset 적용 (UTR 길이만큼 shift)
cds_nonpadding_df.columns = np.arange(
    utr5_nonpadding_df.shape[1], 
    utr5_nonpadding_df.shape[1] + cds_nonpadding_df.shape[1]
)
# Plotting
barplot_utr5_cds(utr5_nonpadding_df, cds_nonpadding_df)
# %%

def count_number_nonzero_input(input_np: np.ndarray) -> np.ndarray:
    """
    변경: numpy-only. 각 position(L)에 대해 실제 존재하는 mRNA 개수 계산.
    input:  (N,4,L)
    return: (L,)"""
    summed = np.sum(input_np, axis=1)          # (N,L)
    return np.sum(summed != 0, axis=0)         # (L,)


def zero_padding_mean(attr_np: np.ndarray, input_np: np.ndarray) -> pd.DataFrame:
    """
    변경: 완전 벡터화 버전.
    channel × position 평균 계산. shape → (4,L)"""

    real_contribution = attr_np * input_np # (N,4,L)
    count_nonzero = count_number_nonzero_input(input_np).astype(float)  # (L,)
    count_nonzero[count_nonzero == 0] = 1   # division zero 방지

    # sum across N → shape (4,L)
    summed = np.sum(real_contribution, axis=0)

    # channel × L mean
    mean_channel_pos = summed / count_nonzero

    return pd.DataFrame(mean_channel_pos)


def _debug_position_analysis(contributions, position):
    """특정 위치(A channel) 기여도 분포 점검용"""
    vals = contributions[:, 0, position]
    print(f"[Debug] pos={position} mean={vals.mean():.6f} (+){np.sum(vals>0)} (-){np.sum(vals<0)}")


def each_level_change_df(utr5_input_np, utr5_attr_np, cds_input_np, cds_attr_np, debug_position=None):
    """UTR/CDS 각각 zero-padding 보정 mean 계산"""
    utr_df = zero_padding_mean(utr5_attr_np, utr5_input_np)
    cds_df = zero_padding_mean(cds_attr_np, cds_input_np)

    if debug_position is not None:
        _debug_position_analysis(utr5_attr_np, debug_position)

    return utr_df, cds_df

#%%
# ================================================================
# [2] UTR Line plot
# ================================================================

def utr_lineplot(utr_mean: pd.DataFrame,  smooth=True, x_range=True):
    """
    utr_mean shape: (4, L)
    index: ['A','C','G','U'] 로 설정되어야 함
    """

    nucleotide = ['A', 'C', 'G', 'U']
    cmap = ['#FFC464', '#D7C464', '#D66E64', '#CB181D']

    df = utr_mean.copy()

    if smooth:
        df = df.apply(lambda x: savgol_filter(x, window_length=4, polyorder=2),
                      axis=1, result_type='expand')

    plt.figure(figsize=(3, 3))

    for i, nuc in enumerate(nucleotide):
        y_vals = df.loc[nuc].values
        x_vals = df.columns
        plt.plot(
            x_vals, y_vals,
            marker='o', linestyle='-', linewidth=1.3, markersize=0.1,
            color=cmap[i], label=nuc
        )

    plt.axhline(y=0, color='black', linestyle='--', linewidth=1.2)
    plt.ylabel("Attribution score", fontsize=9)
    plt.tick_params(axis='y', labelsize=8)
    plt.tick_params(axis='x', labelsize=8)

    if x_range:
        L = df.shape[1]
        plt.xlim(0, L)
        plt.xticks([0, L])

    ymin, ymax = df.values.min(), df.values.max()
    plt.ylim(ymin - 0.0003, ymax + 0.0005)

    plt.legend(frameon=False, loc='lower left', fontsize=8)
    plt.grid(True, axis='y', linestyle='-', linewidth=0.9, alpha=0.6)
    plt.show()
    plt.close()


# ================================================================
# [3] Codon ranking + labeling
# ================================================================

def get_worst_best(cds_mean, C2AA):
    """
    cds_mean index = codons ('AAA','AAC',...)
    codon importance sum 계산 후 top3 / bottom3 구분
    """
    cds = cds_mean.copy()
    codons = list(C2AA.keys())

    sum_dict = {codon: cds.loc[codon].sum() for codon in codons}

    sorted_items = sorted(sum_dict.items(), key=lambda x: x[1], reverse=True)

    group_map = {}
    for i, (codon, _) in enumerate(sorted_items):
        if i < 3:
            group_map[codon] = "top"
        elif i >= len(sorted_items) - 3:
            group_map[codon] = "bottom"
        else:
            group_map[codon] = "middle"

    cds['group'] = cds.index.map(group_map)
    cds['sum'] = cds.index.map(sum_dict)

    return cds


# ================================================================
# [4] CDS lineplot
# ================================================================

def cds_lineplot(df, C2AA, smooth=True):
    """
    df: codon × position + [group, sum]
    """

    sorted_df = df.sort_values(by="sum", ascending=False).drop(['group', 'sum'], axis=1)

    if smooth:
        sorted_df = sorted_df.apply(lambda x: savgol_filter(x, 11, 2),
                                    axis=1, result_type='expand')

    plt.figure(figsize=(6, 4))
    cmap = {0:'#0D47A1', 1:'#1976D2', 2:'skyblue',
            len(sorted_df)-3:'#4A5BA8', len(sorted_df)-2:'#6B78D4', len(sorted_df)-1:'#8C95FF'}

    cmap.update({i:'lightgray' for i in range(3, len(sorted_df)-3)})

    for i, (codon, row) in enumerate(sorted_df.iterrows()):
        color = cmap.get(i, 'lightgray')
        label = None
        if i < 3 or i >= len(sorted_df)-3:
            label = f"{codon.replace('T','U')} ({C2AA[codon]})"

        plt.plot(
            sorted_df.columns,
            row.values,
            linewidth=1.2 if label else 0.8,
            color=color,
            label=label
        )
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, ymax * 1.3)  # 위로 20% 여유 공간 추가
    plt.xlim(0, 500)  # x축 최소값을 0으로, 최대값은 자동
    plt.legend(frameon=False, fontsize=7, loc='upper right')
    plt.ylabel("Attribution score")
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1.2)
    plt.grid(True, axis='y', linestyle='-', alpha=0.5)
    plt.show()
    plt.close()


# ================================================================
# [5] 실행 파이프라인
# ================================================================

# 1) UTR/CDS 파싱
utr5_mean, cds_mean = each_level_change_df(utr5_input_np, utr5_attr_np,
                                           cds_input_np, cds_attr_np,
                                           debug_position=50)

# 2) index 설정
utr5_mean.index = ['A', 'C', 'G', 'U']
cds_mean.index = list(C2AA.keys())

# 3) 저장
utr5_mean.to_csv(f"{result_dir}/Fig5B.Nucleotide_level.csv")

# 4) UTR 플롯
utr_lineplot(utr5_mean, smooth=False)
utr_lineplot(utr5_mean.iloc[:, -30:], smooth=False, x_range=False)

# 5) CDS worst/best 코돈 태깅
cds_grouped = get_worst_best(cds_mean, C2AA)
cds_grouped.to_csv(f"{result_dir}/Fig5B.cds_level.csv")

# 6) CDS 플롯
cds_lineplot(cds_grouped, C2AA)

#%%