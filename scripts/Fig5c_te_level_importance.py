#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

#[1] LOAD ATTRIBUTION SCORE
input_path = '/fsx/s3/project/P240017_mRNA_UTR/motif_explain/analysis_dataset/saliency_wo_dedup_UTR5_100'
celltype='All_celltype'
celltype_importance = Path(f'{input_path}/{celltype}')

def load_numpy_data(file_path: Path, utr5_len: int):
    """
    Load saved numpy arrays for UTR5, CDS, and TE values.
    Automatically matches utr5_len for both input and attribution files.
    """
    utr5_input = np.load(file_path / f'5UTR_{utr5_len}input.npz')['arr_0']
    cds_input = np.load(file_path / 'CDS_1500input.npz')['arr_0']

    utr5_attr = np.load(file_path / f'5UTR_{utr5_len}attr.npz')['arr_0']
    cds_attr = np.load(file_path / 'CDS_1500attr.npz')['arr_0']

    te = np.load(file_path / 'y_value.npz')['arr_0']  # TE 값은 하나만 존재

    return utr5_input, utr5_attr, cds_input, cds_attr, te

def non_zero_mean_vectorized(matrix: np.ndarray):
    """ 
    #2. non-zero mean per position (vector)
    matrix: shape (N_sequences, positions)
    return: shape (positions,)
    """
    # 0이 아닌 값만 선택 (NaN으로 masking 후 nanmean)
    masked = np.where(matrix != 0, matrix, np.nan)
    return np.nanmean(masked, axis=0)

def dataframe_custom_qcut_split(one_hot_data: np.ndarray,
                                importance_matrix: np.ndarray,
                                te_values: np.ndarray,
                                quantiles=[0,0.2,0.4,0.6,0.8,1.0],
                                labels=['Bottom','Low','Medium','High','Top']):
    """##################################
    # 3.Compute TE-grouped importance#
    ##################################"""

    # Contribution 계산 (abs 필요)
    contributions = np.abs(importance_matrix * one_hot_data)   # (N, 4, L)

    # 채널 방향 합산 → (N, L)
    contrib_sum = contributions.sum(axis=1)

    # DataFrame 구성
    df = pd.DataFrame(contrib_sum)
    df['TE'] = te_values

    # TE 높은 순 정렬
    df = df.sort_values('TE', ascending=False).reset_index(drop=True)

    # 그룹 할당
    df['group'] = pd.qcut(df['TE'], q=quantiles, labels=labels)

    # 그룹별 non-zero mean 계산
    result = {}
    for label in labels:
        group_rows = df[df['group'] == label].iloc[:, :-2]   # TE, group 제외
        result[label] = non_zero_mean_vectorized(group_rows.values)

    return pd.DataFrame(result)

def compute_and_save_TE_groups(file_path, utr5_len=100, output_prefix='result_csv/Fig5C'):
    file_path = Path(file_path)

    # Load data
    utr5_input, utr5_attr, cds_input, cds_attr, te = load_numpy_data(file_path, utr5_len)

    # Quantile 정의
    quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels    = ['Bottom(20%)','Low(20%)','Medium(20%)','High(20%)','Top(20%)']

    # Compute grouped contribution
    utr5_group_df = dataframe_custom_qcut_split(
        utr5_input, utr5_attr, te, quantiles, labels
    )

    cds_group_df = dataframe_custom_qcut_split(
        cds_input, cds_attr, te, quantiles, labels
    )

    # Save results
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    utr5_group_df.to_csv(f'{output_prefix}.UTR5_group_TE.csv')
    cds_group_df.to_csv(f'{output_prefix}.CDS_group_TE.csv')

    print("5'UTR grouped result head:")
    print(utr5_group_df.head())
    print("\nCDS grouped result head:")
    print(cds_group_df.head())

    return utr5_group_df, cds_group_df

def heatmap_and_lineplot(df_TE, seq_type, te_levels, figsize=(10, 6), 
                         xticks_size=50, label_fontsize=9):

    """ Heatmap 
    df_TE: DataFrame (positions × TE groups)
    te_levels: plotting 순서 (list of column names)
    """

    positions = df_TE.index
    group_num = len(te_levels)

    # Heatmap color range
    vmin, vmax = df_TE.values.min(), df_TE.values.max()

    # Custom colormap (Blue - White - Red)
    colors = ['#0431AB', '#F7F7F7', '#BD3D30']
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    # === Figure Layout ===
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        group_num + 1, 1, height_ratios=[0.18] + [1] * group_num, hspace=0.15)

    # === Colorbar Axis ===
    cbar_ax = fig.add_axes([0.15, 0.87, 0.70, 0.02])

    # === Main Axes ===
    axes = [fig.add_subplot(gs[i+1]) for i in range(group_num)]

    for idx, ax in enumerate(axes):
        level = te_levels[idx]
        values = df_TE[level].values.reshape(1, -1)

        # --- Heatmap ---
        sns.heatmap( values, ax=ax, cmap=custom_cmap,
            vmin=vmin, vmax=vmax,  cbar=(idx == 0),
            cbar_ax=(cbar_ax if idx == 0 else None),
            cbar_kws={'location': 'top', 'orientation': 'horizontal', 'pad': 0.1},
            xticklabels=(positions if idx == group_num - 1 else False),
            yticklabels=[]
        )

        # --- Annotation (TE level label on right) ---
        ax.text(
            df_TE.shape[0] - 1, 
            0.1,              # Heatmap 위쪽으로 label 배치
            level,
            fontsize=label_fontsize,
            ha="right",
            va="top",
            fontweight="bold"
        )
 
        # --- Lineplot ---
        ax2 = ax.twinx()
        margin = (vmax - vmin) * 0.25 if vmax != vmin else 0.001
        ax2.set_ylim(vmin - margin, vmax + margin)
        ax2.plot(positions, df_TE[level].values, color="black", linewidth=1.5)

        # clean styling
        ax2.spines['right'].set_linewidth(1)
        ax2.spines['right'].set_color('black')
        ax2.tick_params(axis='y', labelsize=7)
        ax.grid(False)
        ax2.grid(False)

    # === X-axis ticks formatting ===
    final_ax = axes[-1]
    tick_positions = list(range(0, len(positions), xticks_size))
    final_ax.set_xticks(tick_positions)
    final_ax.set_xticklabels([str(x) for x in tick_positions], fontsize=8)

    plt.subplots_adjust(left=0.15, right=0.90)
    plt.show()


group_utr5, group_cds = compute_and_save_TE_groups(celltype_importance ,utr5_len=100, output_prefix='result_csv/Fig5C')
te_levels = ['Bottom(20%)','Low(20%)','Medium(20%)','High(20%)','Top(20%)']

heatmap_and_lineplot(group_utr5, '5UTR', te_levels, figsize=(6,5), xticks_size=20)
heatmap_and_lineplot(group_cds, 'CDS', te_levels, figsize=(12,5), xticks_size=100)

# %%
