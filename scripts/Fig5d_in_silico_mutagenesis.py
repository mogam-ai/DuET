#%%
import os, sys
work_path='/fsx/home/jhhong/mogam_project/MGC-UTR/motif_explain/JH_Duet/DuET'
sys.path.append(work_path)

from data.duet import DuetDataset # data load
from models.module import Module # model load
from torch.utils.data import DataLoader
from configs.config import load_cfgs
import pandas as pd
import numpy as np
import torch

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

outdir = 'mutagensis'
"""
in silico mutagenesis analysis
HAMP, SPY의 경우 UTR5 100nt으로 학습시킨 것이 결과가 잘나옴
나머지는 UTR5 500nt으로 학습시킨 것이 더 잘나오는 경향이 있음
"""
# load "All celltype" model ckp
ckp_file = pd.read_csv('duet_v2_checkpoints_500+1500.csv')

# checkpont
# ckp_file = pd.read_csv('duet_v2_checkpoints.csv')
all_celltype_ckp = ckp_file[ckp_file['cellType']=='All_celltypes']['checkpoint'].iloc[0]
meta_path = '/fsx/s3/project/P240017_mRNA_UTR/data/Ribo-seq/mappings/GENCODE_v47/gencode.v47.map.all_tx.fullseq.total_var_metadata_v3.tsv'
############################################
# 1) Mutagenesis (UTR5 한 자리씩 A/C/G/T 변이)
############################################
def mutate_seq(utr5_seq):
    bases = ['A','C','G','T']
    mutated_seqs = []
    mutated_bases = []

    for i, base in enumerate(utr5_seq):
        for b in bases:
            mutated_bases.append(b)
            if b == base:
                mutated_seqs.append(utr5_seq)
            else:
                mutated_seqs.append(utr5_seq[:i] + b + utr5_seq[i+1:])
    return mutated_seqs, mutated_bases


###############################################
# 2) tmp CSV 생성 (predictiion을 위한 dataset 생성)
###############################################
def make_tmp_file(utr5_seq, cds_seq, utr3_seq, gene, mutated_seqs):
    df = pd.DataFrame({
        "txID": ["ENST0"] * len(mutated_seqs),
        "utr5": mutated_seqs,
        "cds": [cds_seq] * len(mutated_seqs),
        "utr3": [utr3_seq] * len(mutated_seqs),
        "full_seq": [utr5_seq + cds_seq + utr3_seq] * len(mutated_seqs),
        "te": [0] * len(mutated_seqs)
    })
    tmp_path = f"{outdir}/tmp_{gene}.csv"
    df.to_csv(tmp_path, sep="\t", index=False)
    return df, tmp_path


############################################
# 3) 모델로 TE prediction 수행
############################################
def predict_te(tmp_csv_path, utr5_seq, mutated_seqs, mutated_bases):

    config_list = [f'{all_celltype_ckp.split("/ckpts/")[0]}/config.yaml']
    cfg, dict_cfg = load_cfgs(config_list)
    cfg.val_random_split = False
    cfg.do_kfold_test = False
    cfg.batch_size = 1
    cfg.num_workers = 0

    # datamodule 대신에 dataset 자체를 로드
    dataset = DuetDataset(data_path=tmp_csv_path,
                          **cfg.dataset.param)
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = Module.load_from_checkpoint(
        all_celltype_ckp,
        cfg=cfg, dict_cfg=dict_cfg,
        strict=cfg.load_model_strict,
        map_location=torch.device('cuda')
    )
    model.eval().cuda()

    y_preds = []
    for batch in data_loader:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pred_detach, pred_raw = model.model.predict(batch)
        y_preds.append(float(pred_detach.squeeze()))

    df = pd.read_csv(tmp_csv_path, sep="\t")
    df["predict_te"] = y_preds

    wt_te = df[df.utr5 == utr5_seq]["predict_te"].iloc[0]
    df["diff_te"] = df["predict_te"] - wt_te

    df["mut_base"] = mutated_bases
    return df, wt_te


############################################
# 4) Pivot table (A,C,G,T × position)
############################################
def pivot_df(df, utr5_len, gene):
    df["pos"] = np.repeat(np.arange(utr5_len), 4)

    pivot = df.pivot_table(
        values="diff_te",
        index="mut_base",
        columns="pos",
        aggfunc="first"
    ).loc[["A","C","G","T"]]

    outpath = f"{outdir}/{gene}_predict_pivot.csv"
    pivot.to_csv(outpath)
    return pivot


############################################
# 5) 전체 실행 함수
############################################
def run_mutagenesis(utr5_seq, cds_seq, utr3_seq, gene):

    print("🧬 Step 1: generating mutations...")
    mutated_seqs, mutated_bases = mutate_seq(utr5_seq)

    print("📝 Step 2: making tmp input file...")
    df_tmp, tmp_path = make_tmp_file(utr5_seq, cds_seq, utr3_seq, gene, mutated_seqs)

    print("🔮 Step 3: running TE prediction...")
    df_pred, wt_te = predict_te(tmp_path, utr5_seq, mutated_seqs, mutated_bases)

    print("📊 Step 4: building pivot mutation matrix...")
    pivot = pivot_df(df_pred, len(utr5_seq), gene)

    print("✔️ Completed!")
    print(f"WT TE = {wt_te}")

    return df_pred, pivot

def combined_plot(pivot_df, gene, mut_pos, mut_labels, figsize=(20, 6)):
    """
    Heatmap과 Lineplot을 하나의 Figure에 그리는 함수.
    ax1은 Heatmap, ax2은 Lineplot에 해당
    뒤에서부터 100nt까지만 dataframe 가져오기
    """
    # Heatmap
    fig = plt.figure(figsize=figsize)  # 전체 Figure 크기 설정
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0)  # 두 개의 subplot 생성
    
    # Index T -> U로 변경
    pivot_df.index = pivot_df.index.str.replace('T', 'U')
    # pivot_df = pivot_df.iloc[:,-100:]
    # Heatmap 생성 (ax1)
    ax1 = fig.add_subplot(gs[1])  # 첫 번째 subplot
    
    colors = ["#0431AB", "#F7F7F7", "#BD3D30"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    heatmap = sns.heatmap(pivot_df, cmap=custom_cmap, ax=ax1, cbar=False, center=0, annot=False) # , square=True) #, linewidths=0.1, linecolor='black')  #, square=True,)
    
    # 컬러바 추가
    cbar = fig.colorbar(heatmap.collections[0], ax=ax1, orientation='vertical' , pad=0.0, fraction=0)
    cbar.ax.set_position([0.91, 0.10, 0.02, 0.37])  # [x, y, width, height]로 위치 지정
    
    # Heatmap 축 설정
    ax1.set_yticklabels(pivot_df.index, rotation=0, ha='center', size=11)
    ax1.set_ylabel(r'$\Delta$ TE', rotation=-90, fontsize=10)
    ax1.yaxis.set_label_coords(1.05, 0.5)  # 오른쪽에서 x=1.08, 중앙 y=0.5 위치
    ax1.set_xticks([len(pivot_df.columns)+mut_pos+0.5])  # x축 위치 변경
    
    ax1.set_xticklabels([mut_labels], rotation=0, fontsize=10)  # x축 레이블 설정
    
    # ylabel ticks 제거하기
    plt.tick_params(axis='y', which='both', left=False, right=False)
    
    # Lineplot 생성
    ax2 = fig.add_subplot(gs[0])  # 두 번째 subplot
    
    # Pos/neg dataframe 생성
    pos_df = pivot_df.applymap(lambda x: x if x > 0 else 0)
    neg_df = pivot_df.applymap(lambda x: x if x < 0 else 0)
    pos_df.loc['sum'] = pos_df.sum(axis=0)
    neg_df.loc['sum'] = -neg_df.sum(axis=0)
    
    x_values = range(len(pivot_df.columns))
    ax2.plot(x_values, pos_df.loc['sum'].values, linestyle='-', color='#BD3D30', linewidth=1.4, markersize=1, label='Gain'+r'$\Delta$ TE' ) # label=r'Positive $\Delta$ TE')
    ax2.plot(x_values, neg_df.loc['sum'].values, linestyle='-', color='#1E346A', linewidth=1.4, markersize=1, label='Loss'+r'$\Delta$ TE' ) #r'Negative $\Delta$ TE')
    
    # Lineplot 축 설정
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_xlim(0, len(pivot_df.columns) - 1)
    ax2.tick_params(axis='y', labelsize=5)
    ax2.set_xticks([])
    ax2.legend(fontsize=8, frameon=False, loc='upper right', bbox_to_anchor=(1.0, 1.05))
    ax2.grid(False)
    
    plt.title(gene, fontsize=14, fontweight='bold')

    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    # plt.savefig(f'./plots/SPY_ISM.png', dpi=500)
    plt.show()
    plt.close()

def choose_gene(i):
    """metadata에서 적합한 유전자 찾기
    유전자, 유전자 위치, 변이 정보"""
    
    meta = pd.read_csv(meta_path, sep='\t')
    gene = ['SRY','IRF6', 'HAMP','GCH1','KCNJ11','PEX7','PRKAR1A','SPINK1','HBB','TWIST1','CFTR','HR','CDKN2A','ENG','GJB1','SHOX'] # IRF6/HAMP/GCH1 X
    mut_pos = [-75, -48, -25, -22, -54, -45, -97, -53, -29, -18, -34, -321, -34, -127, -103, -19]
            # SRY(0) IRF6(1) HAMP(2) GCH1(3) KCNJ11(4) PEX(5) PRKAR1A(6) SPINK1(7) HBB(8) TWIST1(9) CFTR(10) HR(11) CDKN2A(12)  ENG(13) GJB1(14) SHDX(16)
    
    enst_idx = [0,     0,       0,      2,        1,     1,      6,          0,      0,      0,      -1,      0,      0, 10, -1, -2]
    print(f'개수: {len(gene), len(mut_pos), len(enst_idx)}')
    mut_labels = ['-75G>A','-48A>U', '-25G>A','-22C>U','-54C>U','-45C>U','-97G>A','-53C>U','-29G>A','-18C>U','-34C>U','-321A>G','-34G>U','-127C>U','-103C>T','-19G>A']

    all_gene = meta[meta['geneName']==gene[i]]
    print(all_gene)
    if len(all_gene)>= 2:
        all_gene = all_gene.iloc[[enst_idx[i]]]
        print(all_gene[['txID','geneName']])
    gene_utr5_seq = all_gene['utr5'].values[0]
    gene_cds_seq = all_gene['cds'].values[0]
    gene_utr3_seq = all_gene['utr3'].values[0]
    # print(SRY)
    print(f'utr5 len: {len(gene_utr5_seq)}')
    return gene_utr5_seq, gene_cds_seq, gene_utr3_seq, gene[i], mut_pos[i], mut_labels[i]

if __name__ == "__main__":

    i = 7 # SRY 가능, IRF6 가능(utr5_500까지 연장), HAMP 가능(utr5_100), GCH1 X, KCNJ11 X, PRKAR1A O, 
    gene_utr5_seq, gene_cds_seq, gene_utr3_seq, gene, mut_pos, mut_labels \
         = choose_gene(i)
    
    df_pred, df_pivot = run_mutagenesis(
        utr5_seq=gene_utr5_seq,
        cds_seq=gene_cds_seq,
        utr3_seq=gene_utr3_seq,
        gene=gene
    )
    
    combined_plot(df_pivot, gene, mut_pos, mut_labels, figsize=(15, 2.2))

    mutation_info = [0]*len(df_pivot.columns)
    mutation_info[len(df_pivot.columns)+mut_pos] = 1
    new_row = pd.Series(mutation_info, index=df_pivot.columns, name="Mutation")
    df_pivot2 = pd.concat([df_pivot, new_row.to_frame().T], axis=0)
    df_pivot2.index = list(df_pivot.index) + ['Mutation']
    # print(df_pivot2)
    df_pivot2.to_csv(f'result_csv/Fig5D.{gene}_in_silico_mutagenesis.csv')

#%% 