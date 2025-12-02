from captum.attr import Saliency
import torch
import os, sys
import types

work_path='/fsx/home/jhhong/mogam_project/MGC-UTR/motif_explain/JH_Duet/DuET'

sys.path.append(work_path)

from captum.attr import Saliency
from models.module import Module # 수정 필요
from data.datamodule import DataModule # 수정 필요
from configs.config import load_cfgs

import numpy as np
import torch, time
import torch.nn.functional as F
import pandas as pd
from pathlib import Path

### Contents ###
# - Calculate attribution score with the Saliency method from Captum
# - Needs to change the module inside of the UTRmodule (input and predict/forward part)
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # GPU 선정

def custom_get_embedding(self, utr, cds=None):
    """
    utr: (batch, utr_len, 4) or (batch, 4, utr_len)
    cds: (batch, cds_len, 4) or (batch, 4, cds_len)
    """

    # --- UTR ---
    # convert (batch, L, C) → (batch, C, L)
    utr = utr.float().permute(0, 2, 1) # (batch, channel, utr_seq)
    utr = self.utr_block(utr) # (batch, channel, utr_seq)
    
    # --- Dual ---
    if self.dual: 
        cds = cds.float().permute(0,2,1)
        cds = self.cds_block(cds) 
    
    # --- START context? --- 
    # 안 쓰이는 건지 모르겠음 (확인필요)
    
    # --- Layer norm ---
    utr_emb = F.adaptive_avg_pool1d(utr, 1).squeeze(2) # (batch, utr_out_channels)
    if self.do_layernorm:
      utr_emb = self.utr_embed_norm(utr_emb)
    if self.dual:
      cds_emb = F.adaptive_avg_pool1d(cds, 1).squeeze(2) # (batch, cds_out_channels)
      if self.do_layernorm:
        cds_emb = self.cds_embed_norm(cds_emb)
    
      # -- concat ---
      x = torch.cat([utr_emb, cds_emb], dim=-1) # (batch, utr_out_channels+cds_out_channels)
    
    # -- feature 는 사용 X함 --> skip
    return x.detach(), x
    
def custom_predict(self, utr, cds=None):
    """
    utr: one-hot UTR sequence
    cds: one-hot CDS sequence
    """
    x_detach, x = self.custom_get_embedding(utr, cds)
    score = self.dense(x).reshape(-1)

    return score.detach(), score

def forward_for_saliency(self, utr, cds):
    # saliency용으로 forward를 따로 만듦, return은 하나만 하도록
    _, x = self.custom_get_embedding(utr, cds)
    score = self.dense(x).reshape(-1)
    
    return score    

###################################
# From here Get attribution score #
###################################

def get_attribution_score(train_loader, model):
# def get_attribution_score(train_loader, saliency):
    """ 
    Args: 
        train_loader: dict[utr:torch.tensor, cds:torch.tensor. y, metadata]
        
    Return: 
        UTR_list : list of UTR input tensors
        CDS_list : list of CDS input tensors
        UTR_attri_ls : list of UTR attribution scores, norm possible
        CDS_attri_ls : list of CDS attribution scores, norm possible
        y_ls : list of target labels
    """
    # input data
    UTR_list, CDS_list, y_list = [], [], []
    UTR_attri_list, CDS_attri_list = [], []
    saliency = Saliency(model)
    
    start = time.time()
    for i, batch in enumerate(train_loader):
        
        # make tuple for input 
        utr_batch = batch['utr'].cuda().requires_grad_(True)
        cds_batch = batch['cds'].cuda().requires_grad_(True)
        inputs = (utr_batch, cds_batch) # according to saliency format
        if i == 0:
            print(f'# of batch: {i}',
              f'shape of utr & cds input: {utr_batch.shape}, {cds_batch.shape}')
        
        # calculate attribution
        utr_attrib, cds_attrib = saliency.attribute(inputs, target=None, abs=False)
        if i == 0:
            print(f'shape of attri utr & cds: {utr_attrib.shape}, {cds_attrib.shape}')
        
        UTR_list.append(batch['utr'].cpu().numpy())
        CDS_list.append(batch['cds'].cpu().numpy())
        UTR_attri_list.append(utr_attrib.detach().cpu().numpy())
        CDS_attri_list.append(cds_attrib.detach().cpu().numpy())
        y_list.append(batch['y'].cpu().numpy())

        # if i == 1: # for test
        #     break

    y_list = np.concatenate(y_list, axis=0)
    end = time.time()
    spend_time = (end - start)/60
    print(f'done!, time:{spend_time}min')
    return UTR_list, CDS_list, UTR_attri_list, CDS_attri_list, y_list


def save_npz(target_ls, name, norm=False, type='utr5'):
    """Save the numpy array as a .npz file.
    
    - Change the position of nucleotides from A/G/C/T to A/C/G/T for TF-modisco
    - Change the order of dimensions from [seq 개수, 길이, 개수] to [개수, channel, seq 길이] for TF-modisco
    - 2가지 결과 저장 : A,C,G,T 합이 0이 되는 norm & unnorm
    """
    # A/C/G/T 순서로 되어 있어야 되는데 A/G/C/T 순서로 one-hot encoding 되어있음
    # TF modisco에 넣기 위해 순서도 변경 [mRNA 개수, 길이, channel] -> [mRNA 개수, channel, seq 길이]
    target_np = np.concatenate(target_ls, axis=0) # axis=0를 기준으로 list concat
    if type == 'utr5':
        target_np[:,:,[1,2]] = target_np[:,:,[2,1]] #channel 순서 G,C -> C,G로 변경
    print(target_np.shape)
    
    # Reshape the array from [100, 400, 4] to [100, 4, 400]
    reshaped_array = np.transpose(target_np, (0, 2, 1))  # Ensure this matches the input shape
    print(f'Final save shape: {reshaped_array.shape}')
    
    # Normalize each of the channel for TF modisco
    if norm: 
        reshaped_array_norm = np.array([reshaped_array[i,:,:] - np.mean(reshaped_array[i,:,:], axis=0, keepdims=True) \
                                   for i in range(reshaped_array.shape[0])])
        np.savez(f'{name}_norm.npz', reshaped_array_norm)
    np.savez(f'{name}.npz', reshaped_array)


def save_saliency(output_file, ckp_path, utr5_len=500, norm=False):
    """Save the attribution score numpy results
    
    - 1. Load dataset from UTRModule trainloader
    - 2. Load model
    - 3. Calculate attribution score from diff module
    - 4. save npz file - norm on only attribution score
    """
    # attribution 결과 저장 파일 만들기
    os.makedirs(output_file, exist_ok=True)
    
    # make train loader
    config_filepath = [ckp_path.split('/ckpts')[0] + '/config.yaml']  #yaml file load
    cfg, dict_cfg = load_cfgs(config_filepath)
    cfg.datamodule.batch_size = 8
    # load dataset
    train_path = cfg.dataset.train
    datamodule = DataModule(cfg, dataset_path=train_path)
    datamodule.setup(stage='fit')
    train_loader = datamodule.train_dataloader()
    
    # load model with checkpoint
    model = Module.load_from_checkpoint(ckp_path,
                                            cfg=cfg, dict_cfg=dict_cfg,
                                            strict=cfg.load_model_strict,
                                            map_location=torch.device('cpu'))
    model.eval()
    model = model.model.cuda()
    
    # fix model predict and method
    model.custom_get_embedding = types.MethodType(custom_get_embedding, model)
    model.custom_predict = types.MethodType(custom_predict, model)
    model.forward_for_saliency = types.MethodType(forward_for_saliency, model)

    # forward 바꾸기
    model.forward = types.MethodType(forward_for_saliency, model)
    
    UTR_list, CDS_list, UTR_attri_list, CDS_attri_list, y_list = get_attribution_score(train_loader, model)

    save_npz(UTR_list, f'{output_file}/5UTR_{utr5_len}input',  norm=False, type='utr5') # one-hot encoding은 norm X
    save_npz(CDS_list, f'{output_file}/CDS_1500input',  norm=False, type='cds') # one-hot encoding은 norm X

    save_npz(UTR_attri_list, f'{output_file}/5UTR_{utr5_len}attr', norm=norm, type='utr5')
    save_npz(CDS_attri_list, f'{output_file}/CDS_1500attr', norm=norm, type='cds')
    np.savez(f'{output_file}/y_value.npz', y_list)

    
    
if __name__ == "__main__":
    # All_celltype에 해당하는 ckp path
    ckp_filename = 'duet_v2_checkpoints.csv'
    ckp_df = pd.read_csv(ckp_filename)
    utr5_len = 100
    
    # 저장할 output directory
    output_path = '/fsx/s3/project/P240017_mRNA_UTR/motif_explain/analysis_dataset/251201_importance_result'
    for idx, row in ckp_df.iterrows():
        ckp_path = row['checkpoint']
        dir_name = row['cellType'].replace(' ','_').replace('/','_')
        output_file = f'{output_path}/{row["cellType"]}' # 저장 경로 / model 명 / cell type명
        
        print(f'output filename : {output_file}')
        save_saliency(output_file, ckp_path, utr5_len=utr5_len, norm=True)
        
        # if idx == 1:
        #     break