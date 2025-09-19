# Program:     DuET v1.0.0
# Author:      Sungho Lee, Jae-Won Lee, Jinhee Hong
# Affiliation: MOGAM Institute for Biomedical Research
# Contact:     sungho.lee@mogam.re.kr; jaewon.lee@mogam.re.kr; jinhee.hong@mogam.re.kr
# Citation:    TBD


import sys
sys.path.append("..")
import os
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
import numpy as np
import pandas as pd

from captum.attr import Saliency
from pathlib import Path
from tqdm import tqdm

from models.module import Module
from data.datamodule import DataModule
from configs.config import load_cfgs


os.environ["CUDA_VISIBLE_DEVICES"] = "7"


class SaliencyAdapter:
    """A wrapper class to adapt the model for Captum"s Saliency method."""
    def __init__(self, model):
        self.model = model
        
    def __call__(self, utr, cds):
        batch = {"utr": utr, "cds": cds}
        _, score = self.model.predict(batch)
        return score


def get_attribution_scores(train_loader, model):
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
    utr_tensors, cds_tensors, labels = [], [], []
    utr_attr_scores, cds_attr_scores = [], []
    saliency = Saliency(model)
    
    for batch in tqdm(train_loader):
        utr = batch["utr"].cuda().requires_grad_(True)
        cds = batch["cds"].cuda().requires_grad_(True)
        inputs = (utr, cds)

        utr_attr_score, cds_attr_score = saliency.attribute(inputs, target=None, abs=False)
        
        utr_tensors.append(batch["utr"].cpu().numpy())
        cds_tensors.append(batch["cds"].cpu().numpy())
        utr_attr_scores.append(utr_attr_score.detach().cpu().numpy())
        cds_attr_scores.append(cds_attr_score.detach().cpu().numpy())
        labels.append(batch["y"].cpu().numpy())

    labels = np.concatenate(labels, axis=0)

    return utr_tensors, cds_tensors, utr_attr_scores, cds_attr_scores, labels


def save_npz(data, fname, normalize=False, type="utr5"):
    """Save the numpy array as a .npz file."""
    # 5" UTR: Set channel order to A,C,G,T from A,G,C,T
    # [num_samples, seq_length, num_channels] -> [num_samples, num_channels, seq_length]
    target_np = np.concatenate(data, axis=0)
    
    if type == "utr5":
        target_np[:,:,[1,2]] = target_np[:,:,[2,1]] #channel 순서 G,C -> C,G로 변경
    
    reshaped_array = np.transpose(target_np, (0, 2, 1))
    
    if normalize: 
        reshaped_array_norm = np.array([reshaped_array[i,:,:] - np.mean(reshaped_array[i,:,:], 
                                                                        axis=0, keepdims=True) \
                                   for i in range(reshaped_array.shape[0])])
        np.savez(f"{fname}_norm.npz", reshaped_array_norm)
        
    np.savez(f"{fname}.npz", reshaped_array)


def save_saliency(output_dir, ckp_path, utr5_len=100, normalize=True):
    """Save the attribution scores and input tensors as numpy npz format."""

    os.makedirs(output_dir, exist_ok=True)

    config_filepath = [ckp_path.split("/ckpts")[0] + "/config.yaml"]  #yaml file load
    cfg, dict_cfg = load_cfgs(config_filepath, 
                              compat_xref={"dataset.param.use_metadata": "dataset.param.use_sequence_feature",
                                           "dataset.param.metadata_path": "dataset.param.sequence_feature_path",
                                           "dataset.param.metadata_cols": "dataset.param.sequence_feature_cols",
                                           "dataset.param.use_triplet_phase": None,
                                           "dataset.param.utr_overlap_size": None,
                                           "dataset.param.cds_overlap_size": None,
                                           "model.param.global_block": None,})
    cfg.datamodule.batch_size = 8
    cfg.model.name = "duet"
    cfg.dataset.name = "duet"
    # load dataset
    train_path = cfg.dataset.train
    datamodule = DataModule(cfg, dataset_path=train_path)
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    
    # load model with checkpoint
    model = Module.load_from_checkpoint(ckp_path,
                                        cfg=cfg, dict_cfg=dict_cfg,
                                        strict=cfg.load_model_strict,
                                        map_location=torch.device("cpu"))
    model.eval()
    model = SaliencyAdapter(model.model.cuda())

    utr_tensors, cds_tensors, utr_attr_scores, cds_attr_scores, labels = get_attribution_scores(train_loader, model)

    save_npz(utr_tensors, f"{output_dir}/utr5_{utr5_len}_input", type="utr5")
    save_npz(cds_tensors, f"{output_dir}/cds_1500_input", type="cds")

    save_npz(utr_attr_scores, f"{output_dir}/utr5_{utr5_len}_attr", normalize=normalize, type="utr5")
    save_npz(cds_attr_scores, f"{output_dir}/cds_1500_attr", normalize=normalize, type="cds")

    np.savez(f"{output_dir}/y_value.npz", labels)


if __name__ == "__main__":
    sheet_model_dict = {
        "250529_layernorm_100+1500": "saliency_dedup_100+1500",
        "250530_layernorm_500+1500": "saliency_dedup_500+1500",
        "250825_wo_dedup_500+1500": "saliency_nodedup_500+1500", 
        "250825_wo_dedup_100+1500": "saliency_nodedup_100+1500"
    }    
    ckp_path = Path("DuET_ckp_file_path.xlsx")
    DuET_ckp_xlsx = pd.ExcelFile(ckp_path, engine="openpyxl")
    utr5_len = 100
    
    for sheet_name, dir_model_name in sheet_model_dict.items():
        if sheet_name != "250825_wo_dedup_100+1500":
            continue
        DuET_ckp_file = DuET_ckp_xlsx.parse(sheet_name)
        output_path = "attribution_score_results"
        max_iter = len(DuET_ckp_file)
        for idx, row in DuET_ckp_file.iterrows():
            print(f"Calculating saliency for {row['cellType']}... ({idx+1}/{max_iter})")
            ckp_path = row["checkpoint"]
            dir_name = row["cellType"].replace(" ","_").replace("/","_")
            output_dir = f"{output_path}/{dir_model_name}/{dir_name}"
            if os.path.isdir(output_dir):
                print(f"Skipping {row['cellType']} as the output directory already exists.")
                continue
            save_saliency(output_dir, ckp_path, utr5_len=utr5_len, normalize=True)

