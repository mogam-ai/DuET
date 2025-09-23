#!/usr/bin/env python3
# Program:     DuET v1.0.0
# Author:      Sungho Lee, Jae-Won Lee, Jinhee Hong
# Affiliation: MOGAM Institute for Biomedical Research
# Contact:     sungho.lee@mogam.re.kr; jaewon.lee@mogam.re.kr; jinhee.hong@mogam.re.kr
# Citation:    TBD


import os
import argparse
from argparse import RawTextHelpFormatter


print("""
                                        
 ▄▄▄▄▄               ▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄ 
 ██▀▀▀██             ██▀▀▀▀▀▀  ▀▀▀██▀▀▀ 
 ██    ██  ██    ██  ██           ██    
 ██    ██  ██    ██  ███████      ██    
 ██    ██  ██    ██  ██           ██    
 ██▄▄▄██   ██▄▄▄███  ██▄▄▄▄▄▄     ██    
 ▀▀▀▀▀      ▀▀▀▀ ▀▀  ▀▀▀▀▀▀▀▀     ▀▀    
 
\x1B[3mA Unified Deep Learning Framework for 
Predicting mRNA Translation Efficiency 
       Across Human Cell Types\x1B[0m             
                                        
""", flush=True)


parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter,
                                 description="Predicts TE using given 5' UTR and CDS sequences.")
sequence_args = parser.add_argument_group("Predict a single sequence from command line",
                                          "Both 5' UTR and CDS must be specified.")
sequence_args.add_argument("-u", "--utr5", type=str,
                    help="Input 5' UTR sequence")
sequence_args.add_argument("-c", "--cds", type=str,
                    help="Input CDS sequence")
batch_args = parser.add_argument_group("Predict a batch of sequences from a .tsv file",
                                       "(The file must have txID, utr5, cds, te columns)")
batch_args.add_argument("-i", "--input", type=str,
                    help="Path to input .tsv file containing sequences") 
batch_args.add_argument("-o", "--output", type=str,
                    help="Path to output .tsv file (if not specified, prints to stdout)")
model_args = parser.add_argument_group("Model arguments")
model_args.add_argument("--model_ckpt", type=str, default="misc/duet_base_model.ckpt",
                    help="Path to the model checkpoint (default=%(default)s)")
model_args.add_argument("--model_config", type=str, default="misc/duet_base_config.yaml",
                    help="Path to the model config (default=%(default)s)")
model_args.add_argument("--label_scaler", type=str, default="misc/duet_base_scaler.joblib",
                    help="Path to the StandardScaler object (default=%(default)s)")

args = parser.parse_args()

if args.input is None and (args.utr5 is None or args.cds is None):
    print("\x1b[1mEither --input or both --utr5 and --cds must be specified\x1b[0m\n")
    parser.print_help()
    exit(1)

if args.input is not None:
    batch_mode = True
    if args.utr5 is not None or args.cds is not None:
        print("\x1b[1m--input specified, ignoring --utr5 and --cds arguments\x1b[0m")
elif not all([args.utr5, args.cds]):
    print("\x1b[1mBoth --utr5 and --cds must be specified\x1b[0m")
    exit(1)
else:
    batch_mode = False
    if "U" in args.utr5:
        print("\x1b[1mReplacing 'U' in input 5' UTR sequence to 'T'\x1b[0m")
        args.utr5 = args.utr5.replace("U", "T")
    if "U" in args.cds:
        print("\x1b[1mReplacing 'U' in input CDS sequence to 'T'\x1b[0m")
        args.cds = args.cds.replace("U", "T")

if args.output is not None and os.path.exists(args.output):
    print(f"\x1b[1mOutput file {args.output} already exists, overwriting\x1b[0m")


print("Loading libraries...", end="\t", flush=True)
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import torch
import joblib
import secrets
import pandas as pd
import numpy as np
import random

from torch.utils.data import DataLoader
from tqdm import tqdm

from models.module import Module
from data.datamodule import DataModule
from configs.config import load_cfgs
print("Done.")


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


print("Loading model...", end="\t", flush=True)
cfg, dict_cfg = load_cfgs([args.model_config],
                          {"use_wandb":False, "datamodule.do_kfold_test":False},
                          compat_xref={"dataset.param.use_metadata": "dataset.param.use_sequence_feature",
                                       "dataset.param.metadata_path": "dataset.param.sequence_feature_path",
                                       "dataset.param.metadata_cols": "dataset.param.sequence_feature_cols",
                                       "dataset.param.use_triplet_phase": None,
                                       "dataset.param.utr_overlap_size": None,
                                       "dataset.param.cds_overlap_size": None,
                                       "model.param.global_block": None,})

model = Module.load_from_checkpoint(args.model_ckpt, cfg=cfg, dict_cfg=dict_cfg, strict=False)
model.eval()
print("Done.")


print("\nPredicting...")
label_scaler = joblib.load(args.label_scaler)
if batch_mode:
    datamodule = DataModule(cfg, dataset_path=args.input, scaler_obj=label_scaler)
else:
    tmp_fname = f"tmp_{secrets.token_hex(5)}.tsv"
    with open(tmp_fname, "w") as f:
        print("txID", "utr5", "cds", "te", sep="\t", file=f)
        print(0, args.utr5, args.cds, 1.0, sep="\t", file=f)
        
    datamodule = DataModule(cfg, dataset_path=tmp_fname, scaler_obj=label_scaler)    

datamodule.setup(stage="test")
dataloader = DataLoader(datamodule.dataset, batch_size=256, num_workers=10, shuffle=False)

pred_y = []
with torch.no_grad():
    for batch in tqdm(iter(dataloader)):
        for k in batch:
            batch[k] = batch[k].to(model.device)
        output, _ = model.model.predict(batch)
        pred_y.append(output.cpu().detach().numpy())

if batch_mode:
    pred_y = np.expm1(label_scaler.inverse_transform(np.concatenate(pred_y, axis=0).reshape(-1, 1)))
    output = pd.DataFrame({
        "txID": datamodule.dataset.data["txID"],
        "pred_TE": pred_y.flatten()
        })
    if args.output is not None:
        output.to_csv(args.output, sep="\t", index=False)
    else:
        print("\n\n")
        output.to_csv(sys.stdout, sep="\t", index=False)
else:
    pred_y = np.expm1(label_scaler.inverse_transform(pred_y[0].reshape(-1, 1)))
    print()
    print(f"Predicted TE: \x1b[1m{pred_y[0][0]}\x1b[0m")
    os.remove(tmp_fname)
