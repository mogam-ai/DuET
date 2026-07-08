# Source: MJ3_Finetune_extract_append_predictor_Sample_10fold-lr-huber-DDP.py
# Modified for benchlib integration.
# Model: ESM2-SISS + CNN_linear head, HuberLoss, SGD lr=0.01
# Reference: Chu et al., UTR-LM (2024)
#%%
import os, sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
use_cuda = True
device = 'cuda' if use_cuda else 'cpu'
class Empty: pass
args = Empty()

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

#%%
#======== experiment setup =========#
args.modelfile = os.path.join(_THIS_DIR, 'params', 'Model', 'Pretrained',
    'ESM2SISS_FS4.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_lr1e-05_supervisedweight1.0_structureweight1.0_MLMLossMin_epoch93.pkl')

args.random_seed = 1337
args.scaler = True
args.log2 = False
args.batch_toks = 8192
args.avg_emb = False
args.bos_emb = True
args.magic = False
args.init_epoch = 0
args.epochs = 200
args.kfold_split = 10
args.lr = 1e-2

#%%
# data setup
import argparse
import pandas as pd
import numpy as np
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, required=True)
parser.add_argument('--utr_col', type=str, default='utr5')
parser.add_argument('--label_col', type=str, default='te')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--inp_len', type=int, default=50)
parser.add_argument('--train_idx', type=str, required=True)
parser.add_argument('--val_idx', type=str, required=True)
parser.add_argument('--test_idx', type=str, required=True)
parser.add_argument('--test_data_file', type=str, default=None, help='separate test file (for rank split)')
parser.add_argument('--scaler_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('-f', '--file', help='jupyter compat')
data_args, _ = parser.parse_known_args()

args.lr = data_args.lr
args.epochs = data_args.epochs
args.utr_col = data_args.utr_col
inp_len = data_args.inp_len

#%%
# read data and setup
data = pd.read_csv(data_args.data_file, sep='\t')
print(f'Data: {data.shape}')

# Load indices and scaler
train_index = np.load(data_args.train_idx)
val_index = np.load(data_args.val_idx)
test_index = np.load(data_args.test_idx)

scaler_data = joblib.load(data_args.scaler_path)
scaler = scaler_data["scaler"]

# Scale labels
label_col = data_args.label_col
train_obj_col = f'{label_col}_scaled'
data[train_obj_col] = scaler.transform(data[label_col].values.reshape(-1, 1)).flatten()

# Cut UTR
data[args.utr_col] = data[args.utr_col].str[-inp_len:]

#%%
# generate dataset and dataloader
from torch.utils.data import DataLoader
from esm import Alphabet, FastaBatchedDataset
alphabet = Alphabet(mask_prob=0.0, standard_toks='AGCT')

def generate_dataset_dataloader(e_data, utr_col, label_col):
    dataset = FastaBatchedDataset(e_data[label_col], e_data[utr_col], mask_prob=0.0)
    batches = dataset.get_batch_indices(toks_per_batch=args.batch_toks, extra_toks_per_seq=1)
    dataloader = DataLoader(dataset, collate_fn=alphabet.get_batch_converter(),
                            batch_sampler=batches, shuffle=False)
    print(f"{len(dataset)} sequences")
    return dataset, dataloader

#%%
# model code
import torch.nn as nn
from esm.model.esm2_secondarystructure import ESM2 as ESM2_SISS
from esm.model.esm2_supervised import ESM2
from esm.model.esm2_only_secondarystructure import ESM2 as ESM2_SS

embed_dim = 128
heads = 16
layers = 6
nodes = 40
cnn_layers = 0
dropout3 = 0.5

class CNN_linear(nn.Module):
    def __init__(self, border_mode='same', filter_len=8, nbr_filters=120,
                 dropout1=0, dropout2=0):
        super(CNN_linear, self).__init__()
        self.embedding_size = embed_dim
        self.border_mode = border_mode
        self.inp_len = inp_len
        self.nodes = nodes
        self.cnn_layers = cnn_layers
        self.filter_len = filter_len
        self.nbr_filters = nbr_filters
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3

        if 'SISS' in args.modelfile:
            self.esm2 = ESM2_SISS(num_layers=layers, embed_dim=embed_dim,
                                  attention_heads=heads, alphabet=alphabet)
        elif 'SS' in args.modelfile:
            self.esm2 = ESM2_SS(num_layers=layers, embed_dim=embed_dim,
                                attention_heads=heads, alphabet=alphabet)
        else:
            self.esm2 = ESM2(num_layers=layers, embed_dim=embed_dim,
                             attention_heads=heads, alphabet=alphabet)

        self.conv1 = nn.Conv1d(in_channels=self.embedding_size,
                      out_channels=self.nbr_filters, kernel_size=self.filter_len, padding=self.border_mode)
        self.conv2 = nn.Conv1d(in_channels=self.nbr_filters,
                      out_channels=self.nbr_filters, kernel_size=self.filter_len, padding=self.border_mode)
        self.dropout1 = nn.Dropout(self.dropout1)
        self.dropout2 = nn.Dropout(self.dropout2)
        self.dropout3 = nn.Dropout(self.dropout3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        if args.avg_emb or args.bos_emb:
            self.fc = nn.Linear(in_features=embed_dim, out_features=self.nodes)
        else:
            self.fc = nn.Linear(in_features=inp_len * embed_dim, out_features=self.nodes)
        if args.avg_emb or args.bos_emb:
            self.linear = nn.Linear(in_features=self.nbr_filters, out_features=self.nodes)
        else:
            self.linear = nn.Linear(in_features=inp_len * self.nbr_filters, out_features=self.nodes)
        self.output = nn.Linear(in_features=self.nodes, out_features=1)
        if self.cnn_layers == -1: self.direct_output = nn.Linear(in_features=embed_dim, out_features=1)
        if args.magic: self.magic_output = nn.Linear(in_features=1, out_features=1)

    def forward(self, tokens, need_head_weights=True, return_contacts=True, return_representation=True):
        x = self.esm2(tokens, [layers], need_head_weights, return_contacts, return_representation)
        if args.avg_emb:
            x = x["representations"][layers][:, 1:inp_len+1].mean(1)
            x_o = x.unsqueeze(2)
        elif args.bos_emb:
            x = x["representations"][layers][:, 0]
            x_o = x.unsqueeze(2)
        else:
            x_o = x["representations"][layers][:, 1:inp_len+1]
            x_o = x_o.permute(0, 2, 1)

        if self.cnn_layers >= 1:
            x_cnn1 = self.conv1(x_o)
            x_o = self.relu(x_cnn1)
        if self.cnn_layers >= 2:
            x_cnn2 = self.conv2(x_o)
            x_relu2 = self.relu(x_cnn2)
            x_o = self.dropout1(x_relu2)
        if self.cnn_layers >= 3:
            x_cnn3 = self.conv2(x_o)
            x_relu3 = self.relu(x_cnn3)
            x_o = self.dropout2(x_relu3)

        x = self.flatten(x_o)
        if self.cnn_layers != -1:
            if self.cnn_layers != 0:
                o_linear = self.linear(x)
            else:
                o_linear = self.fc(x)
            o_relu = self.relu(o_linear)
            o_dropout = self.dropout3(o_relu)
            o = self.output(o_dropout)
        else:
            o = self.direct_output(x)
        if args.magic:
            o = self.magic_output(o)
        return o

#%%
# model setup
def model_setup():
    model = CNN_linear().to(device)
    ret = model.esm2.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(args.modelfile, map_location='cpu').items()},
        strict=False)
    print('model setup result: ', ret)
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    return model, optimizer

#%%
# metric setup
import scipy.stats as stats
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def performances(label, pred):
    label, pred = list(label), list(pred)
    slope, intercept, r_value, p_value, std_err = stats.linregress(label, pred)
    r = r_value**2
    R2 = r2_score(label, pred)
    rmse = np.sqrt(mean_squared_error(label, pred))
    mae = mean_absolute_error(label, pred)
    try: pearson_r = pearsonr(label, pred)[0]
    except: pearson_r = -1e-9
    try: sp_cor = spearmanr(label, pred)[0]
    except: sp_cor = -1e-9
    print(f'r-squared = {r:.4f} | pearson r = {pearson_r:.4f} | spearman R = {sp_cor:.4f} | R-squared = {R2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}')
    return [r, pearson_r, sp_cor, R2, rmse, mae]

#%%
# train_step, eval_step
from tqdm import tqdm

def train_step(dataloader, model, optimizer, epoch):
    model.train()
    y_pred_list, y_true_list, loss_list = [], [], []
    for (labels, strs, masked_strs, toks, masked_toks, _) in tqdm(dataloader):
        toks = toks.to(device)
        labels = torch.FloatTensor(labels).to(device).reshape(-1, 1)
        outputs = model(toks, return_representation=True, return_contacts=True)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.cpu().detach())
        y_true_list.extend(labels.cpu().reshape(-1).tolist())
        y_pred_list.extend(outputs.reshape(-1).cpu().detach().tolist())
    loss_epoch = float(torch.Tensor(loss_list).mean())
    print(f'Train: Epoch-{epoch}/{args.epochs} | Loss = {loss_epoch:.4f} | ', end='')
    metrics = performances(y_true_list, y_pred_list)
    return metrics, loss_epoch


def eval_step(dataloader, model, epoch):
    model.eval()
    y_pred_list, y_true_list, strs_list = [], [], []
    with torch.no_grad():
        for (labels, strs, masked_strs, toks, masked_toks, _) in tqdm(dataloader):
            strs_list.extend(strs)
            toks = toks.to(device)
            labels = torch.FloatTensor(labels).to(device).reshape(-1, 1)
            outputs = model(toks, return_representation=True, return_contacts=True)
            y_true_list.extend(labels.cpu().reshape(-1).tolist())
            y_pred_list.extend(outputs.reshape(-1).cpu().detach().tolist())
        loss_epoch = criterion(torch.Tensor(y_pred_list).reshape(-1, 1), torch.Tensor(y_true_list).reshape(-1, 1))
        print(f'Test: Epoch-{epoch}/{args.epochs} | Loss = {loss_epoch:.4f} | ', end='')
        metrics = performances(y_true_list, y_pred_list)
        e_pred = pd.DataFrame([strs_list, y_true_list, y_pred_list], index=['utr', 'y_true', 'y_pred']).T
    return metrics, -metrics[2], e_pred

#%%
# main training
from copy import deepcopy

print(f'==================== Single Fold ====================')

e_train = data.iloc[train_index, :]
e_val = data.iloc[val_index, :]
e_test = data.iloc[test_index, :] if data_args.test_data_file is None else pd.read_csv(data_args.test_data_file, sep='\t')
if data_args.test_data_file is not None:
    e_test[train_obj_col] = scaler.transform(e_test[label_col].values.reshape(-1, 1)).flatten()
    e_test[args.utr_col] = e_test[args.utr_col].str[-inp_len:]

train_dataset, train_dataloader = generate_dataset_dataloader(e_train, args.utr_col, train_obj_col)
val_dataset, val_dataloader = generate_dataset_dataloader(e_val, args.utr_col, train_obj_col)
test_dataset, test_dataloader = generate_dataset_dataloader(e_test, args.utr_col, train_obj_col)

loss_best, ep_best = np.inf, -1
model, optimizer = model_setup()
criterion = torch.nn.HuberLoss()
model_best = deepcopy(model)

for epoch in range(args.init_epoch + 1, args.init_epoch + args.epochs + 1):
    print(f'Epoch {epoch}/{args.epochs}')
    metrics_train, loss_train = train_step(train_dataloader, model, optimizer, epoch)
    metrics_val, loss_val, _ = eval_step(val_dataloader, model, epoch)
    if loss_val < loss_best:
        loss_best, ep_best = loss_val, epoch
        model_best = deepcopy(model)

# Test with best model
model = model_best
print(f'Best model: Epoch {ep_best} | Loss = {loss_best:.4f}')
metrics_test, loss_test, data_pred = eval_step(test_dataloader, model, ep_best)

# Save predictions (in original scale)
y_true_scaled = data_pred['y_true'].astype(float).values
y_pred_scaled = data_pred['y_pred'].astype(float).values
y_true_raw = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
y_pred_raw = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

os.makedirs(data_args.output_dir, exist_ok=True)
pd.DataFrame({'y_true': y_true_raw, 'y_pred': y_pred_raw}).to_csv(
    os.path.join(data_args.output_dir, 'predictions.tsv'), sep='\t', index=False)
print(f'Saved predictions to {data_args.output_dir}')
