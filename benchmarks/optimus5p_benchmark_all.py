#%%

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import scipy.stats as stats
#import seaborn as sns
from sklearn import preprocessing
import random

import keras
np.random.seed(1337)

from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv1D

from keras.callbacks import ModelCheckpoint, EarlyStopping

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print('cwd:', os.getcwd())


#%%
def optimus5p_benchmark(dataset_name='U_1',
                        data_df=None,
                        rank_split=True,
                        test_scaler=None,
                        split_num=0,  # ignored if rank_split is true
                        split_max_num=10,
                        split_seed=42,
                        max_epochs=50,
                        border_mode='same',
                        inp_len=50,  # input length for Conv1D. 50 for MRL data, 100 for TE data
                        nodes=40,
                        layers=3,
                        filter_len=8,
                        nbr_filters=120,
                        dropout1=0, dropout2=0, dropout3=0,
                        debug=False,
                        ):

  # data preprocessing - split data into train, validation, and test sets
  if rank_split:
    dataset = data_df[data_df.library == f'{dataset_name}_train']
    # split randomly 95:5 validation data
    val_idx = np.random.choice(dataset.index, size=int(len(dataset) * 0.05), replace=False)
    train_idx = np.setdiff1d(dataset.index, val_idx)  # remove val_idx from train_idx

    train_data = dataset.loc[train_idx]
    val_data = dataset.loc[val_idx]
    test_data = data_df[data_df.library == f'{dataset_name}_test']
    print(f"Rank split: {dataset_name}, number of training samples: {len(train_data)}, "
         f"number of validation samples: {len(val_data)}, number of test samples: {len(test_data)}")

  else: # random split
    dataset = data_df[data_df.library == dataset_name]
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=split_max_num, shuffle=True, random_state=split_seed)
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    # select the k-th fold
    train_idx, test_idx = list(kf.split(dataset))[split_num]
    val_idx = np.random.choice(train_idx, size=int(len(train_idx) * 0.05), replace=False)
    train_idx = np.setdiff1d(train_idx, val_idx)  # remove val_idx from train_idx
    train_data = dataset.iloc[train_idx]
    val_data = dataset.iloc[val_idx]
    test_data = dataset.iloc[test_idx]
    print(f"Random split: {dataset_name}, number of training samples: {len(train_data)}, "
          f"number of validation samples: {len(val_data)}, number of test samples: {len(test_data)}")


  def one_hot_encode(df, col='utr5', seq_len=inp_len):
      # Dictionary returning one-hot encoding of nucleotides.
      nuc_d = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0]}

      # Creat empty matrix.
      vectors=np.empty([len(df),seq_len,4])

      # Iterate through UTRs and one-hot encode
      for i,seq in enumerate(df[col].str[-seq_len:]):
          seq = seq.lower()
          if len(seq) < seq_len:
              # If the sequence is shorter than seq_len, pad with 'n'
              seq = 'n' * (seq_len - len(seq)) + seq
          a = np.array([nuc_d[x] for x in seq])
          vectors[i] = a
      return vectors

  # One-hot encode the UTRs
  train_x = one_hot_encode(train_data, col='utr5', seq_len=inp_len)
  val_x = one_hot_encode(val_data, col='utr5', seq_len=inp_len)
  test_x = one_hot_encode(test_data, col='utr5', seq_len=inp_len)

  train_y = train_data['te'].to_numpy().reshape(-1, 1)
  val_y = val_data['te'].to_numpy().reshape(-1, 1)
  test_y = test_data['te'].to_numpy().reshape(-1, 1)

  if debug:
    return train_x, test_x


  ''' Build model archicture and fit.'''
  model = Sequential()
  if layers >= 1:
      model.add(Conv1D(activation="relu", input_shape=(inp_len, 4), padding=border_mode, filters=nbr_filters, kernel_size=filter_len))
  if layers >= 2:
      model.add(Conv1D(activation="relu", input_shape=(inp_len, 1), padding=border_mode, filters=nbr_filters, kernel_size=filter_len))
      model.add(Dropout(dropout1))
  if layers >= 3:
      model.add(Conv1D(activation="relu", input_shape=(inp_len, 1), padding=border_mode, filters=nbr_filters, kernel_size=filter_len))
      model.add(Dropout(dropout2))
  model.add(Flatten())

  model.add(Dense(nodes))
  model.add(Activation('relu'))
  model.add(Dropout(dropout3))

  model.add(Dense(1))
  model.add(Activation('linear'))

  #compile the model
  adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
  model.compile(loss='mean_squared_error', optimizer=adam)

  # prepare callbacks
  model_name = 'rank' if rank_split else f'random_{split_num}'
  model_save_path = f"./benchmark/{dataset_name}/{model_name}.h5"
  os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
  checkpoint_cb = ModelCheckpoint(
      model_save_path,          # 저장 경로
      monitor="val_loss",       # 모니터링 대상
      save_best_only=True,          # best만 저장
      save_weights_only=False       # 모델 전체 저장
  )
  earlystopping_cb = EarlyStopping(
      monitor='val_loss',  # 모니터링 대상
      patience=10,         # 10 epochs 동안 개선되지 않으면 중단
      restore_best_weights=False # 가장 좋은 가중치로 복원
  )

  if not debug:
    model.fit(train_x, train_y, batch_size=128,
              validation_data=(val_x, val_y),
              epochs=max_epochs, verbose=1,
              callbacks=[checkpoint_cb, earlystopping_cb])

  # load the best model
  model = keras.models.load_model(model_save_path)

  # make predictions on the test set
  pred = model.predict(test_x).reshape(-1, 1)

  # inverse scale the predictions if test_scaler is provided
  if test_scaler is not None:
      test_y = test_scaler.inverse_transform(test_y)
      pred = test_scaler.inverse_transform(pred)

  y = test_y.reshape(-1)
  yhat = pred.reshape(-1)

  result = {}
  r, pval = stats.pearsonr(y, yhat)
  rho, pval = stats.spearmanr(y, yhat)
  result['pearson'] = r
  result['spearman'] = rho

  # calulate r squared, RMSE, MAE
  from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
  result['r2'] = r2_score(y, yhat)
  result['rmse'] = np.sqrt(mean_squared_error(y, yhat))
  result['mae'] = mean_absolute_error(y, yhat)

  # print results
  print(result)
  if debug:
    return
  # save results as json
  import json
  result_path = f"./benchmark/{dataset_name}/{model_name}_result.json"
  with open(result_path, 'w') as f:
    json.dump(result, f, indent=4)
  # save y, yhat as tsv
  result_df = pd.DataFrame({'y': y, 'yhat': yhat})
  result_df.to_csv(f"./benchmark/{dataset_name}/{model_name}_result.tsv", sep='\t', index=False)

  # save results to all benchmark results
  all_results_path = './benchmark/benchmark_results2.tsv'
  if not os.path.exists(all_results_path):
    all_results_df = pd.DataFrame(columns=['dataset', 'model', 'pearson', 'spearman', 'r2', 'rmse', 'mae'])
  else:
    all_results_df = pd.read_csv(all_results_path, sep='\t')
  new_result = {
      'dataset': dataset_name,
      'model': model_name,
      'pearson': result['pearson'],
      'spearman': result['spearman'],
      'r2': result['r2'],
      'rmse': result['rmse'],
      'mae': result['mae']
  }
  all_results_df = all_results_df.append(new_result, ignore_index=True)
  all_results_df.to_csv(all_results_path, sep='\t', index=False)



#%%

def load_benchmark_data_all(save_scalers=True):
  full_df = []
  # we apply standard scaling to te, utr
  from sklearn.preprocessing import StandardScaler
  full_scalers = {}
  def safe_float(x):
      try: return float(x)
      except: return np.nan
  def load_data_from_tsv(file_path, sep='\t'):
    print(f"Loading data from: {file_path}")
    return pd.read_csv(file_path, sep=sep,
      converters={'te': safe_float, 'readcount': safe_float},
      dtype={'utr5': str}, on_bad_lines='skip').dropna()


  # MRL_rank
  data_base_path = '/fsx/s3/project/P240017_mRNA_UTR/data/MPRA/from_utrlm'
  datasets = 'U_1 U_2 m1pU_1 m1pU_2 mC-U_1 mC-U_2 pU_1 pU_2'.split()

  for dataset in datasets:
    train_data_path = f'{data_base_path}/train_{dataset}.tsv'
    test_data_path = f'{data_base_path}/{dataset}.tsv'
    train_df = load_data_from_tsv(train_data_path)
    test_df = load_data_from_tsv(test_data_path)
    train_df['library'] = f'{dataset}_train'
    test_df['library'] = f'{dataset}_test'

    scaler = StandardScaler()
    train_df['te'] = scaler.fit_transform(train_df[['te']])
    # save scaler for later use
    full_scalers[f'{dataset}_train'] = scaler
    if save_scalers:
      scaler_path = f'./benchmark/{dataset}_train_scaler.txt'
      with open(scaler_path, 'w') as f:
          f.write(f"mean: {scaler.mean_[0]}\n")
          f.write(f"scale: {scaler.scale_[0]}\n")

    scaler = StandardScaler()
    test_df['te'] = scaler.fit_transform(test_df[['te']])
    # save scaler for later use
    full_scalers[f'{dataset}_test'] = scaler
    if save_scalers:
      scaler_path = f'./benchmark/{dataset}_test_scaler.txt'
      with open(scaler_path, 'w') as f:
          f.write(f"mean: {scaler.mean_[0]}\n")
          f.write(f"scale: {scaler.scale_[0]}\n")

    full_df.append(train_df)
    full_df.append(test_df)


  # MRL_merged
  data_base_path = '/fsx/s3/project/P240017_mRNA_UTR/data/MPRA/from_utrlm'
  datasets = 'U_1 U_2 m1pU_1 m1pU_2 mC-U_1 mC-U_2 pU_1 pU_2'.split()

  for dataset in datasets:
    data_path = f'{data_base_path}/merged_{dataset}.tsv'
    df = load_data_from_tsv(data_path)
    df['library'] = dataset

    scaler = StandardScaler()
    df['te'] = scaler.fit_transform(df[['te']])
    # save scaler for later use
    full_scalers[dataset] = scaler
    if save_scalers:
      scaler_path = f'./benchmark/{dataset}_scaler.txt'
      with open(scaler_path, 'w') as f:
          f.write(f"mean: {scaler.mean_[0]}\n")
          f.write(f"scale: {scaler.scale_[0]}\n")

    full_df.append(df)

  # TE
  data_base_path = '/fsx/s3/project/P240017_mRNA_UTR/data/utr-lm/PoC'
  datasets = 'HEK Muscle pc3'.split()

  for dataset in datasets:
    data_path = f'{data_base_path}/{dataset}_sequence.tsv'
    df = pd.read_csv(data_path, sep='\t')
    df = df.rename(columns={'UTR5': 'utr5', 'TE': 'te'})
    df = df[['utr5', 'te']]
    # crop utr5 to 100bp, keeping the last 100bp
    df['utr5'] = df['utr5'].apply(lambda x: x[-100:] if isinstance(x, str) and len(x) > 100 else x)
    df['library'] = dataset

    # we convert te to log scale
    df['te'] = np.log1p(df['te'])
    scaler = StandardScaler()
    df['te'] = scaler.fit_transform(df[['te']])
    # save scaler for later use
    full_scalers[dataset] = scaler
    if save_scalers:
      scaler_path = f'./benchmark/{dataset}_scaler.txt'
      with open(scaler_path, 'w') as f:
          f.write(f"mean: {scaler.mean_[0]}\n")
          f.write(f"scale: {scaler.scale_[0]}\n")

    full_df.append(df)


  full_df = pd.concat(full_df, ignore_index=True)
  return full_df, full_scalers

#%%
full_df, full_scalars = load_benchmark_data_all()
full_df



#%%
#optimus5p_benchmark(dataset_name='HEK',
#                    data_df=full_df,
#                    rank_split=False,
#                    #test_scaler=full_scalars['HEK'],
#                    split_num=0, # ignored if rank_split is true
#                    max_epochs=50,
#                    inp_len=100,
#)

# %%

# run benchmark for mrl rank set
#for dataset_name in ['U_2', 'm1pU_1', 'm1pU_2', 'mC-U_1', 'mC-U_2', 'pU_1', 'pU_2']:
for dataset_name in ['U_1', 'U_2', 'm1pU_1', 'm1pU_2', 'mC-U_1', 'mC-U_2', 'pU_1', 'pU_2']:
    print(f"Running benchmark for {dataset_name}")
    optimus5p_benchmark(dataset_name=dataset_name,
                        data_df=full_df,
                        rank_split=True,
                        split_num=0, # ignored if rank_split is true
                        test_scaler=full_scalars[f'{dataset_name}_test'],
                        )

# run benchmark for MRL_merged set
for dataset_name in ['U_1', 'U_2', 'm1pU_1', 'm1pU_2', 'mC-U_1', 'mC-U_2', 'pU_1', 'pU_2']:
    print(f"Running benchmark for {dataset_name}")
    for split_num in range(10):
        print(f"Running split {split_num}")
        optimus5p_benchmark(dataset_name=dataset_name,
                            data_df=full_df,
                            rank_split=False,
                            split_num=split_num,
                            test_scaler=full_scalars[dataset_name],
                            )

# run benchmark for TE set
for dataset_name in ['HEK', 'Muscle', 'pc3']:
    print(f"Running benchmark for {dataset_name}")
    for split_num in range(10):
        print(f"Running split {split_num}")
        optimus5p_benchmark(dataset_name=dataset_name,
                            data_df=full_df,
                            rank_split=False,
                            split_num=split_num,
                            max_epochs=50,
                            inp_len=100,
                            )
## %%
import sys
sys.exit(0)  # exit the script after running the benchmark
#
##%%
## load test
#model_path = './benchmark/m1pU_1/random_3.h5'
#full_df, full_scalars = load_benchmark_data_all(save_scalers=False)
#full_df
#
##%%
#train_x, test_x = optimus5p_benchmark(
#                    dataset_name='m1pU_1',
#                    data_df=full_df,
#                    rank_split=False,
#                    test_scaler=full_scalars['m1pU_1_test'], # use the test scaler for m1pU_1
#                    split_num=3, # ignored if rank_split is true
#                    max_epochs=50,
#                    debug=True,)
## %%
## check test_x is all same tensor
#test_x[0] == test_x[1]  # should be True if all same
## %%


#%%
# run custom runs
for dataset_name, split_num in [
                                #('m1pU_1', 0),
                                #('m1pU_1', 3),
                                #('m1pU_2', 1),
                                #('mC-U_1', 3),
                                #('mC-U_1', 6),
                                ('pU_1', 8),
                                ]:
    print(f"Running benchmark for {dataset_name}")
    print(f"Running split {split_num}")
    optimus5p_benchmark(dataset_name=dataset_name,
                        data_df=full_df,
                        rank_split=False,
                        split_num=split_num,
                        test_scaler=full_scalars[dataset_name],
                        )
# %%
for dataset_name in ['U_1']:
    print(f"Running benchmark for {dataset_name}")
    optimus5p_benchmark(dataset_name=dataset_name,
                        data_df=full_df,
                        rank_split=True,
                        split_num=0, # ignored if rank_split is true
                        test_scaler=full_scalars[f'{dataset_name}_test'],
                        )
# %%
for dataset_name, split_num in [
                                ('HEK', 3),
                                ]:
        optimus5p_benchmark(dataset_name=dataset_name,
                            data_df=full_df,
                            rank_split=False,
                            split_num=split_num,
                            max_epochs=50,
                            inp_len=100,
                            )
# %%
