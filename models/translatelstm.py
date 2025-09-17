# Program:     DuET v1.0.0
# Author:      Sungho Lee, Jae-Won Lee, Jinhee Hong
# Affiliation: MOGAM Institute for Biomedical Research
# Contact:     sungho.lee@mogam.re.kr; jaewon.lee@mogam.re.kr; jinhee.hong@mogam.re.kr
# Citation:    TBD


import torch
import torch.nn as nn

from typing import Dict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.module import Model
from configs.config import LossConfig


class LstmModel(Model):
  def __init__(
      self,
      feature_size: int,
      sequence_feature_size: int,
      hidden_dim: int,
      dropout: float,
      loss_cfg: LossConfig,
  ):
    super().__init__()

    self.hidden_dim = hidden_dim
    self.embedding = nn.Embedding(feature_size, hidden_dim)
    self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
    self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
    self.dropout = nn.Dropout(dropout)

    # Fully connected layer for combined inputs
    self.fc = nn.Linear(hidden_dim * 2 + sequence_feature_size, 1)

    if loss_cfg.name.lower() == 'mse':
      self.loss = nn.MSELoss()
    else:
      raise NotImplementedError(f"loss {loss_cfg.name} is not implemented")

  def predict(self, batch) -> torch.Tensor:
    seq, sequence_feature, seq_len = batch['x'], batch['sequence_feature'], batch['seq_len'].cpu()
    seq = self.embedding(seq)  # (batch_size, seq_len, hidden_dim)
    packed_seq = pack_padded_sequence(seq, seq_len,
                                      batch_first=True, enforce_sorted=False)
    packed_output1, _ = self.lstm1(packed_seq)
    packed_output2, _ = self.lstm2(packed_output1)
    output, _ = pad_packed_sequence(packed_output2, batch_first=True)
    output = self.dropout(output)  # (batch_size, seq_len, hidden_dim * 2)

    # Concatenate the output of the last hidden state of the forward and backward LSTM
    # we should consider different lengths of sequences
    forward_last = output[torch.arange(len(seq_len)), seq_len - 1, :self.hidden_dim]
    backward_first = output[:, 0, self.hidden_dim:]
    output = torch.cat((forward_last, backward_first), dim=1)

    # Concatenate the output of the last hidden state of the forward and backward LSTM with metadata
    output = torch.cat((output, sequence_feature), dim=1)

    yhat = self.fc(output)  # (batch_size, 1)
    yhat = yhat.reshape(-1)  # (batch_size, )
    
    return yhat.detach(), yhat  # first one is for prediction/should be denormalized

  def forward(self, batch: Dict) -> torch.Tensor:
    yhat, yhat_norm = self.predict(batch)
    loss = self.loss(yhat_norm, batch['y'])
    return yhat, loss


class LstmDualModel(Model):
  def __init__(
      self,
      feature_size: int,
      sequence_feature_size: int,
      hidden_dim: int,
      utr_dropout: float,
      cds_dropout: float,
      loss_cfg: LossConfig,
  ):
    super().__init__()

    self.hidden_dim = hidden_dim
    
    self.utr_embedding = nn.Embedding(feature_size, hidden_dim)
    self.utr_lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
    self.utr_lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
    self.utr_dropout = nn.Dropout(utr_dropout)

    self.cds_embedding = nn.Embedding(feature_size, hidden_dim)
    self.cds_lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
    self.cds_lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
    self.cds_dropout = nn.Dropout(cds_dropout)

    # Fully connected layer for combined inputs
    self.fc = nn.Linear(hidden_dim * 4 + sequence_feature_size, 1)

    if loss_cfg.name.lower() == 'mse':
      self.loss = nn.MSELoss()
    else:
      raise NotImplementedError(f"loss {loss_cfg.name} is not implemented")

  def predict(self, batch) -> torch.Tensor:
    
    sequence_feature = batch['sequence_feature']
    utr, utr_len = batch['utr'], batch['utr_len'].cpu()
    cds, cds_len = batch['cds'], batch['cds_len'].cpu()
    
    utr_seq = self.utr_embedding(utr)  # (batch_size, utr_len, hidden_dim)
    packed_utr_seq = pack_padded_sequence(utr_seq, utr_len,
                                          batch_first=True, enforce_sorted=False)
    packed_utr_output1, _ = self.utr_lstm1(packed_utr_seq)
    packed_utr_output2, _ = self.utr_lstm2(packed_utr_output1)
    utr_output, _ = pad_packed_sequence(packed_utr_output2, batch_first=True)
    utr_output = self.utr_dropout(utr_output)  # (batch_size, utr_len, hidden_dim * 2)
    
    cds_seq = self.cds_embedding(cds)  # (batch_size, cds_len, hidden_dim)
    packed_cds_seq = pack_padded_sequence(cds_seq, cds_len,
                                          batch_first=True, enforce_sorted=False)
    packed_cds_output1, _ = self.cds_lstm1(packed_cds_seq)
    packed_cds_output2, _ = self.cds_lstm2(packed_cds_output1)
    cds_output, _ = pad_packed_sequence(packed_cds_output2, batch_first=True)
    cds_output = self.cds_dropout(cds_output)  # (batch_size, cds_len, hidden_dim * 2)

    # Concatenate the output of the last hidden state of the forward and backward LSTM
    # we should consider different lengths of sequences
    forward_last_utr = utr_output[torch.arange(len(utr_len)), utr_len - 1, :self.hidden_dim]
    backward_first_utr = utr_output[:, 0, self.hidden_dim:]
    utr_output = torch.cat((forward_last_utr, backward_first_utr), dim=1)
    
    forward_last_cds = cds_output[torch.arange(len(cds_len)), cds_len - 1, :self.hidden_dim]
    backward_first_cds = cds_output[:, 0, self.hidden_dim:]
    cds_output = torch.cat((forward_last_cds, backward_first_cds), dim=1)
    
    output = torch.cat((utr_output, cds_output, sequence_feature), dim=1)  # (batch_size, hidden_dim * 4 + sequence_feature_size)

    yhat = self.fc(output)  # (batch_size, 1)
    yhat = yhat.reshape(-1)  # (batch_size, )
    
    return yhat.detach(), yhat  # first one is for prediction/should be denormalized


  def forward(self, batch: Dict) -> torch.Tensor:
    yhat, yhat_norm = self.predict(batch)
    loss = self.loss(yhat_norm, batch['y'])
    return yhat, loss
