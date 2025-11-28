# Program:     DuET v1.0.0
# Author:      Sungho Lee, Jae-Won Lee, Jinhee Hong
# Affiliation: MOGAM Institute for Biomedical Research
# Contact:     sungho.lee@mogam.re.kr; jaewon.lee@mogam.re.kr; jinhee.hong@mogam.re.kr
# Citation:    TBD


import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import Dict, Callable, Literal

from models.module import Model
from configs.config import LossConfig


class SELayerSimple(nn.Module):
  def __init__(self, inp, oup, reduction=4):
    super().__init__()
    # Prevent inner dimension from being too small
    inp_div = max(reduction, int(inp // reduction))
    self.fc = nn.Sequential(
        nn.Linear(oup, inp_div),
        nn.SiLU(),
        nn.Linear(inp_div, oup),
        nn.Sigmoid()
    )

  def forward(self, x):
    b, c, _, = x.size() # (batch, channel, seq)
    y = x.view(b, c, -1).mean(dim=2) # (batch, channel)
    y = self.fc(y).view(b, c, 1) # (batch, channel, 1)
    return x * y # (batch, channel, seq): channel-wise scaling


class ResidualBlock(nn.Module):
  def __init__(self, inner_block: nn.Module, residual: bool=True):
    super().__init__()
    self.inner_block = inner_block
    self.residual = residual

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.residual: return x + self.inner_block(x)
    else: return self.inner_block(x)


class MBConvBlock(nn.Module):
  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      resize_only: bool,
      fused: bool,
      kernel_size: int,
      stride: int,
      dropout: float,
      norm_layer: Callable[..., nn.Module],
      activation: Callable[..., nn.Module],
      expand_ratio: int=4,
      se_reduction: int=4,
      filter_per_group: int=2,
      residual: bool=True,
      resize_bias: bool=False,
  ):
    '''
    LegNet+EfficientNetV2 like block

    Args:
      in_channels (int): number of input channels
      out_channels (int): number of output channels
      resize_only (bool): whether to use only resize block
      fused (bool): whether to use fused MBConv
      kernel_size (int): kernel size
      stride (int): stride
      norm_layer (Callable[..., nn.Module]): normalization layer
      activation (Callable[..., nn.Module]): activation function
      expand_ratio (int): expansion ratio
      se_reduction (int): reduction ratio for SE layer
      filter_per_group (int): number of filter for groups in depthwise convolution (set to 1 for depthwise)
      residual (bool): whether to use residual connection
      resize_bias (bool): whether to use bias in resize block
    '''
    super().__init__()

    self.inv_res_concat = not resize_only and stride == 1 # for inv_res concatenation, length should be same
    self.padding = 'same' if stride == 1 else 'valid'
    self.resize_only = resize_only

    if not self.resize_only:
      # inv residual block
      self.inv_res_block = []
      if fused: # fused MBConv
        self.inv_res_block.extend([
          nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels*expand_ratio,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            bias=False,
          ),
          norm_layer(out_channels*expand_ratio),
          activation(),
        ])
      else: # MBConv
        self.inv_res_block.extend([
          nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels*expand_ratio,
            kernel_size=1,
            padding='same',
            bias=False,
          ),
          norm_layer(out_channels*expand_ratio),
          activation(),
          ResidualBlock(
            nn.Conv1d(
              in_channels=out_channels*expand_ratio,
              out_channels=out_channels*expand_ratio,
              kernel_size=kernel_size,
              stride=stride,
              padding=self.padding,
              groups=out_channels*expand_ratio // filter_per_group,
              bias=False,
            ),
            residual=residual if stride == 1 else False),
          norm_layer(out_channels*expand_ratio),
          activation(),
        ])
      # SE layer
      self.inv_res_block.append(
        SELayerSimple(
          in_channels,
          out_channels*expand_ratio,
          reduction=se_reduction))
      # pointwise linear
      self.inv_res_block.extend([
        nn.Conv1d(
          in_channels=out_channels*expand_ratio,
          out_channels=in_channels,
          kernel_size=1,
          padding='same',
          bias=False,
        ),
        norm_layer(in_channels),
        nn.Dropout(p=dropout),
        activation(),
      ])
      # init inv res block
      self.inv_res_block = nn.Sequential(*self.inv_res_block)

    # resize block
    self.resize_block = nn.Sequential(
      nn.Conv1d(
        in_channels=in_channels*2 if self.inv_res_concat else in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding='same',
        bias=resize_bias,
      ),
      norm_layer(out_channels),
      #nn.Dropout(p=dropout),
      activation(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if not self.resize_only:
      ret = self.inv_res_block(x)
      x = torch.cat([x, ret], dim=1) if self.inv_res_concat else ret
    return self.resize_block(x) # (batch, out_channels, seq)


class DuetModel(Model):
  def __init__(
      self,
      
      utr_input_channels: int,
      utr_block: list[dict],
      utr_expand_ratio: int,
      utr_se_reduction: int,
      utr_filter_per_group: int,
      utr_stride: int,
      utr_dropout: float,

      norm_layer: str,
      act_layer: str,

      use_sequence_feature: bool,
      sequence_feature_size: int,
      utr_seq_size: int,
      cds_seq_size: int,
      
      loss_cfg: LossConfig,
      
      cds_input_channels: int=64,
      cds_block: list[dict]=[],
      cds_expand_ratio: int=4,
      cds_se_reduction: int=4,
      cds_filter_per_group: int=2,
      cds_stride: int=1,
      cds_dropout: float=0.1,
      
      residual: bool = True,
      bn_momentum: float = 0.1,
      
      feature_compress_scale: int = 1,
      encoder: Literal['single', 'dual'] = 'dual',
      
      log_label: bool = True,
      scale_label: bool = True,
      label_scale: float = 1.0,
      label_mean: float = 0.0,
      use_start_context: bool = False,
      do_layernorm: bool = False,
      
  ):
    super().__init__()
    # loss parsing
    match loss_cfg.name.lower():
      case 'mse': self.loss = nn.MSELoss(reduction='mean')
      case 'mae': self.loss = nn.L1Loss(reduction='mean')
      case _: raise NotImplementedError(f"loss {loss_cfg.name} is not implemented")
    #  norm layer parsing
    match norm_layer.lower():
      case 'batchnorm':
        self.norm_layer = partial(nn.BatchNorm1d, momentum=bn_momentum)
      case _: raise NotImplementedError(f"norm layer {norm_layer} is not implemented")
    # act layer parsing
    match act_layer.lower():
      case 'silu': self.act_layer = nn.SiLU
      case _: raise NotImplementedError(f"activation layer {act_layer} is not implemented")
    # residual setup
    self.residual = residual
    
    assert encoder in ['single', 'dual'], f"{encoder} encoder is not implemented"
    self.dual = encoder == "dual"

    self.log_label = log_label
    self.scale_label = scale_label
    self.register_buffer('label_scale', torch.tensor(label_scale, dtype=torch.float32))
    self.register_buffer('label_mean', torch.tensor(label_mean, dtype=torch.float32))

    # main model building - parsing modules
    def build_core_block(core_block: list[dict], input_channels: int,
                         default_stride: int, default_expand_ratio: int, 
                         default_se_reduction: int, default_filter_per_group:int, 
                         default_dropout: float):
      block = []
      prev_channels = input_channels
      for block_idx in range(len(core_block)):
        # parse block config
        block_cfg = core_block[f'block_{block_idx}']
        layer_name = block_cfg['model']
        assert layer_name in ['NormActConv', 'FusedMBConv', 'MBConv'], f"layer {layer_name} is not implemented"
        n_layer = int(block_cfg.get('n_layer', 1)) # default one layer
        out_channels = int(block_cfg['out_channels'])
        disable_norm = block_cfg.get('disable_norm', False)

        for _ in range(n_layer):
          block.append(MBConvBlock(
            in_channels=prev_channels,
            out_channels=out_channels,
            resize_only=(layer_name == 'NormActConv'),
            fused=(layer_name == 'FusedMBConv'),
            kernel_size=int(block_cfg['kernel_size']),
            stride=int(block_cfg.get('stride', default_stride)),
            expand_ratio=int(block_cfg.get('expand_ratio', default_expand_ratio)),
            se_reduction=int(block_cfg.get('se_reduction', default_se_reduction)),
            filter_per_group=int(block_cfg.get('filter_per_group', default_filter_per_group)),
            dropout=float(block_cfg.get('dropout', default_dropout)),
            norm_layer=self.norm_layer if not disable_norm else nn.Identity,
            activation=self.act_layer,
            residual=self.residual,
            resize_bias=block_cfg.get('resize_bias', False),
          ))
          
          prev_channels = out_channels # update prev_channels
          
      return nn.Sequential(*block), prev_channels
    # setup utr and cds block
    
    
    if not self.dual and utr_seq_size == 0: # single CDS encoder. ignore triplet phase and other things
      utr_input_channels = cds_input_channels
    
    # if encoder == "single", use UTR block for both UTR and CDS. Dataloader will merge CDS into UTR.
    self.utr_block, self.utr_output_channels = build_core_block(utr_block, utr_input_channels, utr_stride, 
                                                                utr_expand_ratio, utr_se_reduction, utr_filter_per_group, 
                                                                utr_dropout)
    if self.dual:
      self.cds_block, self.cds_output_channels = build_core_block(cds_block, cds_input_channels, cds_stride, 
                                                                  cds_expand_ratio, cds_se_reduction, cds_filter_per_group, 
                                                                  cds_dropout)
    self.use_start_context = use_start_context
    if use_start_context:
      start_block = utr_block
      for block, cfg in start_block.items():
        cfg["out_channels"] = cfg["out_channels"] // 2
      self.start_block, self.start_output_channels = build_core_block(start_block, 4, utr_stride, 
                                                                      utr_expand_ratio, utr_se_reduction, utr_filter_per_group, 
                                                                      utr_dropout)


    final_input_channels = (self.utr_output_channels 
                            + (self.cds_output_channels if encoder == "dual" else 0)
                            + (self.start_output_channels if use_start_context else 0))

    # output layer
    self.use_sequence_feature = use_sequence_feature
    if self.use_sequence_feature:
      self.sequence_feature_block = nn.Sequential(
        nn.Linear(sequence_feature_size,
                  sequence_feature_size // feature_compress_scale),
        self.act_layer()
      )
      sequence_feature_size = sequence_feature_size // feature_compress_scale
    
    self.do_layernorm = do_layernorm
    if self.do_layernorm:
      self.utr_embed_norm = nn.LayerNorm(self.utr_output_channels)
      if self.dual:
        self.cds_embed_norm = nn.LayerNorm(self.cds_output_channels)
      if self.use_start_context:
        self.start_embed_norm = nn.LayerNorm(self.start_output_channels)
      if self.use_sequence_feature:
        self.sequence_feature_embed_norm = nn.LayerNorm(sequence_feature_size // feature_compress_scale)
    
    dense_input_size = final_input_channels + (sequence_feature_size if use_sequence_feature else 0)
    self.dense = nn.Sequential(
      #nn.Linear(
      #  in_features=dense_input_size,
      #  out_features=dense_input_size,
      #),
      nn.Linear(
        in_features=dense_input_size,
        out_features=1,
      ),
      nn.Flatten(),
    )

  def get_embedding(self, batch) -> torch.Tensor:
    utr = batch['utr'].float().permute(0,2,1) # (batch, channel, utr_seq)
    utr = self.utr_block(utr) # (batch, out_channels, utr_seq)
    
    if self.dual:
      cds = batch['cds'].float().permute(0,2,1) # (batch, channel, cds_seq)
      cds = self.cds_block(cds) # (batch, out_channels, cds_seq)

    if self.use_start_context:
      start = batch['start'].float().permute(0,2,1) # (batch, channel, cds_seq)
      start = self.start_block(start) # (batch, out_channels, cds_seq)

    x = F.adaptive_avg_pool1d(utr, 1).squeeze(2) # (batch, utr_out_channels)
    if self.do_layernorm:
      x = self.utr_embed_norm(x)
    if self.dual:
      cds = F.adaptive_avg_pool1d(cds, 1).squeeze(2) # (batch, cds_out_channels)
      if self.do_layernorm:
        cds = self.cds_embed_norm(cds)
      x = torch.cat([x, cds], dim=-1) # (batch, utr_out_channels+cds_out_channels)
    if self.use_start_context:
      start = F.adaptive_avg_pool1d(start, 1).squeeze(2) # (batch, start_out_channels)
      if self.do_layernorm:
        start = self.start_embed_norm(start)
      x = torch.cat([x, start], dim=-1) # (batch, utr_out_channels+cds_out_channels+start_out_channels)
      
    sequence_feature_tensor = None
    if self.use_sequence_feature:
      sequence_feature_tensor = self.sequence_feature_block(batch['sequence_feature']) # (batch, sequence_feature)
      if self.do_layernorm:
        sequence_feature_tensor = self.sequence_feature_embed_norm(sequence_feature_tensor)
      x = torch.cat([x, sequence_feature_tensor], dim=-1) # (batch, out_channels+sequence_feature)
      
    return x.detach(), x

  def predict(self, batch) -> torch.Tensor:
    _, x = self.get_embedding(batch)
    score = self.dense(x).reshape(-1) # (batch,)
    
    return score.detach(), score
  
  def forward(self, batch: Dict) -> torch.Tensor:
    yhat, yhat_logit = self.predict(batch)
    loss = self.loss(yhat_logit, batch['y'])
    return yhat, loss
