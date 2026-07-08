from typing import Tuple
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional import pearson_corrcoef, r2_score, spearman_corrcoef
from src.utils.helpers import masked_mse_loss


class ConvBlock(nn.Module):
    """Conv1D block with or without residual layers"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        eps: float,
        dropout: float,
        residual: bool = False,
        is_initial: bool = False,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.residual = residual
        self.is_initial = is_initial
        self.activation = activation

        if self.is_initial:
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False,
            )
            if self.residual:
                # (N, L, C) --> (N, L, C), normalize across the last dimension
                self.residual_layernorm = nn.LayerNorm(
                    normalized_shape=out_channels, eps=eps
                )
                self.residual_conv = nn.Sequential(
                    self.activation(),
                    nn.Conv1d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding="same",
                    ),
                    nn.Dropout(dropout),
                )
                self.scale = nn.Parameter(torch.zeros(out_channels))
        else:
            # (N, L, C) --> (N, L, C), normalize across the last dimension
            self.layernorm = nn.LayerNorm(normalized_shape=in_channels, eps=eps)
            self.conv = nn.Sequential(
                self.activation(),
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.Dropout(dropout),
            )
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

            if self.residual:
                # (N, L, C) --> (N, L, C), normalize across the last dimension
                self.residual_layernorm = nn.LayerNorm(
                    normalized_shape=out_channels, eps=eps
                )
                self.residual_conv = nn.Sequential(
                    self.activation(),
                    nn.Conv1d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding="same",
                    ),
                    nn.Dropout(dropout),
                )
                self.scale = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        """The forward pass

        Args:
            x (Tensor of shape: (N, C, L1)): input tensor

        Returns:
            Tensor of shape: (N, C, L2)
        """
        if self.is_initial:
            # (N, C, L1) --> (N, C, L2)
            x = self.conv(x)

            if self.residual:
                # (N, C, L) --> (N, L, C)
                xt = x.transpose(1, 2)
                # (N, L, C) --> (N, L, C)
                xt = self.residual_layernorm(xt)
                # (N, L, C) --> (N, C, L)
                xt = xt.transpose(1, 2)
                # scale in the channel dimension
                x += self.residual_conv(xt) * self.scale[None, :, None]

            return x

        # (N, C, L) --> (N, L, C)
        x = x.transpose(1, 2)
        # (N, L, C) --> (N, L, C)
        x = self.layernorm(x)
        # (N, L, C) --> (N, C, L)
        x = x.transpose(1, 2)
        # (N, C, L1) --> (N, C, L2)
        x = self.conv(x)

        if self.residual:
            # (N, C, L) --> (N, L, C)
            xt = x.transpose(1, 2)
            # (N, L, C) --> (N, L, C)
            xt = self.residual_layernorm(xt)
            # (N, L, C) --> (N, C, L)
            xt = xt.transpose(1, 2)
            # scale in the channel dimension
            x += self.residual_conv(xt) * self.scale[None, :, None]

        return self.pool(x)


class DenseBlock(nn.Module):
    """Dense block for the penultimate layer"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bn_momentum: float,
        dropout: float,
    ):
        super().__init__()
        self.dense = nn.Sequential(
            # (N, C) --> (N, C), normalize across the N dimension
            # BatchNorm momentum in Pytorch is different from that in Tensorflow
            nn.BatchNorm1d(in_features, momentum=1 - bn_momentum),
            nn.ReLU(),
            nn.Linear(in_features, out_features),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """The forward pass

        Args:
            x: input tensor of shape: (N, C)

        Returns:
            Tensor of shape: (N, C)
        """
        # (N, C) --> (N, C)
        return self.dense(x)


class Saluki(pl.LightningModule):
    """Modified Saluki that predicts TE"""

    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()  # required for logging params by MLFlowLogger
        for k, v in self.hparams.items():
            setattr(self, k, v)
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # number of channels of initial convolution
        seq_channels = 4 * 3 if self.split_utr5_cds_utr3_channels else 4
        label_channels = (
            self.label_codons
            + self.label_utr5
            + self.label_utr3
            + self.label_splice_sites
            + self.label_up_probs
        )
        input_channels = seq_channels + label_channels

        # initial convolution
        self.initial_conv = ConvBlock(
            in_channels=input_channels,
            out_channels=self.filters,
            kernel_size=self.kernel_size,
            stride=self.conv_stride,
            padding=self.conv_padding,
            eps=self.ln_epsilon,
            dropout=self.dropout,
            residual=self.residual,
            is_initial=True,
        )

        # middle convolutions
        self.middle_convs = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=self.filters,
                    out_channels=self.filters,
                    kernel_size=self.kernel_size,
                    stride=self.conv_stride,
                    padding=self.conv_padding,
                    eps=self.ln_epsilon,
                    dropout=self.dropout,
                    residual=self.residual,
                    is_initial=False,
                )
                for _ in range(self.num_conv_layers)
            ]
        )

        # RNN
        if self.pad_5_prime:
            if self.rnn_type == "lstm":
                self.rnn_left = nn.LSTM(
                    input_size=self.filters, hidden_size=self.filters, batch_first=True
                )
                self.rnn_right = nn.LSTM(
                    input_size=self.filters, hidden_size=self.filters, batch_first=True
                )
            else:
                self.rnn_left = nn.GRU(
                    input_size=self.filters, hidden_size=self.filters, batch_first=True
                )
                self.rnn_right = nn.GRU(
                    input_size=self.filters, hidden_size=self.filters, batch_first=True
                )
        else:
            if self.rnn_type == "lstm":
                self.rnn = nn.LSTM(
                    input_size=self.filters, hidden_size=self.filters, batch_first=True
                )
            else:
                self.rnn = nn.GRU(
                    input_size=self.filters, hidden_size=self.filters, batch_first=True
                )

        # penultimate
        if self.pad_5_prime:
            self.dense = DenseBlock(
                self.filters * 2,
                self.filters,
                self.bn_momentum,
                self.dropout,
            )
        else:
            self.dense = DenseBlock(
                self.filters,
                self.filters,
                self.bn_momentum,
                self.dropout,
            )

        # final representation
        self.final = nn.Sequential(
            nn.BatchNorm1d(self.filters, momentum=1 - self.bn_momentum),
            nn.ReLU(),
            nn.Linear(self.filters, self.num_targets),
        )

    def forward(self, x):
        """The forward pass

        Args:
            x: input tensor of shape: (N, C, L)

        Returns:
            Tensor of shape: (N, 1)
        """
        # augmentation
        # (N, C, L) --> (N, C, L)
        if self.max_shift > 0:
            x = self._stochastic_shift(x)

        # initial convolutions
        # (N, C1, L1) --> (N, C2, L2)
        x = self.initial_conv(x)

        # middle convolutions
        # (N, C1, L1) --> (N, C2, L2)
        for l in self.middle_convs:
            x = l(x)

        # (N, C, L) --> (N, L, C)
        x = x.transpose(1, 2)

        if self.pad_5_prime:
            # 971 is the max 5'UTR size in the data set.
            # start_codon_position = 971 // 2**self.num_conv_layers
            start_codon_position = self.start_codon_position_after_ConvBlocks
            # # x_left: 5'UTR + partial CDS
            # # x_right: CDS + 3'UTR
            x_left = x[:, : start_codon_position + 1, :]
            x_right = x[:, start_codon_position:, :]

            # (N, 0:L-1, C) --> (N, L-1:0, C)
            x_right = x_right.flip(1)

            # RNN from 5' to 3'
            # (N, L, C) --> (1, N, C)
            _, h_left = self.rnn_left(x_left)
            # RNN from 3' to 5'
            # (N, L, C) --> (1, N, C)
            _, h_right = self.rnn_right(x_right)

            h = torch.concat([h_left.squeeze(), h_right.squeeze()], dim=1)

        else:
            # (N, 0:L-1, C) --> (N, L-1:0, C)
            if self.go_backwards:
                x = x.flip(1)

            # RNN from 3' to 5'
            # (N, L, C) --> (1, N, C)
            _, h = self.rnn(x)

            # dim=0 is required when there is only one sample in the batch
            h = h.squeeze(dim=0)

        # penultimate
        # (N, C) --> (N, C)
        h = self.dense(h)

        # final representation
        # (N, C) --> (N, 1)
        return self.final(h)

    def training_step(self, train_batch: torch.Tensor, batch_idx):
        """Computes training loss for one batch"""
        x, y = train_batch
        output = self.forward(x)
        loss = F.mse_loss(output, y)
        self.log("train_loss", loss)
        return {"loss": loss}

    def _stochastic_shift(self, x: torch.Tensor):
        """Stochastically shift a batch of one hot encoded DNA sequences.
        x.shape: (N, C, L)
        """
        if self.max_shift == 0:
            return x

        if self.symmetric_shift:
            shifts = torch.arange(
                -self.max_shift, self.max_shift + 1, dtype=torch.int64
            )
        else:
            shifts = torch.arange(0, self.max_shift + 1, dtype=torch.int64)

        index = torch.randint(0, len(shifts), size=(1,)).item()
        shift = shifts[index].item()

        return self._shift(x, shift)

    def _shift(self, x: torch.Tensor, shift: int):
        """Simply shift a batch of one hot encoded DNA sequences by 'shift'.
        x.shape: (N, C, L)
        """
        if shift == 0:
            return x

        padding = torch.zeros((x.shape[0], x.shape[1], abs(shift))).to(self.device)
        if shift < 0:
            # shift to the left
            x = torch.concat([x[:, :, -shift:], padding], axis=2)
        else:
            # shift to the right
            x = torch.concat([padding, x[:, :, :-shift]], axis=2)
        return x

    def _compute_output(self, x: Tuple[torch.Tensor], max_shift: int = 3):
        """Computes output for one batch after data augmentation"""

        outputs = []
        max_shift = max(max_shift, self.max_shift)
        for shift in range(-max_shift, max_shift + 1):
            shifted_x = self._shift(x, shift)
            outputs.append(self.forward(shifted_x))

        output = torch.stack(outputs, axis=-1).mean(axis=-1)

        return output

    def validation_step(self, val_batch, batch_idx):
        """Computes test metrics for one batch"""

        x, y = val_batch
        output = self._compute_output(x)
        loss = F.mse_loss(output, y)

        self.validation_step_outputs.append(
            {
                "loss": loss,
                "output": output,
                "target": y,
            }
        )

    def test_step(self, test_batch, batch_idx):
        """Computes test metrics for one batch"""

        x, y = test_batch
        output = self._compute_output(x)
        loss = F.mse_loss(output, y)

        self.test_step_outputs.append(
            {
                "loss": loss,
                "output": output,
                "target": y,
            }
        )

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self._compute_output(x)

    def on_validation_epoch_end(self):
        """Log average validation metrics"""

        output = torch.cat(
            [x["output"] for x in self.validation_step_outputs], axis=0
        ).squeeze()
        target = torch.cat(
            [x["target"] for x in self.validation_step_outputs], axis=0
        ).squeeze()
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()  # free memory

        self.log("val_loss", avg_loss, sync_dist=True)

        # Calculate Pearson's correlation coefficient for each column
        avg_pearson = pearson_corrcoef(output, target).mean()
        self.log("val_pearson", avg_pearson, sync_dist=True)

        # Calculate Spearman's correlation coefficient for each column
        avg_spearman = spearman_corrcoef(output, target).mean()
        self.log("val_spearman", avg_spearman, sync_dist=True)

        # Calculate R2 for each column
        avg_r2 = r2_score(output, target).mean()
        self.log("val_r2", avg_r2, sync_dist=True)

    def on_test_epoch_end(self):
        """Log average test metrics"""

        output = torch.cat(
            [x["output"] for x in self.test_step_outputs], axis=0
        ).squeeze()
        target = torch.cat(
            [x["target"] for x in self.test_step_outputs], axis=0
        ).squeeze()
        avg_loss = torch.stack([x["loss"] for x in self.test_step_outputs]).mean()
        self.test_step_outputs.clear()  # free memory

        self.log("test_loss", avg_loss, sync_dist=True)

        avg_pearson = pearson_corrcoef(output, target).mean()
        self.log("test_pearson", avg_pearson, sync_dist=True)

        avg_spearman = spearman_corrcoef(output, target).mean()
        self.log("test_spearman", avg_spearman, sync_dist=True)

        avg_r2 = r2_score(output, target).mean()
        self.log("test_r2", avg_r2, sync_dist=True)


    def configure_optimizers(self):
        """Initializes the optimizer and learning rate scheduler"""

        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.l2_scale,
        )
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.25,
                patience=8,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [self.optimizer], [self.scheduler]


class RiboNN(pl.LightningModule):
    """CNN model that predicts TEs in multiple samples using the same input"""

    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()  # required for logging params by MLFlowLogger
        self.activation = nn.ReLU
        for k, v in self.hparams.items():
            if k == "activation":
                if v.lower() in ("silu", "swish"):
                    v = nn.SiLU
                elif v.lower() == "leakyrelu":
                    v = nn.LeakyReLU
                else:
                    continue
            setattr(self, k, v)

        self.validation_step_outputs = []
        self.test_step_outputs = []
        if self.with_NAs:
            self.loss = masked_mse_loss
        else:
            self.loss = F.mse_loss

        # number of channels of initial convolution
        seq_channels = 4 * 3 if self.split_utr5_cds_utr3_channels else 4
        label_channels = (
            self.label_codons
            + self.label_utr5
            + self.label_utr3
            + self.label_splice_sites
            + self.label_up_probs
        )
        input_channels = seq_channels + label_channels

        # shared initial convolution
        self.initial_conv = ConvBlock(
            in_channels=input_channels,
            out_channels=self.filters,
            kernel_size=5,
            stride=self.conv_stride,
            padding=self.conv_padding,
            eps=self.ln_epsilon,
            dropout=self.dropout,
            residual=self.residual,
            is_initial=True,
            activation=self.activation,
        )

        # shared middle convolutions
        self.middle_convs = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=self.filters,
                    out_channels=self.filters,
                    kernel_size=self.kernel_size,
                    stride=self.conv_stride,
                    padding=self.conv_padding,
                    eps=self.ln_epsilon,
                    dropout=self.dropout,
                    residual=self.residual,
                    is_initial=False,
                    activation=self.activation,
                )
                for i in range(self.num_conv_layers)
            ]
        )

        self.head = nn.Sequential(
            self.activation(),
            # nn.AdaptiveMaxPool1d(8),
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(self.filters * self.len_after_conv, self.filters, bias=False),
            self.activation(),
            nn.BatchNorm1d(self.filters),
            nn.Dropout(self.dropout),
            nn.Linear(self.filters, self.num_targets),
        )

    def forward(self, x):
        """The forward pass

        Args:
            x: input tensor of shape: (N, C, L)
            task_idx: task index

        Returns:
            Tensor of shape (N, self.num_targets)
        """
        # augmentation
        # (N, C, L) --> (N, C, L)
        if self.max_shift > 0:
            x = self._stochastic_shift(x)

        # initial convolutions
        # (N, C1, L1) --> (N, C2, L2)
        x = self.initial_conv(x)

        # middle convolutions
        # (N, C1, L1) --> (N, C2, L2)
        for l in self.middle_convs:
            x = l(x)

        x = self.head(x)

        return x

    def training_step(self, train_batch: torch.Tensor, batch_idx):
        """Computes training loss for one batch"""
        x, y = train_batch
        output = self.forward(x)
        loss = self.loss(output, y)
        self.log("train_loss", loss)
        return {"loss": loss}


    def _stochastic_shift(self, x: torch.Tensor):
        """Stochastically shift a batch of one hot encoded DNA sequences.
        x.shape: (N, C, L)
        """
        if self.max_shift == 0:
            return x

        if self.symmetric_shift:
            shifts = torch.arange(
                -self.max_shift, self.max_shift + 1, dtype=torch.int64
            )
        else:
            shifts = torch.arange(0, self.max_shift + 1, dtype=torch.int64)

        index = torch.randint(0, len(shifts), size=(1,)).item()
        shift = shifts[index].item()

        return self._shift(x, shift)

    def _shift(self, x: torch.Tensor, shift: int):
        """Simply shift a batch of one hot encoded DNA sequences by 'shift'.
        x.shape: (N, C, L)
        """
        if shift == 0:
            return x

        padding = torch.zeros((x.shape[0], x.shape[1], abs(shift))).to(self.device)
        if shift < 0:
            # shift to the left
            x = torch.concat([x[:, :, -shift:], padding], axis=2)
        else:
            # shift to the right
            x = torch.concat([padding, x[:, :, :-shift]], axis=2)

        return x

    def _compute_output(self, x: Tuple[torch.Tensor]):
        """Computes metrics for one batch"""

        outputs = []
        for shift in range(-self.max_shift, self.max_shift + 1):
            shifted_x = self._shift(x, shift)
            outputs.append(self.forward(shifted_x))

        output = torch.stack(outputs, axis=-1).mean(axis=-1)

        return output

    def validation_step(self, val_batch, batch_idx):
        """Computes test metrics for one batch"""

        x, y = val_batch
        output = self._compute_output(x)
        # loss = F.mse_loss(output, y)
        loss = self.loss(output, y)

        self.validation_step_outputs.append(
            {
                "loss": loss,
                "output": output,
                "target": y,
            }
        )

    def test_step(self, test_batch, batch_idx):
        """Computes test metrics for one batch"""

        x, y = test_batch
        output = self._compute_output(x)
        loss = self.loss(output, y)

        self.test_step_outputs.append(
            {
                "loss": loss,
                "output": output,
                "target": y,
            }
        )

    def predict_step(self, x, batch_idx):
        return self._compute_output(x)

    def on_validation_epoch_end(self):
        """Log average validation metrics"""

        if not self.validation_step_outputs:
            print("    [skip] validation_step_outputs is empty", flush=True)
            return

        output = torch.cat(
            [x["output"] for x in self.validation_step_outputs], axis=0
        ).squeeze()
        target = torch.cat(
            [x["target"] for x in self.validation_step_outputs], axis=0
        ).squeeze()
        losses = [x["loss"] for x in self.validation_step_outputs if not x["loss"].isnan()]
        self.validation_step_outputs.clear()  # free memory

        avg_loss = torch.stack(losses).mean() if losses else torch.tensor(float('nan'))
        self.log("val_loss", avg_loss, sync_dist=True)

        if target.dim() == 1:
            mask = ~target.isnan()
            avg_pearson = pearson_corrcoef(output[mask], target[mask]).mean()
            avg_spearman = spearman_corrcoef(output[mask], target[mask]).mean()
            avg_r2 = r2_score(output[mask], target[mask]).mean()
        else:
            pearsons, spearmans, r2s = [], [], []
            for c in range(target.shape[1]):
                col_mask = ~target[:, c].isnan()
                if col_mask.sum() < 10:
                    continue
                pearsons.append(pearson_corrcoef(output[col_mask, c], target[col_mask, c]))
                spearmans.append(spearman_corrcoef(output[col_mask, c], target[col_mask, c]))
                r2s.append(r2_score(output[col_mask, c], target[col_mask, c]))
            avg_pearson = torch.stack(pearsons).mean() if pearsons else torch.tensor(float('nan'))
            avg_spearman = torch.stack(spearmans).mean() if spearmans else torch.tensor(float('nan'))
            avg_r2 = torch.stack(r2s).mean() if r2s else torch.tensor(float('nan'))

        self.log("val_pearson", avg_pearson, sync_dist=True)
        self.log("val_spearman", avg_spearman, sync_dist=True)
        self.log("val_r2", avg_r2, sync_dist=True)

    def on_test_epoch_end(self):
        """Log average test metrics"""

        output = torch.cat(
            [x["output"] for x in self.test_step_outputs], axis=0
        ).squeeze()
        target = torch.cat(
            [x["target"] for x in self.test_step_outputs], axis=0
        ).squeeze()
        avg_loss = torch.stack([x["loss"] for x in self.test_step_outputs]).mean()
        self.test_step_outputs.clear()  # free memory

        self.log("test_loss", avg_loss, sync_dist=True)

        mask = target.isnan()
        avg_pearson = pearson_corrcoef(output[~mask], target[~mask]).mean()
        self.log("test_pearson", avg_pearson, sync_dist=True)

        avg_spearman = spearman_corrcoef(output[~mask], target[~mask]).mean()
        self.log("test_spearman", avg_spearman, sync_dist=True)

        avg_r2 = r2_score(output[~mask], target[~mask]).mean()
        self.log("test_r2", avg_r2, sync_dist=True)


    def configure_optimizers(self):
        """Initializes the optimizer and learning rate scheduler"""

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.l2_scale,
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=8,
                min_lr=self.min_lr,
                verbose=True,
            ),
            "monitor": "val_loss",
        }

        return [optimizer], [scheduler]


class MeanTE(RiboNN):
    """Model that uses the multi-task model to predict mean TE.
    This model is only used for prediction or feature attribution.
    """

    def __init__(self, pretrained_multi_task_model: pl.LightningModule, **hparams):
        super().__init__(**hparams)
        state_dict = pretrained_multi_task_model.state_dict()
        self.load_state_dict(state_dict, strict=False)
        self.freeze()
        self.eval()

    def forward(self, x):
        """The forward pass

        Args:
            x: input tensor of shape: (N, C, L)
            task_idx: task index

        Returns:
            Tensor of shape (N, 1)
        """
        # augmentation
        # (N, C, L) --> (N, C, L)
        if self.max_shift > 0:
            x = self._stochastic_shift(x)

        # initial convolutions
        # (N, C1, L1) --> (N, C2, L2)
        x = self.initial_conv(x)

        # middle convolutions
        # (N, C1, L1) --> (N, C2, L2)
        for l in self.middle_convs:
            x = l(x)

        x = self.head(x)

        return x.mean(-1, keepdim=True)
