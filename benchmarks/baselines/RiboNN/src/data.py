from typing import Dict, Literal, Optional

import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

STAGE = Literal["train", "valid", "test", "predict", "attribute"]


class DataFrameDataset(Dataset):
    """Dataset for turning transcript features and TEs into tensors"""

    def __init__(
        self,
        df: pd.DataFrame,
        target_column_pattern: Optional[str],
        max_utr5_len: int,
        max_cds_utr3_len: int,
        max_tx_len: int,
        pad_5_prime: bool,
        split_utr5_cds_utr3_channels: bool,
        label_codons: bool,
        label_3rd_nt_of_codons: bool,
        label_utr5: bool,
        label_utr3: bool,
        label_splice_sites: bool,
        label_up_probs: bool,
        pad_cds: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        if target_column_pattern:
            self.target_column_pattern = target_column_pattern
            if self.target_column_pattern in self.df.columns:
                self.targets = self.df[[self.target_column_pattern]]
            else:
                self.targets = self.df.filter(regex=rf"{self.target_column_pattern}")
        else:
            self.targets = None
        self.pad_5_prime = pad_5_prime
        self.split_utr5_cds_utr3_channels = split_utr5_cds_utr3_channels
        self.label_codons = label_codons
        self.label_3rd_nt_of_codons = label_3rd_nt_of_codons
        self.label_utr5 = label_utr5
        self.label_utr3 = label_utr3
        self.label_splice_sites = label_splice_sites
        self.label_up_probs = label_up_probs
        self.pad_cds = pad_cds

        # Calculate number of input channels
        self.seq_channels = 4 * 3 if self.split_utr5_cds_utr3_channels else 4
        self.label_channels = (
            self.label_codons
            + self.label_utr5
            + self.label_utr3
            + self.label_splice_sites
            + self.label_up_probs
        )

        # Calculate sizes for aligning sequences at the start codon.
        if pad_5_prime:
            # Pad the 5' UTRs
            self.padded_utr5_len = max_utr5_len
            # Pad the 3' UTRs.
            self.padded_tx_len = max_utr5_len + max_cds_utr3_len
        # Calculate sizes for aligning sequences at the 5' end.
        else:
            # Pad the 3' UTRs.
            self.padded_tx_len = max_tx_len

        self.base_index = {
            "A": 0,
            "T": 1,
            "C": 2,
            "G": 3,
        }

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        # Initiate a tensor to save encodings for sequence, labels, etc.
        x = torch.zeros(
            (self.seq_channels + self.label_channels, self.padded_tx_len),
            dtype=torch.float32,
        )

        original_tx_seq = self.df.tx_sequence[i]
        original_utr5_len = self.df.utr5_size[i]
        original_cds_len = self.df.cds_size[i]
        original_tx_len = self.df.tx_size[i]
        if self.label_splice_sites:
            ss = self.df.splice_sites.values[i]
        if self.label_up_probs:
            probs = self.df.up_prob.values[i].split(";")[:original_tx_len]

        # Encode sequences
        if (
            self.pad_5_prime
        ):  # Padding the 5' end and align sequences at the start codon.
            if self.split_utr5_cds_utr3_channels:
                # Encode 5'UTR
                for idx, nt in enumerate(original_tx_seq[:original_utr5_len]):
                    x[
                        self.base_index[nt],
                        self.padded_utr5_len - original_utr5_len + idx,
                    ] = 1
                # Encode CDS
                for idx, nt in enumerate(
                    original_tx_seq[
                        original_utr5_len : original_utr5_len + original_cds_len
                    ]
                ):
                    x[self.base_index[nt] + 4, self.padded_utr5_len + idx] = 1
                # Encode 3'UTR
                for idx, nt in enumerate(
                    original_tx_seq[
                        original_utr5_len + original_cds_len : original_tx_len
                    ]
                ):
                    x[
                        self.base_index[nt] + 8,
                        self.padded_utr5_len + original_cds_len + idx,
                    ] = 1
            else:
                # Encode the entire transcript in 4 channels
                for idx, nt in enumerate(original_tx_seq):
                    x[
                        self.base_index[nt],
                        self.padded_utr5_len - original_utr5_len + idx,
                    ] = 1

            # Encode labels
            row_index = self.seq_channels
            if self.label_utr5:
                x[
                    row_index,
                    self.padded_utr5_len - original_utr5_len : self.padded_utr5_len,
                ] = 1
                row_index += 1

            if self.label_codons:
                start = self.padded_utr5_len
                stop = start + original_cds_len - 3
                for idx in range(start, stop + 1, 3):
                    x[row_index, idx] = 1
                row_index += 1

            if self.label_utr3:
                x[
                    row_index,
                    (self.padded_utr5_len + original_cds_len) : (
                        self.padded_utr5_len - original_utr5_len + original_tx_len
                    ),
                ] = 1
                row_index += 1

            if self.label_splice_sites:
                if isinstance(
                    ss, str
                ):  # Empty strings are turned into np.nan, which is a float
                    ss = ss.split(";")
                    for idx in ss:
                        x[
                            row_index,
                            self.padded_utr5_len - original_utr5_len + int(idx) - 1,
                        ] = 1
                row_index += 1

            # Encode structure
            if self.label_up_probs:
                for idx, prob in enumerate(probs):
                    x[row_index, self.padded_utr5_len - original_utr5_len + idx - 1] = (
                        float(prob)
                    )

        else:  # Not padding the 5' end. Simply align sequences at the 5' end.
            if self.split_utr5_cds_utr3_channels:
                # Encode 5'UTR
                for idx, nt in enumerate(original_tx_seq[:original_utr5_len]):
                    x[self.base_index[nt], idx] = 1
                # Encode CDS
                for idx, nt in enumerate(
                    original_tx_seq[
                        original_utr5_len : original_utr5_len + original_cds_len
                    ]
                ):
                    x[self.base_index[nt] + 4, original_utr5_len + idx] = 1
                # Encode 3'UTR
                for idx, nt in enumerate(
                    original_tx_seq[
                        original_utr5_len + original_cds_len : original_tx_len
                    ]
                ):
                    x[
                        self.base_index[nt] + 8,
                        original_utr5_len + original_cds_len + idx,
                    ] = 1
            else:
                # Encode the entire transcript in 4 channels
                for idx, nt in enumerate(original_tx_seq):
                    x[self.base_index[nt], idx] = 1

            # Encode labels
            row_index = self.seq_channels
            if self.label_utr5:
                x[row_index, :original_utr5_len] = 1
                row_index += 1

            if self.label_codons:
                start = original_utr5_len
                stop = start + original_cds_len - 3
                if self.label_3rd_nt_of_codons:
                    for idx in range(start + 2, stop + 3, 3):
                        x[row_index, idx] = 1
                else:
                    for idx in range(start, stop + 1, 3):
                        x[row_index, idx] = 1
                row_index += 1

            if self.label_utr3:
                x[
                    row_index,
                    (original_utr5_len + original_cds_len) : original_tx_len,
                ] = 1
                row_index += 1

            if self.label_splice_sites:
                if isinstance(
                    ss, str
                ):  # Empty strings are turned into np.nan, which is a float
                    ss = ss.split(";")
                    for idx in ss:
                        x[row_index, int(idx) - 1] = 1
                row_index += 1

            # Encode structure
            if self.label_up_probs:
                for idx, prob in enumerate(probs):
                    x[row_index, idx - 1] = float(prob)

        if self.targets is None:
            return x

        # Encode targets
        y = self.targets.loc[i, :].values
        y = torch.from_numpy(y).float()

        return x, y


class RiboNNDataModule(pl.LightningDataModule):
    """DataModule for the RiboNN model"""

    def __init__(self, config: Dict):
        super().__init__()
        for k, v in config.items():
            setattr(self, k, v)

        if self.tx_info_path is None:
            self.df = pd.DataFrame()  # will be set via setup_from_df()
            return

        self.df = pd.read_csv(self.tx_info_path, delimiter="\t")
        self._prepare_df(config)

    def setup_from_df(self, df, config):
        """Set df externally and run the same preparation as __init__."""
        self.df = df
        self._prepare_df(config)

    def _prepare_df(self, config):
        if "utr5_sequence" in self.df.columns and "cds_sequence" in self.df.columns and "utr3_sequence" in self.df.columns:
            self.df.utr5_sequence = (
                self.df.utr5_sequence.str.strip().str.upper().str.replace("U", "T")
            )
            self.df.cds_sequence = (
                self.df.cds_sequence.str.strip().str.upper().str.replace("U", "T")
            )
            self.df.utr3_sequence = (
                self.df.utr3_sequence.str.strip().str.upper().str.replace("U", "T")
            )
            self.df['tx_sequence'] = self.df.utr5_sequence + self.df.cds_sequence + self.df.utr3_sequence
            self.df["tx_size"] = self.df.tx_sequence.str.len()
            self.df["utr5_size"] = self.df.utr5_sequence.str.len()
            self.df["cds_size"] = self.df.cds_sequence.str.len()
            self.df["utr3_size"] = self.df.utr3_sequence.str.len()
        else:
            assert "tx_sequence" in self.df.columns
            assert "utr5_size" in self.df.columns
            assert "cds_size" in self.df.columns
            self.df.tx_sequence = (
                self.df.tx_sequence.str.strip().str.upper().str.replace("U", "T")
            )
            self.df["utr5_sequence"] = self.df.apply(
                lambda row: row.tx_sequence[:row.utr5_size],
                axis = 1
            )
            self.df["cds_sequence"] = self.df.apply(
                lambda row: row.tx_sequence[row.utr5_size : row.utr5_size + row.cds_size],
                axis = 1
            )
            self.df["utr3_sequence"] = self.df.apply(
                lambda row: row.tx_sequence[row.utr5_size + row.cds_size : ],
                axis = 1
            )
            self.df["tx_size"] = self.df.tx_sequence.str.len()
            self.df["utr3_size"] = self.df.tx_size - self.df.utr5_size - self.df.cds_size

        # Validate cds_sequence
        if not self.df.cds_size.apply(lambda x: x % 3 == 0).all():
            print("WARNING: Not all CDS sequences have a length that is a multiple of 3!")
        if not self.df.cds_sequence.apply(lambda x: x[-3:] in ("TAA", "TGA", "TAG")).all():
            print("WARNING: Not all CDS sequences end with a stop codon (stop codon may be excluded from annotation).")

        if "tx_id" not in self.df.columns:
            self.df["tx_id"] = range(len(self.df))

        if self.df.tx_size.max() > self.max_seq_len:
            n_removed = (self.df.tx_size > self.max_seq_len).sum()
            print(
                f"Transcripts longer than {self.max_seq_len} in the input data are removed! (n={n_removed})"
            )
            self.df = self.df.query("tx_size <= @self.max_seq_len").reset_index(drop=True)

        if self.remove_extreme_txs:
            utr5_len_max = self.df.utr5_size.quantile(0.99)
            cds_len_max = self.df.cds_size.quantile(0.99)
            utr3_len_max = self.df.utr3_size.quantile(0.99)
            print(
                "Top 1% of transcripts with extreme 5'UTR, CDS, or 3'UTR sizes in the input data are removed!",
                "To keep them, set remove_extreme_txs in config/conf.yml to False.",
            )
            self.df = self.df.query(
                "utr5_size <= @utr5_len_max and cds_size <= @cds_len_max and utr3_size <= @utr3_len_max"
            ).reset_index(drop=True)

        if config.get("max_utr5_len", np.Inf) < self.df["utr5_size"].max():
            print(
                f"Transcripts with 5'UTRs longer than {config['max_utr5_len']} in the input data are removed!"
            )
            self.df = self.df.query("utr5_size <= @config['max_utr5_len']").reset_index(
                drop=True
            )
        self.max_utr5_len = config["max_utr5_len"]
        config["max_utr5_len"] = self.max_utr5_len

        self.df["cds_utr3_len"] = self.df.apply(
            lambda row: row.tx_size - row.utr5_size, 1
        )
        if config.get("max_cds_utr3_len", np.Inf) < self.df["cds_utr3_len"].max():
            print(
                f"Transcripts with combined CDS and 3'UTR sizes more than {config['max_cds_utr3_len']} in the input data are removed!"
            )
            self.df = self.df.query(
                "cds_utr3_len <= @config['max_cds_utr3_len']"
            ).reset_index(drop=True)
        self.max_cds_utr3_len = config["max_cds_utr3_len"]
        config["max_cds_utr3_len"] = self.max_cds_utr3_len

        self.max_tx_len = self.df.tx_size.max()

        if self.target_column_pattern:
            targets = self.df.filter(regex=rf"{self.target_column_pattern}")
            self.num_targets = targets.shape[1]
            self.target_names = targets.columns.tolist()
            self.df = self.df.loc[~targets.isnull().all(1), :]

    def get_sequence_length_after_ConvBlocks(self):
        """Calculate the sequence length after the ConvBlocks"""
        # Input sequence length
        if self.pad_5_prime:
            seq_len = self.max_utr5_len + self.max_cds_utr3_len
        else:
            seq_len = self.max_tx_len

        # Sequence length after initial Conv1d
        seq_len = (
            seq_len + 2 * self.conv_padding - self.conv_dilation * (5 - 1) - 1
        ) // self.conv_stride + 1

        for _ in range(self.num_conv_layers):
            # Sequence length after Conv1d
            seq_len = (
                seq_len
                + 2 * self.conv_padding
                - self.conv_dilation * (self.kernel_size - 1)
                - 1
            ) // self.conv_stride + 1
            # Sequence length after MaxPool1d
            seq_len = (seq_len + 2 * 0 - 1 * (2 - 1) - 1) // 2 + 1

        return seq_len

    def make_dataloader(
        self,
        stage: STAGE = "train",
        batch_size: int = 32,
        drop_last: bool = False,
    ):
        if stage in ("attribute", "predict"):
            input_df = self.df
            self.target_column_pattern = None
        else:
            input_df = self.df.query("split == @stage")
        dataset = DataFrameDataset(
            input_df,
            self.target_column_pattern,
            self.max_utr5_len,
            self.max_cds_utr3_len,
            self.max_tx_len,
            self.pad_5_prime,
            self.split_utr5_cds_utr3_channels,
            self.label_codons,
            self.label_3rd_nt_of_codons,
            self.label_utr5,
            self.label_utr3,
            self.label_splice_sites,
            self.label_up_probs,
        )

        reorder = stage == "train"

        return DataLoader(
            dataset,
            batch_size,
            shuffle=reorder,
            num_workers=self.num_workers,
            drop_last=drop_last,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        """Training stage data loader"""
        return self.make_dataloader("train", self.train_batch_size, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """Validation stage data loader"""
        return self.make_dataloader("valid", self.val_batch_size)

    def test_dataloader(self) -> DataLoader:
        """Testing stage data loader"""
        return self.make_dataloader("test", self.test_batch_size)

    def attri_dataloader(self) -> DataLoader:
        """Attribution stage data loader"""
        return self.make_dataloader("attribute", self.test_batch_size)

    def predict_dataloader(self) -> DataLoader:
        """Prediction stage data loader"""
        return self.make_dataloader("predict", self.test_batch_size)
