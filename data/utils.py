# Program:     DuET v1.0.0
# Author:      Sungho Lee, Jae-Won Lee, Jinhee Hong
# Affiliation: MOGAM Institute for Biomedical Research
# Contact:     sungho.lee@mogam.re.kr; jaewon.lee@mogam.re.kr; jinhee.hong@mogam.re.kr
# Citation:    TBD


import torch
import torch.nn.functional as F

from typing import Optional, Callable


BASE_CODES: dict[str, int] = {
    "A": 0,
    "G": 1,
    "C": 2,
    "T": 3,
    "U": 3,  # Treat 'U' as 'T'
    'P': 4,  # Pseudouridine
    '1': 5,  # m1-pseudouridine
    'N': 6,
}

CODON_TO_AA: dict[str, str] = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'TCT': 'S', 'TCC': 'S',
    'TCA': 'S', 'TCG': 'S', 'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W', 'CTT': 'L', 'CTC': 'L',
    'CTA': 'L', 'CTG': 'L', 'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q', 'CGT': 'R', 'CGC': 'R',
    'CGA': 'R', 'CGG': 'R', 'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'AAT': 'N', 'AAC': 'N',
    'AAA': 'K', 'AAG': 'K', 'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V', 'GCT': 'A', 'GCC': 'A',
    'GCA': 'A', 'GCG': 'A', 'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    } 

# CodonWeight was created using the values from:
# https://www.kazusa.or.jp/codon/cgi-bin/showcodon.cgi?species=9606&aa=1&style=N
# Maps codon -> weight = frequency of codon / max frequency of synonymous codons
# Only used for calculating CAI w.r.t. human codon frequencies
codon_weight = {
    'TTT': 0.85185, 'TTC': 1.0, 'TTA': 0.2, 'TTG': 0.325, 'TCT': 0.79167,
    'TCC': 0.91667, 'TCA': 0.625, 'TCG': 0.20833, 'TAT': 0.78571, 'TAC': 1.0,
    'TAA': 0.6383, 'TAG': 0.51064, 'TGT': 0.85185, 'TGC': 1.0, 'TGA': 1.0,
    'TGG': 1.0, 'CTT': 0.325, 'CTC': 0.5, 'CTA': 0.175, 'CTG': 1.0,
    'CCT': 0.90625, 'CCC': 1.0, 'CCA': 0.875, 'CCG': 0.34375, 'CAT': 0.72414,
    'CAC': 1.0, 'CAA': 0.36986, 'CAG': 1.0, 'CGT': 0.38095, 'CGC': 0.85714,
    'CGA': 0.52381, 'CGG': 0.95238, 'ATT': 0.76596, 'ATC': 1.0, 'ATA': 0.3617,
    'ATG': 1.0, 'ACT': 0.69444, 'ACC': 1.0, 'ACA': 0.77778, 'ACG': 0.30556,
    'AAT': 0.88679, 'AAC': 1.0, 'AAA': 0.75439, 'AAG': 1.0, 'AGT': 0.625,
    'AGC': 1.0, 'AGA': 1.0, 'AGG': 1.0, 'GTT': 0.3913, 'GTC': 0.52174,
    'GTA': 0.26087, 'GTG': 1.0, 'GCT': 0.675, 'GCC': 1.0, 'GCA': 0.575,
    'GCG': 0.275, 'GAT': 0.85185, 'GAC': 1.0, 'GAA': 0.72414, 'GAG': 1.0,
    'GGT': 0.47059, 'GGC': 1.0, 'GGA': 0.73529, 'GGG': 0.73529
    }

CODON_CODES: dict[str, int] = {k:v for k,v in zip(sorted(CODON_TO_AA.keys()), range(len(CODON_TO_AA)))}
CODON_CODES["N"] = 64  # N/A codon default value

unique_aa = sorted(set(CODON_TO_AA.values()))
AA_CODES: dict[str, int] = {k:v for k,v in zip(unique_aa + ["X"], range(len(unique_aa) + 1))}

LSTM_VOCAB = {
  'N': 0,
  'A': 1,
  'C': 2,
  'G': 3,
  'T': 4,
  'U': 4,
  'P': 4,  # pU
  '1': 4,  # m1Ψ
}
PAD_TOKEN = LSTM_VOCAB['N']


def seq2tensor(seq: str, max_len: int) -> torch.Tensor:
  ids = seq2ids(LSTM_VOCAB, seq, PAD_TOKEN)
  # truncate sequence
  if len(ids) > max_len: ids = ids[-max_len:]  # keep last max_len-2 elements
  # pad sequence
  ids = ids + [PAD_TOKEN] * (max_len - len(ids))
  return torch.tensor(ids, dtype=torch.long)


def seq2ids(code: dict, seq: str, default: int=6, quantizer: Callable=None) -> list[int]:
    if quantizer is not None:
        seq = quantizer(seq)
    return [code.get(x.upper(), default) for x in seq]


def tensorize(seq: str,
              code: Optional[dict[str, int]]=BASE_CODES,
              channel_size: int=4,  # 4 (+2 pU) for base encoding, 64 for codon encoding
              **kwargs) -> torch.Tensor:
    """Encode sequences using one-hot encoding. This function do not pad the sequence.
    Pass quantizer=lambda x: [x[k:k+3] for k in range(0, len(x), 3)] and default=64 to encode codons.

    Args:
      seq (str): DNA sequence
      do_padding (bool): whether to pad the sequence
      padding_pos (Literal['left', 'right']): where to pad the sequence
      max_len (Optional[int]): maximum length of the sequence. 
        If None, the length is not restricted
      return_ids (bool): whether to return sequence as integers

    Returns:
      onehot_tensor (torch.Tensor): one-hot encoded DNA sequence.
    """
    # map nucleotides to integers
    onehot_tensor = torch.tensor(seq2ids(code, seq, **kwargs), dtype=torch.long)

    onehot_tensor = F.one_hot(onehot_tensor, num_classes=max(code.values())+1)  # 7th class is N (5 before)
    return onehot_tensor[:, :channel_size].float()  # remove N (4 before)


def Singleton(class_):
    instances = {}

    def get_instance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return get_instance