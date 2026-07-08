# UTR-LM baseline

Vendored code for the UTR-LM baseline. Pretrained weights are **not** included
and must be obtained from the upstream UTR-LM repository.

## Setup

The model code expects a `params/` directory here containing the upstream
UTR-LM repository (its `Model/Pretrained/` subdirectory holds the checkpoint):

```
benchmarks/baselines/UTR-LM/params/
└── Model/
    └── Pretrained/
        └── ESM2SI_3.1_fiveSpeciesCao_6layers_16heads_128embedsize_4096batchToks_MLMLossMin.pkl
```

Obtain it from the upstream project (https://github.com/a96123155/UTR-LM) and
place or symlink it as `params/` in this directory, e.g.:

```bash
$ git clone https://github.com/a96123155/UTR-LM params
```

The benchmark runner (`run.py`) then loads the pretrained checkpoint via
`model_te.py` / `model_mrl.py`.
