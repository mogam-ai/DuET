#!/usr/bin/env python3
"""Check overlap between outer test split and saved RiboNN inner splits."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fold_dir")
    parser.add_argument("--inner-file", default="ribonn_inner_indices.json")
    args = parser.parse_args()

    fold_dir = Path(args.fold_dir)
    outer = json.loads((fold_dir / "indices.json").read_text())
    inner = json.loads((fold_dir / args.inner_file).read_text())

    test = set(outer["test"])
    trainval_ordered = list(outer["train"]) + list(outer["val"])
    trainval = set(trainval_ordered)

    print("outer sizes:", {key: len(value) for key, value in outer.items()})
    print("inner_cv_folds:", inner["inner_cv_folds"])

    for i, split in enumerate(inner["folds"]):
        train_local = set(split["train"])
        val_local = set(split["val"])
        train = {trainval_ordered[idx] for idx in train_local}
        val = {trainval_ordered[idx] for idx in val_local}
        print(
            "inner %d: train=%d val=%d train_val_overlap=%d "
            "train_test_overlap=%d val_test_overlap=%d union_ok=%s"
            % (
                i,
                len(train),
                len(val),
                len(train & val),
                len(train & test),
                len(val & test),
                (train | val) == trainval,
            )
        )


if __name__ == "__main__":
    main()
