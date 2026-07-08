# Program: DuET v1.1.0
# Author: Sungho Lee, Jae-Won Lee
# Affiliation: MOGAM Institute for Biomedical Research
# Contact: https://github.com/mogam-ai/DuET/issues
# Citation: TBD

"""DuET: A Unified Deep Learning Framework for Predicting mRNA Translation Efficiency."""

__version__ = "1.0.0"

from .api import predict, train

__all__ = ["predict", "train", "__version__"]
