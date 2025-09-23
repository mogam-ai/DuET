# Program:     DuET v1.0.0
# Author:      Sungho Lee, Jae-Won Lee, Jinhee Hong
# Affiliation: MOGAM Institute for Biomedical Research
# Contact:     sungho.lee@mogam.re.kr; jaewon.lee@mogam.re.kr; jinhee.hong@mogam.re.kr
# Citation:    TBD


import re
import pandas as pd

from sklearn import preprocessing
from typing import Optional
from .utils import Singleton


def find_matching_columns(query: str, header: Optional[list]=None, fname: Optional[str]=None):
  """Search for matching columns in the header list or file given a query list of column names.
  Supports exact match, partial match, and wildcard search.
  Returns non-redundant list of matching column names."""
  
  if header is None:
    header = open(fname).readline().strip().split("\t")
    
  found_columns = []
  for k in query:
    if k in header:  # exact match
      found_columns.append(k)
    elif matches := [col for col in header if k in col]:  # partial match
      found_columns.extend(matches)
    elif "*" in k:  # wildcard search
      found_columns.extend([col for col in header if re.search(k.replace("*",".*"), col)])
    else:
      raise ValueError(f"Column '{k}' not found in sequence feature columns.")
    
  return list(set(found_columns))


@Singleton
class SequenceFeatureStore():
  """A singleton class to store sequence feature."""
  
  def __init__(self, data_path: str, join_col: str,
               cols_to_use: Optional[list]=["*"], data_object: Optional[pd.DataFrame]= None):
    """Initialize SequenceFeatureStore instance.
    
    Args:
      data_path (str): path to sequence feature file
      join_col (str): column to join sequence feature with data.
      cols_to_use (list, optional): list of columns to use from sequence feature. 
                                    Default: use all columns ('*')
                                    Supports exact, partial match and wildcards("*")
                                    to match columns in the sequence feature dataframe.
      data_object (pd.DataFrame, optional): pre-loaded sequence feature DataFrame
    """

    if data_object is not None:
      self.sequence_features = data_object.copy()
    else:
      self.sequence_features = pd.read_csv(data_path, sep='\t', low_memory=False)
      
    self.join_col = join_col
    join_column = self.sequence_features.pop(self.join_col)
    self.sequence_features.insert(0, self.join_col, join_column)
    
    if self.sequence_features[self.join_col].isnull().any() or self.sequence_features[self.join_col].duplicated().any():
      raise RuntimeError(f"Column {self.join_col} has null or duplicated values.")
    
    if cols_to_use == ["*"]:
      cols_to_use = self.sequence_features.columns.tolist()
    else:
      cols_to_use = find_matching_columns(cols_to_use, self.sequence_features.columns.tolist())
      if self.join_col not in cols_to_use:
        cols_to_use.insert(0, self.join_col)
      
    self.sequence_features = self.sequence_features[cols_to_use].fillna(0.0)
    self.sequence_features_cols = self.sequence_features.columns.tolist()
    
    self.sequence_features_scalers = []
    self._normalize_sequence_features()

  def get_sequence_features(self):
    """Return a copy of sequence feature dataFrame."""
    return self.sequence_features.copy()

  def get_sequence_feature_cols(self):
    """ Return sequence feature columns."""
    return self.sequence_features_cols[1:]  # skip join_col

  def merge_sequence_feature(self, data: pd.DataFrame, on: str):
    """ Merge sequence features with data."""
    
    return data.merge(self.sequence_features, on=on, how='inner')

  def drop_sequence_feature_cols(self, *to_drop):
    """ Drop specified sequence feature columns. """
    
    to_drop = set(to_drop)
    self.sequence_features = self.sequence_features.drop(columns=to_drop, errors="ignore")
    self.sequence_features_cols = [col for col in self.sequence_features_cols if col not in to_drop]

  def _normalize_sequence_features(self):
    """Normalize self.sequence_features columns and replace NaN values to 0 (average)."""

    if self.sequence_features_scalers:
      for idx, col in enumerate(self.sequence_features.columns):
        if col == self.join_col:
          continue
        
        target_column = self.sequence_features[col]
        not_na_mask = target_column.notna()

        if not_na_mask.any():
          # to suppress warning
          if pd.api.types.is_integer_dtype(self.sequence_features[col].dtype) or pd.api.types.is_object_dtype(self.sequence_features[col].dtype):
            try:
              self.sequence_features[col] = self.sequence_features[col].astype('float32') 
            except ValueError:
              print(f"Warning: Column {col} could not be converted to float. Skipping scaling.")
              continue
          not_na_values = target_column[not_na_mask].values.astype('float32').reshape(-1, 1)
          self.sequence_features.loc[not_na_mask, col] = self.sequence_features_scalers[idx].transform(not_na_values).flatten()
          self.sequence_features[col].fillna(0, inplace=True)
        else:
          raise RuntimeError(f"Every value is NA in sequence feature {col}.")

    else:
      for col in self.sequence_features.columns:
        if col == self.join_col:
          continue

        feature_scaler = preprocessing.StandardScaler()
        
        target_column = self.sequence_features[col]
        not_na_mask = target_column.notna()

        if not_na_mask.any():
          # to suppress warning
          if pd.api.types.is_integer_dtype(self.sequence_features[col].dtype) or pd.api.types.is_object_dtype(self.sequence_features[col].dtype):
            try:
              self.sequence_features[col] = self.sequence_features[col].astype('float32') 
            except ValueError:
              print(f"Warning: Column {col} could not be converted to float. Skipping scaling.")
              continue
          not_na_values = target_column[not_na_mask].values.astype('float32').reshape(-1, 1)
          self.sequence_features.loc[not_na_mask, col] = feature_scaler.fit_transform(not_na_values).flatten()
        else:
          raise RuntimeError(f"Every value is NA in sequence feature {col}.")

        self.sequence_features[col] = self.sequence_features[col].fillna(0)  # Replace NaN to 0 (average)
        self.sequence_features_scalers.append(feature_scaler)

  def _backup_sequence_features(self):
    self.backup = self.sequence_features.copy()

  def _restore_sequence_features(self):
    self.sequence_features = self.backup
