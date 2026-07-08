"""Optimus5p model architecture (Sample et al., 2019)."""

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Activation, Flatten


def one_hot_encode(df, col='utr5', seq_len=50):
    """One-hot encode nucleotide sequences."""
    nuc_d = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0]}
    vectors = np.empty([len(df), seq_len, 4])
    for i, seq in enumerate(df[col].str[-seq_len:]):
        seq = seq.lower()
        if len(seq) < seq_len:
            seq = 'n' * (seq_len - len(seq)) + seq
        vectors[i] = np.array([nuc_d[x] for x in seq])
    return vectors


def build_model(inp_len=50, layers=3, nbr_filters=120, filter_len=8, nodes=40,
                dropout1=0, dropout2=0, dropout3=0):
    model = Sequential()
    if layers >= 1:
        model.add(Conv1D(activation="relu", input_shape=(inp_len, 4),
                         padding="same", filters=nbr_filters, kernel_size=filter_len))
    if layers >= 2:
        model.add(Conv1D(activation="relu", padding="same",
                         filters=nbr_filters, kernel_size=filter_len))
        model.add(Dropout(dropout1))
    if layers >= 3:
        model.add(Conv1D(activation="relu", padding="same",
                         filters=nbr_filters, kernel_size=filter_len))
        model.add(Dropout(dropout2))
    model.add(Flatten())
    model.add(Dense(nodes))
    model.add(Activation('relu'))
    model.add(Dropout(dropout3))
    model.add(Dense(1))
    model.add(Activation('linear'))
    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.Adam(learning_rate=0.001,
                                                 beta_1=0.9, beta_2=0.999, epsilon=1e-08))
    return model
