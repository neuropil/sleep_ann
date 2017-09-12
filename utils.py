import os
import scipy.io as sio
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def load_preprocessed(data_path='/home/elijahc/data/uminn/preprocessed',
file_names=[
    'band_pow1.mat',
    'band_pow2.mat',
    'band_pow3.mat',
    'band_pow4.mat',
    'band_pow5.mat',
    'band_pow6.mat',
    'band_pow7.mat',
    'band_pow8.mat',
    'band_pow9.mat',
    'band_pow10.mat']):
    

    file_paths = [os.path.join(data_path,fname) for fname in file_names]
    return [ sio.loadmat(fp) for fp in file_paths ]

def get_integer_labels(labels):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(labels)

def get_label_freq(labels):
    int_enc = get_integer_labels(labels)
    bcount = np.bincount(int_enc)
    bfreq = bcount/len(int_enc)
    return bfreq

def get_oh_labels(dat):
    stages = dat['stages']
    windows = stages[:,0:2]
    sleep_labels = stages[:,2].astype(np.int8)
    window_diffs = windows[:,1] - windows[:,0]

    oh_encoder = OneHotEncoder(sparse=False)
    integer_encoded = get_integer_labels(sleep_labels)
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
    oh_labels = oh_encoder.fit_transform(integer_encoded)

    return oh_labels

def get_pow_bands(dat,scaler=None):
    pow_bands = dat['pows']
    if scaler is not None:
        pow_bands = scaler.transform(pow_bands)

    return pow_bands

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
