import os
import scipy.io as sio
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def merge_nrem_state(states):
    return np.array([2 if s in [1,2,3] else s for s in states])

def load_preprocessed(
        data_path='/home/elijahc/data/uminn/preprocessed',
        merge_keys=None,
        simple=False,
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
            # 'band_pow10.mat',
            ]):

    file_paths = [os.path.join(data_path,fname) for fname in file_names]
    all_data = [ sio.loadmat(fp) for fp in file_paths ]
    if merge_keys is not None:
        merged_data = {}
        for k in merge_keys:
            merged_data[k] = np.concatenate( [d[k] for d in all_data] )
        if simple:
            merged_data['stages_simple'] = merge_nrem_state(merged_data['stages'][:,2])
        return merged_data
    else:
        if simple:
            for d in all_data:
                d['stages_simple']=merge_nrem_state(d['stages'][:,2])
        return all_data

def get_inverse_freq_weights(labels):
    label_bct = np.bincount(labels)
    label_pct = label_bct/label_bct.sum()
    nz = np.nonzero(label_pct)[0].tolist()
    sparse_weights = 1/label_pct[np.nonzero(label_pct)]
    sample_weights = np.array([ sparse_weights[nz.index(s)] for s in labels ])
    return sample_weights


def get_integer_labels(labels):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(labels)

def get_label_freq(labels):
    int_enc = get_integer_labels(labels)
    bcount = np.bincount(int_enc)
    bfreq = bcount/len(int_enc)
    return bfreq

def get_oh_labels(dat):
    if isinstance(dat,np.ndarray):
        sleep_labels = dat.astype(np.int8)
    elif isinstance(dat,dict):
        stages = dat['stages']
        windows = stages[:,0:2]
        sleep_labels = stages[:,2].astype(np.int8)

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
