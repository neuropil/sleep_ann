import os
import numpy as np
import scipy.io as sio
from pprint import pprint as pp
from nn_models import feedforward

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

'''
Preprocessed datafiles are mat files with the following keys:
bands : ndarray
'''

data_path = '/home/elijahc/data/uminn/preprocessed'
file_names = ['band_pow1.mat','band_pow2.mat']

file_paths = [os.path.join(data_path,fname) for fname in file_names]

def load_file(file_path):
    return sio.loadmat(file_path)

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

def get_pow_bands(dat):
    pow_bands = dat['pows']
    return pow_bands

# def iter_train(pow_bands,stages,model,n_splits=5,shuffle=False):
#     import ipdb; ipdb.set_trace()

#     for train_idx, test_idx in kf.split(pow_bands):
#         print("Train: ",len(train_idx),"Test:",len(test_idx))
#         X_train,Y_train = pow_bands[train_idx],stages[train_idx]
#         X_test,Y_test = pow_bands[test_idx],stages[test_idx]
#         # print("pow_bands shape: ",X_train.shape)
#         print('')
#         model.fit(X_train,Y_train,epochs=100, batch_size=32,verbose=1)
#         score = model.evaluate(X_test,Y_test,batch_size=128,verbose=0)
#         print('')
#         pp(score)
#     return model,score

data = load_file(file_paths[0])

s_lab = data['stages'][:,2]
Y = get_oh_labels(data)
lab_weights = 1/get_label_freq(s_lab)
lab_weights = lab_weights.tolist()
X = get_pow_bands(data)
naive_model = feedforward(layer_spec=[16,16],num_labels=Y.shape[-1],
                          droprate=0.5,optim='nadam',reg_weight=0.01)

kf = KFold(n_splits=5,shuffle=True)
kf.get_n_splits(X)
train_idx,test_idx = next(kf.split(X))
X_train,Y_train = X[train_idx],Y[train_idx]
X_test,Y_test = X[test_idx],Y[test_idx]

naive_model.fit(X_train,Y_train,epochs=200,verbose=1,
                # class_weight=lab_weights,
                validation_split=0.1)

score = naive_model.evaluate(X_test,Y_test,batch_size=128,verbose=0,sample_weight=None)

print('')
print('Final score: ',score)