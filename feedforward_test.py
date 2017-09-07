import numpy as np
from pprint import pprint as pp
from nn_models import feedforward
from utils import *

from sklearn.model_selection import KFold

'''
Preprocessed datafiles are mat files with the following keys:
bands : ndarray
'''

all_data = load_preprocessed()
merged_data = {}
merged_data['pows'] = np.concatenate([get_pow_bands(d) for d in all_data])
merged_data['stages'] = np.concatenate( [d['stages'] for d in all_data] )

# data = all_data[1]
data = merged_data
s_lab = data['stages'][:,2]
Y = get_oh_labels(data)
lab_weights = 1/get_label_freq(s_lab)
lab_weights = lab_weights.tolist()
X = get_pow_bands(data,rescale=True)
naive_model = feedforward(layer_spec=[64],num_labels=Y.shape[-1],
                          optim='nadam',reg_weight=0.01)

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