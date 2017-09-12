import numpy as np
import pandas as pd
from pprint import pprint as pp
from nn_models import feedforward
from utils import *

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

'''
Preprocessed datafiles are mat files with the following keys:
bands : ndarray
'''

files = [
    'band_pow1.mat',
    'band_pow2.mat',
    'band_pow3.mat',
    'band_pow4.mat',
    'band_pow5.mat',
    'band_pow6.mat',
    'band_pow7.mat',
    'band_pow8.mat',
    'band_pow9.mat',
    'band_pow10.mat',
    ]
all_data = load_preprocessed(file_names=files)
merged_data = {}
merged_data['pows'] = np.concatenate([get_pow_bands(d) for d in all_data])
merged_data['stages'] = np.concatenate( [d['stages'] for d in all_data] )

def cross_validation(X,Y,kfolds,model_generator):

    score_results=[]

    for train_idx,test_idx in kfolds.split(X):

        X_train,Y_train = X[train_idx],Y[train_idx]
        X_test,Y_test = X[test_idx],Y[test_idx]
        # Instantiate model
        m = model_generator(layer_spec=[128],num_labels=Y.shape[-1],
                                optim='adam',reg_weight=0.01)

        # Fit model
        m.fit(X_train,Y_train,
              epochs=2000,verbose=1, batch_size=32,
              validation_split=0.1)

        # Score model
        score = m.evaluate(X_test,Y_test,batch_size=128,verbose=0)
        print('Score: ',score)
        score_results.append(score)

    return score_results
all_pow = merged_data['pows']
scaler = StandardScaler()
scaler.fit(all_pow)

# data = all_data[1]
data = merged_data
X = get_pow_bands(data,scaler=scaler)
Y = get_oh_labels(data)

# s_lab = data['stages'][:,2]
# lab_weights = 1/get_label_freq(s_lab)
# lab_weights = lab_weights.tolist()


# Take a single 80:20 split of the data
# train_idx,test_idx = next(kf.split(X))

# Make 5 randomly chosen 80:20 splits of the data
kf = KFold(n_splits=10,shuffle=True)
kf.get_n_splits(X)

results = cross_validation(X,Y,kf,feedforward)
results_df = pd.DataFrame(results,columns=['loss','accuracy'])
print('Writing results to pickle...')
results_df.to_pickle('ts_merge_e1000_results.df')
print('')
print(results_df['accuracy'].describe())
# import ipdb; ipdb.set_trace()