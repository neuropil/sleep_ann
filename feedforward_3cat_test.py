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
all_data = load_preprocessed(simple=True)
merged_data = load_preprocessed(merge_keys=['pows','stages'],simple=True)

all_pow = merged_data['pows']
scaler = StandardScaler()
scaler.fit(all_pow)

def cross_validation(X,Y,kfolds,model_generator,sample_weight=None,verbose=0):

    score_results=[]

    i=0
    for train_idx,test_idx in kfolds.split(X):
        print('Split ',str(i+1))

        X_train,Y_train = X[train_idx],Y[train_idx]
        X_test,Y_test = X[test_idx],Y[test_idx]
        # Instantiate model
        m = model_generator(layer_spec=[128],num_labels=Y.shape[-1],
                                optim='adam',reg_weight=0.01)

        # Fit model
        m.fit(X_train,Y_train,
              epochs=500,verbose=verbose, batch_size=32,
              sample_weight=sample_weight[train_idx],
              validation_split=0.1,
              )

        # Score model
        score = m.evaluate(X_test,Y_test,batch_size=128,verbose=0)
        print('Split ',str(i+1),' Score: ',score)
        score_results.append(score)
        i+=1

    return score_results

def run_cross_validation(pows,stages,id,filename_base='results',sample_weight=None,verbose=0):
    print('Begining cross validation for pt',id,'...')

    # Make 10 randomly chosen 90:10 splits of the data
    kf = KFold(n_splits=10,shuffle=True)
    kf.get_n_splits(pows)

    print('Running cross validation for pt ',id,'...')
    results = cross_validation(pows,stages,kf,feedforward,sample_weight=sample_weight,verbose=verbose)
    results_df = pd.DataFrame(results,columns=['loss','accuracy'])
    filename = str(id) + '_' + filename_base + '.df'
    print('Writing file ',filename, '...')
    results_df.to_pickle(filename)
    print('')
    print(id,' results:')
    print(results_df['accuracy'].describe())

def run_individual(all_data):
    stages = [ d['stages_simple'] for d in all_data ]
    pows = [ get_pow_bands(d,scaler) for d in all_data ]
    pt_ids = np.arange(1,11)

    for id,X,stg in zip(pt_ids,pows,stages):
        pt_id = 'pt'+str(id)
        sample_weights = get_inverse_freq_weights(stg)
        stages_oh = get_oh_labels(stg)
        run_cross_validation(
                X,
                stages_oh,
                pt_id,
                sample_weight=sample_weights,
                verbose=0,
                filename_base='3cat_results')

# Run individual
# run_individual(all_data)

# Run combined
merged_data
merge_stages = merged_data['stages_simple']
merge_pows = get_pow_bands(merged_data,scaler)
merge_stages_oh = get_oh_labels(merge_stages)
prefix='ts_combined'
merge_sample_weights = get_inverse_freq_weights(merge_stages)
run_cross_validation(
        merge_pows,
        merge_stages_oh,
        prefix,
        filename_base='9pt_results',
        sample_weight=merge_sample_weights,
        verbose=0)
