import numpy as np
import pandas as pd
from pprint import pprint as pp
from nn_models import feedforward
from utils import *
from model_selection import prepare_cvs

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score,recall_score,make_scorer

'''
Preprocessed datafiles are mat files with the following keys:
bands : ndarray
'''
osx_data_path = '/Users/elijahc/data/uminn/preprocessed'
all_data = load_preprocessed(data_path=osx_data_path,simple=True)
merged_data = load_preprocessed(data_path=osx_data_path,merge_keys=['pows','stages'],simple=True)

scaler = StandardScaler()
scaler.fit(merged_data['pows'])

model_params = dict(
    layer_spec=[128],
    reg_weight=0.1,
    num_labels=3,
)
fit_params = dict(
    validation_split=0.1,
    epochs=500,
)
cvs = prepare_cvs(model=feedforward,cv='sss',
    scoring=[make_scorer(precision_score)],
    **model_params,
)

# Run combined
merge_stages = merged_data['stages_simple']
merge_pows = get_pow_bands(merged_data,scaler)
merge_stages_oh = get_oh_labels(merge_stages)
merge_sample_weights = get_inverse_freq_weights(merge_stages)
pp(cvs)
results = cross_val_score(**cvs,X=merge_pows,y=merge_stages_oh,verbose=1)
import ipdb; ipdb.set_trace()


def run_cross_validation(pows,stages,id,filename_base='results',sample_weight=None,verbose=0):
    print('Begining cross validation for pt',id,'...')

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

prefix='ts_combined'
run_cross_validation(
        merge_pows,
        merge_stages_oh,
        prefix,
        filename_base='9pt_results',
        sample_weight=merge_sample_weights,
        verbose=0)