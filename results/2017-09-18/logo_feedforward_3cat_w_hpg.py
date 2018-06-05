import numpy as np
import pandas as pd
from pprint import pprint as pp
from nn_models import feedforward
import src.utils.io as uio
from model_selection import prepare_cvs,CVScore

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support

# Load data
# osx_data_path = uio.osx_path()
# all_data = uio.load_preprocessed(data_path=osx_data_path,simple=True)
all_data = uio.load_preprocessed(simple=True)
merge_data = uio.merge_data(all_data,merge_keys=['pows','stages'],simple=True)
scaler = StandardScaler()
scaler.fit(merge_data['pows'])
band_pows = uio.get_pow_bands(merge_data,scaler=scaler)
labels = merge_data['stages_simple']

groups = [ [i]*len(d['pows']) for i,d in zip(np.arange(len(all_data)),all_data) ]
groups = np.concatenate(groups,axis=0)
logo = LeaveOneGroupOut()

model_params = dict(
    layer_spec=[32],
    activ='relu',
    optim='nadam',
    num_labels=3
)
fit_params = dict(
    # validation_split=0.1,
    epochs=300,
    verbose=0,
    sample_weight=uio.get_inverse_freq_weights(labels,sqrt=True)
)
cvs_comp = prepare_cvs(model='feedforward',cv='logo',n_splits=9,groups=groups,**model_params)
cvs = CVScore(**cvs_comp)
cvs_iter = cvs.cv.split(band_pows,labels,groups)
oh_encoder = OneHotEncoder(sparse=False)
int_encoded = labels.reshape(len(labels),1)
oh_encoder.fit(int_encoded)
cvs.fit(band_pows,labels,cvs_iter,oh_encoder,fit_params=fit_params)
cvs.save_results('results/performance_results.df')
import ipdb; ipdb.set_trace()
cvs.save_parameters('results/logo_merge_32_results.pk',fit_params,model_params)
