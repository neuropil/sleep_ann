import numpy as np
import pandas as pd
from pprint import pprint as pp
from nn_models import feedforward
import utils.io as uio
from model_selection import prepare_cvs,CVScore

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

# Load data
osx_data_path = uio.osx_path()
all_data = uio.load_preprocessed(data_path=osx_data_path,simple=True)
merge_data = uio.merge_data(all_data,merge_keys=['pows','stages'],simple=True)
scaler = StandardScaler()
scaler.fit(merge_data['pows'])
band_pows = uio.get_pow_bands(merge_data,scaler=scaler)
labels = merge_data['stages_simple']

groups = [ [i]*len(d['pows']) for i,d in zip(np.arange(len(all_data)),all_data) ]
groups = np.concatenate(groups,axis=0)
logo = LeaveOneGroupOut()

model_params = dict(
    layer_spec=[64],
    activ='relu',
    optim='nadam'
)
fit_params = dict(
    validation_split=0.1,
    epochs=500,
    verbose=0
)
cvs_comp = prepare_cvs(model='feedforward',cv='logo',groups=groups,**model_params)
cvs = CVScore(**cvs_comp)
cvs_iter = cvs.cv.split(band_pows,labels,groups)
import ipdb; ipdb.set_trace()