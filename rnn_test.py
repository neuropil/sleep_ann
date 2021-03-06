import os
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler

from nn_models import basic_rnn
from utils import rolling_window,load_preprocessed

# Load preprocessed files
all_data = load_preprocessed()
data = all_data[0]

def timestep_slice_data(data,slice_size=10,rescale=True):
    # Load inputs and outputs
    labels = data['stages'][:,2]
    pows = data['pows']
    if rescale:
        scaler = StandardScaler()
        scaler.fit(pows)
        pows = scaler.transform(pows)

    pows = pows.swapaxes(0,1)


    # timeslice labels [ N,slice_size ]
    seq_labels = rolling_window(labels,slice_size)

    # timeslicing pows is awkward...
    seq_pows = rolling_window(pows,slice_size)
    seq_pows = seq_pows.swapaxes(0,1)
    seq_pows = seq_pows.swapaxes(1,2)

    return seq_pows,seq_labels

tsteps = 10
X,Y = timestep_slice_data(data,tsteps)
model = basic_rnn(tsteps)
import ipdb; ipdb.set_trace()