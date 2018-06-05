import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
# Input sequence
# Original whole sequence
# wholeSequence = [[0,0,0,0,0,0,0,0,0,2,1],
#                  [0,0,0,0,0,0,0,0,2,1,0],
#                  [0,0,0,0,0,0,0,2,1,0,0],
#                  [0,0,0,0,0,0,2,1,0,0,0],
#                  [0,0,0,0,0,2,1,0,0,0,0],
#                  [0,0,0,0,2,1,0,0,0,0,0],
#                  [0,0,0,2,1,0,0,0,0,0,0],
#                  [0,0,2,1,0,0,0,0,0,0,0],
#                  [0,2,1,0,0,0,0,0,0,0,0],
#                  [2,1,0,0,0,0,0,0,0,0,0]]

# equivalent to above but generated using rolling window
longSequence = [0,0,0,0,0,0,0,0,0,2,1,0,0,0,0,0,0,0,0,0]
wholeSequence = rolling_window(np.array(longSequence),11)


# Preprocess Data:
wholeSequence = np.array(wholeSequence, dtype=float) # Convert to NP array.
data = wholeSequence[:-1] # all but last
target = wholeSequence[1:] # all but first

# Reshape training data for Keras LSTM model
# The training data needs to be (batchIndex, timeStepIndex, dimentionIndex)
# Single batch, 9 time steps, 11 dimentions
data = data.reshape((1, 9, 11))
target = target.reshape((1, 9, 11))

# Build Model
model = Sequential()  
model.add(LSTM(11, input_shape=(9, 11), unroll=True, return_sequences=True))
model.add(Dense(11))
model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(data, target, epochs=2000, batch_size=1, verbose=1)
import ipdb; ipdb.set_trace()