from keras.layers import Input, Dense, Dropout, LSTM
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras import regularizers

def feedforward(layer_spec=[64],num_labels=5,activ='sigmoid',
                optim='adam',loss='categorical_crossentropy',
                droprate=None,loss_weights=None,reg_weight=0.01):
    model = Sequential()
    input_shape = (8,)
    for i,units in enumerate(layer_spec):
        if i == 0:
            d_layer = Dense(units,activation=activ,kernel_regularizer=regularizers.l1(reg_weight),input_shape=input_shape)
        else:
            d_layer = Dense(units,activation=activ,kernel_regularizer=regularizers.l1(reg_weight))
        model.add(d_layer)
        if droprate is not None:
            model.add(Dropout(droprate))

    model.add(Dense(num_labels,activation='softmax'))

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=optim,
                loss=loss,
                loss_weights=loss_weights,
                metrics=['categorical_accuracy'])
    
    return model

# TODO: Finish the function below

def basic_rnn(timesteps,output_dim=4):
    model = Sequential()
    model.add( LSTM(10, input_shape=(timesteps,8), unroll=True, return_sequences=True) )
    model.add( Dense(10) )
    return model