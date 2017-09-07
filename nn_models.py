from keras.layers import Input, Dense, Dropout, LSTM
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras import regularizers

def feedforward(layer_spec=[64],num_labels=5,activ='sigmoid',
                optim='adam',loss='categorical_crossentropy',
                droprate=None,loss_weights=None,reg_weight=0.01):
    inputs = Input(shape=(8,))
    for i,units in enumerate(layer_spec):
        d_layer = Dense(units,activation=activ,kernel_regularizer=regularizers.l1(reg_weight))
        if i is 0:
            x = d_layer(inputs)
            if droprate is not None:
                x = Dropout(droprate)(x)
        else:
            x = d_layer(x)
            if droprate is not None:
                x = Dropout(droprate)(x)

    predictions = Dense(num_labels,activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=optim,
                loss=loss,
                loss_weights=loss_weights,
                metrics=['categorical_accuracy'])
    
    return model

def basic_rnn():
    inputs = Input(shape=(8,))
    model = Sequential()
    model.add(LSTM)