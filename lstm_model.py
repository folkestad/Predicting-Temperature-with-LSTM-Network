from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers

import numpy


def fit(train, batch_size, epochs, neurons, hidden_layers):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()

    for i in range(1, hidden_layers + 1):
        if i == 1 and hidden_layers <= 1:
            print(1, i)
            model.add(LSTM(
                neurons,
                batch_input_shape=(batch_size, X.shape[1], X.shape[2]),
                stateful=True  # Means that the LSTM remember from the last batch,
            ))
        elif i == 1 and hidden_layers > 1:
            print(2, i)
            model.add(LSTM(
                neurons,
                batch_input_shape=(batch_size, X.shape[1], X.shape[2]),
                return_sequences=True,
                stateful=True  # Means that the LSTM remember from the last batch,
            ))
        elif i > 1 and i < hidden_layers:
            print(3, i)
            model.add(LSTM(
                neurons,
                return_sequences=True,
                stateful=True  # Means that the LSTM remember from the last batch
            ))
        elif i == hidden_layers:
            print(4, i)
            model.add(LSTM(
                neurons,
                stateful=True  # Means that the LSTM remember from the last batch
            ))
        else:
            print(5, "Error")
    model.add(Dense(1))
    adam = optimizers.adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=adam)
    # model.compile(loss='mean_squared_error', optimizer='adam')

    for i in range(epochs):
        print(i, "/", epochs)
        model.fit(X, y, epochs=1, batch_size=batch_size,
                  verbose=1, shuffle=False)
        model.reset_states()
    return model

# make a one-step forecast


def forecast(model=None, batch_size=None, X=None):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=1)
    return yhat[0, 0]
