from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from math import sqrt

from matplotlib import pyplot

import numpy

from data_handler import get_data, invert_scale, inverse_difference

# fit an LSTM network to training data


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
    model.compile(loss='mean_squared_error', optimizer='adam')

    for i in range(epochs):
        print(i, "/", epochs)
        model.fit(X, y, epochs=1, batch_size=batch_size,
                  verbose=1, shuffle=False)
        model.reset_states()
    return model

# make a one-step forecast


def forecast(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=1)
    return yhat[0, 0]

##########################################################################

# Set how many years we want to predict and convert the years to months
n_total = 30
cuttoff_dataset = 0

# get data sets from data handler
scaler, raw_values, train_scaled, test_scaled = get_data(
    file_name='Data/high-and-low-water-levels-of-the-amazon-at-iquitos-1962-1978.csv',
    predict_n=n_total,
    cuttoff_dataset=cuttoff_dataset
)

print("Train size and Test size: {} - {}".format(len(train_scaled), len(test_scaled)))

n_rounds = 1
epochs = 1
neurons = 1
hidden_layers = 1
batch_size = 1
rmses = []
maes = []
# for epochs range(10, epochs+1, 10)
for n in range(n_rounds):
    print("Round {} (epochs -> {}, neurons -> {} and hidden layers -> {})".format(n +
                                                                                  1, epochs, neurons, hidden_layers))
    # fit the model
    lstm_model = fit(
        train=train_scaled,
        batch_size=batch_size,
        epochs=epochs,
        neurons=neurons,
        hidden_layers=hidden_layers
    )

    # forecast the entire training dataset to build up state for forecasting
    train_and_predicted = train_scaled[:, 0]
    lstm_model.predict(train_and_predicted.reshape(
        len(train_and_predicted), 1, 1), batch_size=batch_size)

    # walk-forward validation on the test data
    predictions = list()
    for i in range(len(test_scaled)):
        # get next element in test set
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]

        # make one-step forecast
        yhat = forecast(model=lstm_model, batch_size=1, X=X)

        # build up new state for next prediction
        print(train_and_predicted.shape)
        train_and_predicted = numpy.append(train_and_predicted, yhat)
        lstm_model.reset_states()
        lstm_model.predict(train_and_predicted.reshape(
            len(train_and_predicted), 1, 1), batch_size=batch_size)

        # invert scaling
        yhat = invert_scale(scaler=scaler, X=X, value=yhat)
        # invert differencing
        yhat = inverse_difference(
            history=raw_values, yhat=yhat, interval=len(test_scaled) - i)
        # store forecast
        predictions.append(yhat)
        expected = raw_values[len(train_scaled) + i + 1]
        print('Month=%d, Predicted=%f, Expected=%f, difference=%f' %
              (i + 1, yhat, expected, yhat - expected))

    # report performance
    rmse = sqrt(mean_squared_error(raw_values[-n_total:], predictions))
    mae = mean_absolute_error(raw_values[-n_total:], predictions)
    print('Test RMSE: %.3f' % rmse)
    print('Test MAE: %.3f' % mae)
    rmses.append(rmse)
    maes.append(mae)

    if True == True:

        # line plot of observed vs predicted
        true_values = raw_values[-n_total - 1:]
        pyplot.plot(true_values)
        predictions = numpy.asarray(predictions)
        preceding = raw_values[-n_total - 1:-n_total]
        pyplot.plot(numpy.hstack((preceding, predictions)))
        pyplot.show()

        # second plot
        real_values = raw_values
        predictions = numpy.asarray(predictions)
        preceding = real_values[:-len(predictions)]
        pyplot.plot(real_values)
        pyplot.plot(numpy.hstack((preceding, predictions)))
        minimum = min(true_values)
        maximum = max(true_values)
        pyplot.plot((len(preceding) - 1, len(preceding) - 1),
                    (minimum - 5, maximum + 5), 'r-')
        pyplot.show()

print(rmses)
print(maes)
print("Avg RMSE with {} epochs: {}".format(epochs, sum(rmses) / n_rounds))
print("Avg MAE with  {} epochs: {}".format(epochs, sum(maes) / n_rounds))

# if True==False:
#     real_values = raw_values[-200:]
#     predictions = numpy.asarray(predictions)
#     preceding = real_values[:-len(predictions)-1]
#     pyplot.plot(real_values)
#     pyplot.plot(numpy.hstack((preceding,predictions)))
#     pyplot.plot((len(preceding)-1, len(preceding)-1), (-3, 3), 'r-')
#     pyplot.show()
