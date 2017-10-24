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


def forecast(model=None, batch_size=None, X=None):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=1)
    return yhat[0, 0]

##########################################################################

# Set how many years we want to predict and convert the years to months
test_size = 1
cuttoff_dataset = 0

n_rounds = 1
epochs = 30
neurons = 1
hidden_layers = 1
batch_size = 1
rmses = []
maes = []

# get data sets from data handler
scaler, real_values, train_scaled, test_scaled = get_data(
    file_name='Data/high-and-low-water-levels-of-the-amazon-at-iquitos-1962-1978.csv',
    predict_n=test_size,
    cuttoff_dataset=cuttoff_dataset
)

print("Train size and Test size: {} - {}".format(len(train_scaled), len(test_scaled)))

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
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=batch_size)

    predictions = list(real_values[:-test_size])
    expectations = list(real_values)

    # predictions_untransformed = list(train_scaled[:, 0].tolist())
    # expectations_untransformed = list(train_scaled[:, 0].tolist())

    history = real_values[:-test_size]

    for i in range(len(test_scaled)):
        # get next element in test set
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        print(y)

        # make one-step forecast
        yhat_inverted_scaled = forecast(model=lstm_model, batch_size=1, X=X)
        print(yhat_inverted_scaled)

        # predictions_untransformed.append(yhat_inverted_scaled)
        # expectations_untransformed.append(y)

        # invert scaling
        yhat_inverted = invert_scale(
            scaler=scaler, X=X, value=yhat_inverted_scaled)
        print(yhat_inverted)

        # invert differencing by adding last inverse differenced value to the
        # predicted change
        # pred = inverse_difference(
        #     history=history, yhat=yhat_inverted, interval=1)
        # print(pred)

        pred = inverse_difference(
            history=real_values, yhat=yhat_inverted, interval=len(test_scaled) - i)

        # print("yhats:", yhat_inverted_scaled, yhat_inverted, pred)
        # print("\n")
        # print("comparison: ", pred, real_values[-len(test_scaled) + i])
        # history = numpy.append(history, real_values[-len(test_scaled) + i])
        history = numpy.append(history, pred)

        # store forecast
        predictions.append(pred)
        # expectations.append(real_values[-len(test_scaled) + i])
        print('Year=%d, Predicted=%f, Expected=%f, difference=%f' %
              (i + 1, predictions[-1], expectations[-1], predictions[-1] - expectations[-1]))
        print("\n")

    # report performance
    rmse = sqrt(mean_squared_error(expectations, predictions))
    mae = mean_absolute_error(expectations, predictions)
    print('Test RMSE: %.3f' % rmse)
    print('Test MAE: %.3f' % mae)
    rmses.append(rmse)
    maes.append(mae)

    if True == True:

        # line plot of observed vs predicted
        # true_values = real_values[-test_size - 1:]
        # pyplot.plot(true_values)
        # predictions = numpy.asarray(predictions)
        # preceding = real_values[-test_size - 1:-test_size]
        # pyplot.plot(numpy.hstack((preceding, predictions)))
        # pyplot.show()

        # print(history)
        # print(real_values[:len(history)])
        pyplot.plot(expectations)
        pyplot.plot(predictions)
        pyplot.show()

        # pyplot.plot(expectations_untransformed)
        # pyplot.plot(predictions_untransformed)
        # pyplot.show()

        # plot scaled values
        # print(targets_inverted_scaled)

        # targets = targets_inverted_scaled[-test_size - 1:]
        # pyplot.plot(targets)
        # predictions_inverted_scaled = numpy.asarray(
        #     predictions_inverted_scaled)
        # preceding_inverted_scaled = targets_inverted_scaled[
        #     -test_size - 1: -test_size]
        # pyplot.plot(numpy.hstack(
        #     (preceding_inverted_scaled, predictions_inverted_scaled)))
        # pyplot.show()

        # print(targets_inverted)

        # targets = targets_inverted[-test_size - 1:]
        # pyplot.plot(targets)
        # predictions_inverted = numpy.asarray(
        #     predictions_inverted)
        # preceding_inverted = targets_inverted[
        #     -test_size - 1: -test_size]
        # pyplot.plot(numpy.hstack(
        #     (preceding_inverted, predictions_inverted)))
        # pyplot.show()
        # print(len(predictions_scaled), len(targets_scaled))
        # for i in range(len(predictions_scaled)):
        #     print(
        #         "P:{} --> T:{}".format(predictions_scaled[i], targets_scaled[len(train_scaled) + i]))

        # second plot
        # real_values = real_values
        # predictions = numpy.asarray(predictions)
        # preceding = real_values[:-len(predictions)]
        # pyplot.plot(real_values)
        # pyplot.plot(numpy.hstack((preceding, predictions)))
        # minimum = min(true_values)
        # maximum = max(true_values)
        # pyplot.plot((len(preceding) - 1, len(preceding) - 1),
        #             (minimum - 5, maximum + 5), 'r-')
        pyplot.show()

print(rmses)
print(maes)
print("Avg RMSE with {} epochs: {}".format(epochs, sum(rmses) / n_rounds))
print("Avg MAE with  {} epochs: {}".format(epochs, sum(maes) / n_rounds))

# if True==False:
#     real_values = real_values[-200:]
#     predictions = numpy.asarray(predictions)
#     preceding = real_values[:-len(predictions)-1]
#     pyplot.plot(real_values)
#     pyplot.plot(numpy.hstack((preceding,predictions)))
#     pyplot.plot((len(preceding)-1, len(preceding)-1), (-3, 3), 'r-')
#     pyplot.show()
