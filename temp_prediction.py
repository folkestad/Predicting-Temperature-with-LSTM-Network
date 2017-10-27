from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from math import sqrt

from matplotlib import pyplot

import numpy

from data_handler import get_data, invert_scale, inverse_difference
from lstm_model import *

##########################################################################

# Set how many <time unit>s we want to predict
test_size = 8
cuttoff_dataset = 249

# on t=2: 1 100 1 1 gives 0.058, 1 200 2 1 gives 0.053, 1 200 10 1 gives,
# 0.023, 1 200 20 1 gives 0.024, 1 200 40 1 gives 0.084, 1 50 40 1 gives
# 0.036, 1 15 40 1 gives 0.026,
n_rounds = 1  # 1
epochs = 200  # 10
neurons = 2  # 10
hidden_layers = 1  # 10
batch_size = 1
rmses = []
maes = []

# get data sets from data handler
scaler, real_values, train_scaled, test_scaled = get_data(
    file_name='Data/monthly_mean_global_surface_tempreratures_1880-2017.csv',
    predict_n=test_size,
    cuttoff_dataset=cuttoff_dataset
)

print("Train size and Test size: {} - {}".format(len(train_scaled), len(test_scaled)))

# runs of training and predicting
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

    # create lists for un-inverse transformed predictions and expectations
    predictions_untransformed = list(train_scaled[:, 0].tolist())
    expectations_untransformed = list(train_scaled[:, 0].tolist())

    # create lists for inverse transformed predictions and expectations
    predictions = list(real_values[:-test_size])
    expectations = list(real_values)

    # create prediction history from the last values predicted which should be
    # the real values to begin with
    history = real_values[:-test_size]

    # predict every element in test set
    for i in range(len(test_scaled)):

        # get next element in test set
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]

        # make one-step forecast
        yhat_inverted_scaled = forecast(model=lstm_model, batch_size=1, X=X)

        predictions_untransformed.append(yhat_inverted_scaled)
        expectations_untransformed.append(y)

        # invert scaling
        yhat_inverted = invert_scale(
            scaler=scaler, X=X, value=yhat_inverted_scaled)

        # invert differencing by adding last inverse differenced value to the
        # predicted change
        pred = inverse_difference(
            history=history, yhat=yhat_inverted, interval=1
        )

        # append forecast to history
        history = numpy.append(history, pred)

        # store forecast and real value
        predictions.append(pred)

        # print predicted vs expectation for both transformed and untransformed
        # predictions
        print('Year=%d, Predicted=%f, Expected=%f, difference=%f' %
              (i + 1, predictions[-1], expectations[-1], predictions[-1] - expectations[-1]))
        print('Year=%d, Predicted=%f, Expected=%f, difference=%f' %
              (i + 1, predictions_untransformed[-1], expectations_untransformed[-1], predictions_untransformed[-1] - expectations_untransformed[-1]))
        print("\n")

    # report performance
    rmse = sqrt(mean_squared_error(
        expectations[-test_size:], predictions[-test_size:]))
    mae = mean_absolute_error(
        expectations[-test_size:], predictions[-test_size:])
    print('Test RMSE: %.3f' % rmse)
    print('Test MAE: %.3f' % mae)
    rmses.append(rmse)
    maes.append(mae)

    # print if True
    if True:

        pyplot.plot(expectations_untransformed[:])
        pyplot.plot(predictions_untransformed[:])
        pyplot.show()

        pyplot.plot(expectations[:])
        pyplot.plot(predictions[:])
        pyplot.show()

# print average results
print("Avg RMSE with {} epochs: {}".format(epochs, sum(rmses) / n_rounds))
print("Avg MAE with  {} epochs: {}".format(epochs, sum(maes) / n_rounds))
