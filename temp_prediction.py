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
cuttoff_dataset = 241

# on t=2: 1 100 1 1 gives 0.058, 1 200 2 1 gives 0.053, 1 200 10 1 gives,
# 0.023, 1 200 20 1 gives 0.024, 1 200 40 1 gives 0.084, 1 50 40 1 gives
# 0.036, 1 15 40 1 gives 0.026,
n_rounds = 1
epochs = 20
neurons = 100
hidden_layers = 2
batch_size = 1
rmses = []
maes = []

# get data sets from data handler
scaler, real_values, train_scaled, test_scaled = get_data(
    file_name='Data/monthly_mean_global_surface_tempreratures_1880-2017_new.csv',
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

    predictions_untransformed = list(train_scaled[:, 0].tolist())
    expectations_untransformed = list(train_scaled[:, 0].tolist())

    history = real_values[:-test_size]

    for i in range(len(test_scaled)):
        # get next element in test set
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        print(y)

        # make one-step forecast
        yhat_inverted_scaled = forecast(model=lstm_model, batch_size=1, X=X)
        print(yhat_inverted_scaled)

        predictions_untransformed.append(yhat_inverted_scaled)
        expectations_untransformed.append(y)

        # invert scaling
        yhat_inverted = invert_scale(
            scaler=scaler, X=X, value=yhat_inverted_scaled)
        print(yhat_inverted)

        # invert differencing by adding last inverse differenced value to the
        # predicted change
        # pred = inverse_difference(
        #     history=history, yhat=yhat_inverted, interval=1)
        # print(pred)

        # print(len(test_scaled) - i)
        # print(real_values)
        # pred = inverse_difference(
        # history=real_values, yhat=yhat_inverted, interval=len(test_scaled) -
        # i)

        pred = inverse_difference(
            history=history, yhat=yhat_inverted, interval=1)

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
        # pyplot.plot(expectations)
        # pyplot.plot(predictions)
        # pyplot.show()

        pyplot.plot(expectations_untransformed)
        pyplot.plot(predictions_untransformed)
        pyplot.show()

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
        # pyplot.show()

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
