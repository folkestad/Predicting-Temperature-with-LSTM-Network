from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from math import sqrt

from matplotlib import pyplot

import numpy

from data_handler import get_data, invert_scale, inverse_difference
 
# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(
        neurons, 
        batch_input_shape=(batch_size, X.shape[1], X.shape[2]), 
        stateful=True # Means that the LSTM remember from the last batch
    ))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        print(i,"/",nb_epoch)
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()
    return model
 
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=1)
    return yhat[0,0]

#######################################################################################

# Set how many years we want to predict and convert the years to months
n_years = 1
n_months = 8
n_months_total = n_years*12 + n_months

# get data sets from data handler
scaler, raw_values, train_scaled, test_scaled = get_data(
    file_name='Data/monthly_mean_global_surface_tempreratures_1880-2017_new.csv', 
    predict_n_months=n_months_total
)

n_rounds = 10
epochs = 1
neurons = 2
results = []
for n in range(n_rounds):
    print("Round {} (epochs -> {} and neurons -> {})".format(n+1, epochs, neurons))
    # fit the model
    lstm_model = fit_lstm(train=train_scaled, batch_size=1, nb_epoch=epochs, neurons=neurons)

    # forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)
    
    # walk-forward validation on the test data
    predictions = list()
    for i in range(len(test_scaled)):
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        yhat = inverse_difference(raw_values, yhat, len(test_scaled) - i)
        # store forecast
        predictions.append(yhat)
        expected = raw_values[len(train_scaled) + i + 1] 
        print('Month=%d, Predicted=%f, Expected=%f, difference=%f' % (i + 1, yhat, expected, yhat-expected))
    
    # report performance
    rmse = sqrt(mean_squared_error(raw_values[-n_months_total:], predictions))
    print('Test RMSE: %.3f' % rmse)
    results.append(rmse)

    if True==False:
        # line plot of observed vs predicted
        true_values = raw_values[-n_months_total:]
        pyplot.plot(true_values)
        pyplot.plot(predictions)
        pyplot.show()

print(results)
print("Avg RMSE with {} epochs: {}".format(epochs, sum(results)/n_rounds))

# if True==False:
#     real_values = raw_values[-200:]
#     predictions = numpy.asarray(predictions)
#     preceding = real_values[:-len(predictions)-1]
#     pyplot.plot(real_values)
#     pyplot.plot(numpy.hstack((preceding,predictions)))
#     pyplot.plot((len(preceding)-1, len(preceding)-1), (-3, 3), 'r-')
#     pyplot.show()