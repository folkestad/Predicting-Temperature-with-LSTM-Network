from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from sklearn.preprocessing import MinMaxScaler

import numpy

from os import path, sys

# date-time parsing function for loading the dataset
def date_parser(x):
    return datetime.strptime(x, '%Y-%m')

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    df.columns = ['value', 'target']
    return df

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

def get_data(file_name='monthly-temperature-in-england.csv', predict_n_months=10):
    # load dataset
    series = read_csv(
        filepath_or_buffer='Data/'+file_name, 
        sep=',', 
        header=0, 
        parse_dates=[0], 
        index_col=0, 
        squeeze=True, 
        date_parser=date_parser,
        skip_blank_lines=True,
        skiprows=[2979-1]
    )

    # transform data to be stationary
    raw_values = [ float(x.replace('?', '')) for x in series.values ]
    diff_values = difference(raw_values, 1)

    # transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values

    # split data into train and test-sets
    train, test = supervised_values[0:-predict_n_months], supervised_values[-predict_n_months:]

    # transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)

    return scaler, raw_values, train_scaled, test_scaled