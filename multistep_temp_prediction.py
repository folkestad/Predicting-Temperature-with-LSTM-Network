from multistep_data_handler import *
from multistep_lstm_model import *


# configure
n_lag = 1
n_seq = 8
n_test = 8

scaler, series, train, test = get_data(
    file_name='Data/monthly_mean_global_surface_tempreratures_1880-2017_new.csv',
    predict_n=8,
    cuttoff_dataset=,
    n_lag=n_lag,
    n_seq=n_seq,
    n_test=n_test
)

model = fit_lstm(train, n_lag, n_seq, 1, 50, 1)

forecasts = make_forecasts(model, 1, train, test, n_lag, n_seq)
forecasts = inverse_transform(series, forecasts, scaler, n_test + 2)

actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test + 2)

# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# plot forecasts
plot_forecasts(series, forecasts, n_test + 2)
