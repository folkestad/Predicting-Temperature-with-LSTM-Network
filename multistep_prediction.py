from multistep_data_handler import *
from multistep_lstm_model import *


# configure
n_lag = 1
n_seq = 8
n_test = 8

scaler, series, train, test = get_data(
    file_name='Data/monthly_mean_global_surface_tempreratures_1880-2017.csv',
    predict_n=8,
    cuttoff_dataset=249,
    n_lag=n_lag,
    n_seq=n_seq,
    n_test=n_test
)

runs = 1
batch_size = 1
epochs = 5
neurons = 16

results = []
for i in range(runs):

    model = fit_lstm(train, n_lag, n_seq, batch_size, epochs, neurons)

    forecasts = make_forecasts(model, 1, train, test, n_lag, n_seq)
    forecasts = inverse_transform(series, forecasts, scaler, n_test + 7)

    actual = [row[n_lag:] for row in test]
    actual = inverse_transform(series, actual, scaler, n_test + 7)

    # evaluate forecasts
    result = evaluate_forecasts(actual, forecasts, n_lag, n_seq)
    # plot forecasts
    if runs == 1:
        plot_forecasts(series[-20:], forecasts, n_test + 7)

    results.append(result)
    print("Run {}/{} finished.".format(i + 1, runs))

print("runs {} epochs {} neurons {}".format(runs, epochs, neurons))
print("RMSE RUNS AVERAGE: {}".format(sum(results) / len(results)))
