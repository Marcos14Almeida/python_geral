# https://github.com/marcopeix/time-series-analysis/blob/master/NBEATS.ipynb
# https://www.datasciencewithmarco.com/blog/the-easiest-way-to-forecast-time-series-using-n-beats

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from darts import TimeSeries

import warnings
warnings.filterwarnings('ignore')

plt.rcParams["figure.figsize"] = (9, 6)

df = pd.read_csv('data/daily_traffic.csv')

df.head()

series = TimeSeries.from_dataframe(df, time_col='date_time')

series.plot()

# Plot the DataFrame
df.plot()

print()
from darts.utils.statistics import check_seasonality

is_daily_seasonal, daily_period = check_seasonality(series, m=24, max_lag=400, alpha=0.05)
is_weekly_seasonal, weekly_period = check_seasonality(series, m=168, max_lag=400, alpha=0.05)

print(f'Daily seasonality: {is_daily_seasonal} - period = {daily_period}')
print(f'Weekly seasonality: {is_weekly_seasonal} - period = {weekly_period}')

print("\nTRAIN AND TEST")

train, test = series[:-120], series[-120:]

train.plot(label='train')
test.plot(label='test')

# Plot the train and test portions
plt.figure(figsize=(12, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.legend()
plt.title('Train and Test Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Baseline

print()
print("BASELINE")
from darts.models.forecasting.baselines import NaiveSeasonal

naive_seasonal = NaiveSeasonal(K=168)
naive_seasonal.fit(train)

pred_naive = naive_seasonal.predict(120)

test.plot(label='test')
pred_naive.plot(label='Baseline')

from darts.metrics import mae

naive_mae = mae(test, pred_naive)

print(naive_mae)

# NBEATS
print()
print("N-BEATS")
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler

train_scaler = Scaler()
scaled_train = train_scaler.fit_transform(train)

nbeats = NBEATSModel(
    input_chunk_length=168, 
    output_chunk_length=24,
    generic_architecture=True,
    random_state=42)

nbeats.fit(
    scaled_train,
    epochs=20
)

scaled_pred_nbeats = nbeats.predict(n=120)

pred_nbeats = train_scaler.inverse_transform(scaled_pred_nbeats)

mae_nbeats = mae(test, pred_nbeats)

print(mae_nbeats)

# NBEATS COVARIATES

print()
print("N-BEATS 2")
from darts import concatenate
from darts.utils.timeseries_generation import datetime_attribute_timeseries as dt_attr

cov = concatenate(
    [dt_attr(series.time_index, 'day', dtype=np.float32), dt_attr(series.time_index, 'week', dtype=np.float32)],
    axis='component'
)

cov_scaler = Scaler()

scaled_cov = cov_scaler.fit_transform(cov)

train_scaled_cov, test_scaled_cov = scaled_cov[:-120], scaled_cov[-120:]

scaled_cov.plot()

nbeats_cov = NBEATSModel(
    input_chunk_length=168,
    output_chunk_length=24,
    generic_architecture=True,
    random_state=42)

nbeats_cov.fit(
    scaled_train,
    past_covariates=scaled_cov,
    epochs=20
)

scaled_pred_nbeats_cov = nbeats_cov.predict(past_covariates=scaled_cov, n=120)

pred_nbeats_cov = train_scaler.inverse_transform(scaled_pred_nbeats_cov)

mae_nbeats_cov = mae(test, pred_nbeats_cov)

print(mae_nbeats_cov)

test.plot(label='test')
pred_nbeats.plot(label='N-BEATS')

fig, ax = plt.subplots()

x = ['Baseline', 'N-Beats', 'N-BEATS + covariates']
y = [naive_mae, mae_nbeats, mae_nbeats_cov]

ax.bar(x, y, width=0.4)
ax.set_xlabel('Models')
ax.set_ylabel('MAE')
ax.set_ylim(0, 350)
ax.grid(False)

for index, value in enumerate(y):
    plt.text(x=index, y=value + 10, s=str(round(value, 0)), ha='center')

plt.tight_layout()
plt.show()
