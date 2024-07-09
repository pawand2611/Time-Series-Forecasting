import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

electricity_data = pd.read_csv('electricity_data.csv')
electricity_data['datetime'] = pd.to_datetime(electricity_data['datetime'])
electricity_data.set_index('datetime', inplace=True)

electricity_data_resampled = electricity_data.resample('15T').sum()

plot_acf(electricity_data_resampled['consumption'])
plot_pacf(electricity_data_resampled['consumption'])
plt.show()

order = (1,2,96)
model = ARIMA(electricity_data_resampled['consumption'], order=order)

results = model.fit()


forecast_steps = 96
forecast = results.get_forecast(steps=forecast_steps)

forecast_values = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

plt.plot(electricity_data_resampled['consumption'], label='Actual Data')
plt.plot(forecast_values.index, forecast_values, color='red', label='Forecasted Data')
plt.fill_between(confidence_intervals.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='red', alpha=0.2, label='Confidence Intervals')
plt.legend()
plt.show()
