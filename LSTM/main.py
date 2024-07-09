import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

electricity_data = pd.read_csv('electricity_data.csv')
electricity_data['datetime'] = pd.to_datetime(electricity_data['datetime'])
electricity_data.set_index('datetime', inplace=True)

electricity_data_resampled = electricity_data.resample('15T').sum()

scaler = MinMaxScaler()
electricity_data_normalized = scaler.fit_transform(electricity_data_resampled[['consumption']])

# Function to create sequences for LSTM
def create_sequences(data, seq_length, forecast_steps):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length - forecast_steps + 1):
        seq = data[i:i + seq_length]
        target = data[i + seq_length:i + seq_length + forecast_steps]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


# Define forecast steps
sequence_length = 96
forecast_steps = 96

# Create sequences and targets
X, y = create_sequences(electricity_data_normalized, sequence_length, forecast_steps)


# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(units=96))  # Adjust to the number of steps for the next day
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform the predictions
y_pred_inv = scaler.inverse_transform(np.concatenate((X_test[:, -1, :], y_pred), axis=1))[:, -96:]  # Adjust for 96 steps

# Plot results
actual_data = electricity_data_resampled['consumption'][train_size + sequence_length:]
forecast_index = pd.date_range(start=actual_data.index[-1], periods=forecast_steps, freq='15T')
plt.plot(actual_data.index, actual_data, label='Actual Data')
plt.plot(forecast_index, y_pred_inv[0][:forecast_steps], color='red', label='Forecasted Data')
plt.legend()
plt.show()



