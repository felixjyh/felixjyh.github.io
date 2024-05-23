# Time Series Forecasting

Time series forecasting involves predicting future values based on previously observed values in a time series data set. Time series data is a sequence of data points indexed in time order, often measured at regular intervals. Common applications include stock market analysis, weather forecasting, demand planning, and more.

### Common Time Series Forecasting Algorithms

1. **ARIMA (AutoRegressive Integrated Moving Average)**:
   - ARIMA models are used to describe and forecast stationary time series data. The model combines three components: autoregression (AR), differencing (I), and moving average (MA).

2. **Exponential Smoothing (ETS)**:
   - Exponential Smoothing models apply weighted averages of past observations, with exponentially decreasing weights over time. Variants include Simple Exponential Smoothing, Holt’s Linear Trend Model, and Holt-Winters Seasonal Model.

3. **Prophet**:
   - Prophet is a forecasting tool developed by Facebook, designed for forecasting time series data with daily observations that display patterns on different time scales (e.g., weekly, yearly).

4. **LSTM (Long Short-Term Memory)**:
   - LSTM networks are a type of recurrent neural network (RNN) that can capture long-term dependencies in time series data and are particularly useful for complex sequences.

### Examples

#### 1. ARIMA

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

# Load dataset
data = pd.read_csv('airline-passengers.csv', index_col='Month', parse_dates=True)

# Fit ARIMA model
model = ARIMA(data['Passengers'], order=(5,1,0))
model_fit = model.fit(disp=0)

# Forecast
forecast, stderr, conf_int = model_fit.forecast(steps=12)
print(forecast)
```

#### 2. Exponential Smoothing

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load dataset
data = pd.read_csv('airline-passengers.csv', index_col='Month', parse_dates=True)

# Fit Exponential Smoothing model
model = ExponentialSmoothing(data['Passengers'], trend='add', seasonal='add', seasonal_periods=12)
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=12)
print(forecast)
```

#### 3. Prophet

```python
import pandas as pd
from fbprophet import Prophet

# Load dataset
data = pd.read_csv('airline-passengers.csv')
data.columns = ['ds', 'y']  # Prophet requires columns 'ds' (date) and 'y' (value)

# Fit Prophet model
model = Prophet()
model.fit(data)

# Create future dataframe
future = model.make_future_dataframe(periods=12, freq='M')

# Forecast
forecast = model.predict(future)
model.plot(forecast)
plt.show()
```

#### 4. LSTM

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv('airline-passengers.csv', usecols=[1])
scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(data.values.reshape(-1, 1))

# Convert data to sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 12
x_data, y_data = create_sequences(data, seq_length)

# Convert to PyTorch tensors
x_data = torch.from_numpy(x_data).float()
y_data = torch.from_numpy(y_data).float()

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# Initialize model, loss function, and optimizer
model = LSTMModel()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 100
for epoch in range(epochs):
    for i in range(len(x_data)):
        seq = x_data[i]
        label = y_data[i]

        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        loss = loss_function(y_pred, label)
        loss.backward()
        optimizer.step()

    if epoch%10 == 0:
        print(f'Epoch {epoch} Loss: {loss.item()}')

# Predict
model.eval()
with torch.no_grad():
    test_seq = x_data[-1]
    for _ in range(12):
        y_pred = model(test_seq)
        test_seq = torch.cat((test_seq[1:], y_pred.view(1, -1)))

# Convert predictions back to original scale
predicted = scaler.inverse_transform(y_pred.view(-1, 1).numpy())
print(predicted)
```

### Explanation

1. **ARIMA**: This example uses the `statsmodels` library to fit an ARIMA model to the time series data and forecast future values.
2. **Exponential Smoothing**: This example uses the Holt-Winters method from `statsmodels` for exponential smoothing to forecast future values.
3. **Prophet**: This example uses Facebook's `Prophet` library to model and forecast the time series data.
4. **LSTM**: This example uses a PyTorch implementation of an LSTM network to forecast future values. The data is scaled, transformed into sequences, and then passed through the LSTM model for training and prediction.

These examples illustrate different methods for time series forecasting, each with its own strengths and suitable applications.
