# Ex.No: 6               HOLT WINTERS METHOD
### Date: 



### AIM:
To forecast future prices using Holt-Winters exponential smoothing by analyzing XAUUSD prices. The goal is to predict closing prices for the next 30 days.
### ALGORITHM:
1. Import the necessary libraries for data manipulation, visualization, and time series modeling.
2. Load a CSV file containing daily sales data into a DataFrame. Parse the 'date' column as datetime and perform initial data exploration.
3. Group the data by date and resample it to monthly frequency (beginning of the month).
4. Plot the time series data to observe trends, seasonality, and irregularities.
5. Import statsmodels libraries needed for time series analysis (Holt-Winters method).
6. Decompose the time series data into additive components (trend, seasonality, and residuals) and plot them.
7. Calculate the root mean squared error (RMSE) to evaluate the model’s performance.
8. Calculate the mean and standard deviation of the sales dataset. Then, fit the Holt-Winters model to the entire dataset and make future predictions.
9. Plot the original sales data and the predictions on the same graph.
### PROGRAM:
```py
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

data = pd.read_csv('XAUUSD_2010-2023.csv')

# Convert 'time' column to datetime format and set it as the index
data['time'] = pd.to_datetime(data['time'], format='%d-%m-%Y %H:%M')
data.set_index('time', inplace=True)

# Resample the data to daily frequency using the 'close' price
data_resampled = data['close'].resample('D').mean()

# Forward-fill any missing data
data_clean = data_resampled.ffill()

# Split the data into training and test sets
train_size = len(data_clean) - 60
train_data = data_clean[:train_size]
test_data = data_clean[train_size:]

# Fit the Holt-Winters model on the training data (weekly seasonality)
model = ExponentialSmoothing(train_data, trend="add", seasonal="add", seasonal_periods=7)
fit = model.fit()

# Test prediction: Predict for the test period (the last 60 days)
test_predictions = fit.forecast(steps=len(test_data))

# Final prediction: Forecast for the next 30 days (after the test period)
n_steps = 30
final_predictions = fit.forecast(steps=n_steps)

# Plot the original data, test predictions, and final forecast
plt.figure(figsize=(9, 5))
plt.plot(data_clean.index, data_clean, label='Original Data')
plt.plot(test_data.index, test_predictions, label='Test Predictions', color='orange')
plt.plot(pd.date_range(start=data_clean.index[-1], periods=n_steps+1, freq='D')[1:], final_predictions, label='Final Forecast', color='green')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Test and Final Predictions for XAUUSD Prices')
plt.legend()
plt.show()

```
### OUTPUT:

TEST_PREDICTION
![image](https://github.com/user-attachments/assets/539f38f0-9030-4306-9413-45807dc6686c)


FINAL_PREDICTION
![image](https://github.com/user-attachments/assets/4d7a07e8-444e-42d1-a631-7d733d064974)

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
