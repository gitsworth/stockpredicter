import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from datetime import datetime, timedelta

st.title("Stock Trend Predictor")
st.write("Predict future stock prices based on historical data using linear regression.")
st.write("Select a stock symbol, number of years of past data, and how many days to predict.")

# Sidebar inputs
stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL", help="Valid ticker symbol, e.g., AAPL")
num_years = st.sidebar.slider("Years of historical data", 1, 10, 5)
pred_days = st.sidebar.number_input("Days to predict", 1, 30, 30)

end_date = datetime.today()
start_date = end_date - timedelta(days=365 * num_years)

# Download data with error handling
try:
    data = yf.download(stock_symbol, start=start_date, end=end_date)
except Exception as e:
    st.error(f"Error downloading data: {e}")
    st.stop()

if data.empty:
    st.error("No data found for the given symbol. Try a different ticker.")
    st.stop()

prices = data['Close']

if len(prices) < 30:
    st.error("Not enough historical data. Try a different ticker or more years.")
    st.stop()

N = 5  # Number of lag days
lagged_data = pd.DataFrame()
for i in range(N, 0, -1):
    lagged_data[f'lag_{i}'] = prices.shift(i)
lagged_data['target'] = prices
lagged_data.dropna(inplace=True)

# Train/test split
train_size = int(0.8 * len(lagged_data))
train_data = lagged_data.iloc[:train_size]
test_data = lagged_data.iloc[train_size:]

X_train = train_data[[f'lag_{i}' for i in range(N, 0, -1)]]
y_train = train_data['target']
X_test = test_data[[f'lag_{i}' for i in range(N, 0, -1)]]
y_test = test_data['target']

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test set
y_pred_test = model.predict(X_test)

# Predict future recursively
last_N = prices[-N:].values
future_predictions = []
current_features = last_N.copy()
for _ in range(pred_days):
    next_pred = model.predict([current_features])[0]
    future_predictions.append(next_pred)
    current_features = np.roll(current_features, -1)
    current_features[-1] = next_pred

# Future dates (business days)
last_date = prices.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=pred_days, freq='B')

# Plot results
fig = go.Figure()
fig.add_trace(go.Scatter(x=prices.index, y=prices, mode='lines', name='Historical', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=test_data.index, y=y_pred_test, mode='lines', name='Test Predictions', line=dict(color='green', dash='dash')))
fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Future Predictions', line=dict(color='red', dash='dot')))
fig.add_vline(x=test_data.index[0], line=dict(color='gray', dash='dash'))
fig.add_vline(x=last_date, line=dict(color='gray', dash='dash'))

fig.update_layout(
    title=f"{stock_symbol} Stock Price Prediction",
    xaxis_title="Date",
    yaxis_title="Close Price",
    legend_title="Legend",
    hovermode='x unified'
)

st.plotly_chart(fig)
