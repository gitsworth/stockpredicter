import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from datetime import datetime, timedelta

# App title
st.title("📈 Stock Trend Predictor")
st.markdown("Predict future stock prices using historical data and a linear regression model.")

# Sidebar inputs
stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
num_years = st.sidebar.slider("Years of historical data", 1, 10, 5)
pred_days = st.sidebar.number_input("Days to predict", 1, 30, 7)

# Date range
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * num_years)

# Download stock data
data = yf.download(stock_symbol, start=start_date, end=end_date)

if data.empty:
    st.error("⚠️ No data found. Please enter a valid stock symbol.")
    st.stop()

prices = data['Close']

# Check if there's enough data
if len(prices) < 30:
    st.error("⚠️ Not enough data to perform prediction. Try increasing the historical range.")
    st.stop()

# Create lagged features
N = 5
lagged_data = pd.DataFrame()
for i in range(N, 0, -1):
    lagged_data[f'lag_{i}'] = prices.shift(i)
lagged_data['target'] = prices
lagged_data = lagged_data.dropna()

# Split into training and testing
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

# Predict on test set
y_pred_test = model.predict(X_test) if not X_test.empty else []

# Predict future
future_predictions = []
current_features = prices[-N:].values
try:
    for _ in range(pred_days):
        next_pred = model.predict([current_features])[0]
        future_predictions.append(next_pred)
        current_features = np.roll(current_features, -1)
        current_features[-1] = next_pred
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# Future dates
future_dates = pd.date_range(start=prices.index[-1] + pd.Timedelta(days=1), periods=pred_days, freq='B')

# Plotting
fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(x=prices.index, y=prices, mode='lines', name='Historical', line=dict(color='blue')))

# Test predictions
if len(y_pred_test) == len(test_data):
    fig.add_trace(go.Scatter(x=test_data.index, y=y_pred_test, mode='lines', name='Test Predictions', line=dict(color='green', dash='dash')))
else:
    st.warning("Test predictions not plotted due to size mismatch.")

# Future predictions
if len(future_predictions) == len(future_dates):
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Future Predictions', line=dict(color='red', dash='dot')))
else:
    st.warning("Future predictions not plotted due to size mismatch.")

# Layout
fig.update_layout(
    title=f"{stock_symbol} Stock Price Prediction",
    xaxis_title="Date",
    yaxis_title="Price",
    legend_title="Legend",
    hovermode="x unified",
    yaxis=dict(autorange=True)
)

st.plotly_chart(fig)
