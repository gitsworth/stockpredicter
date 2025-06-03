import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from datetime import datetime, timedelta

st.title("Stock Trend Predictor")
st.write("Predict future stock prices based on historical data using a linear regression model.")

# Sidebar inputs
stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
num_years = st.sidebar.slider("Years of historical data", 1, 10, 5)
pred_days = st.sidebar.number_input("Days to predict into the future", 1, 30, 10)

# Calculate date range
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * num_years)

# Fetch data
data = yf.download(stock_symbol, start=start_date, end=end_date)
if data.empty:
    st.error("No data found for the given stock symbol.")
    st.stop()

prices = data['Close']

if len(prices) < 30:
    st.error("Not enough data to make predictions.")
    st.stop()

# Create lag features
N = 5
df_lag = pd.DataFrame()
for i in range(N, 0, -1):
    df_lag[f'lag_{i}'] = prices.shift(i)
df_lag['target'] = prices
df_lag = df_lag.dropna()

# Split data
train_size = int(len(df_lag)*0.8)
train = df_lag.iloc[:train_size]
test = df_lag.iloc[train_size:]

X_train = train[[f'lag_{i}' for i in range(N, 0, -1)]]
y_train = train['target']
X_test = test[[f'lag_{i}' for i in range(N, 0, -1)]]
y_test = test['target']

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test set
y_pred_test = model.predict(X_test)

# Predict future recursively
last_values = prices[-N:].values
future_preds = []
current_features = last_values.copy()

for _ in range(pred_days):
    pred = model.predict(current_features.reshape(1, -1))[0]
    future_preds.append(pred)
    current_features = np.roll(current_features, -1)
    current_features[-1] = pred

# Generate future dates (business days)
last_date = prices.index[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=pred_days)

# Debug info
st.write(f"Historical points: {len(prices)}")
st.write(f"Test points: {len(test)}")
st.write(f"Test predictions: {len(y_pred_test)}")
st.write(f"Future prediction points: {len(future_preds)}")
st.write(f"Future dates points: {len(future_dates)}")

# Plotting
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=prices.index, y=prices,
    mode='lines+markers', name='Historical',
    line=dict(color='blue')))

fig.add_trace(go.Scatter(
    x=test.index, y=y_pred_test,
    mode='lines+markers', name='Test Predictions',
    line=dict(color='green', dash='dash')))

fig.add_trace(go.Scatter(
    x=future_dates, y=future_preds,
    mode='lines+markers', name='Future Predictions',
    line=dict(color='red', dash='dot')))

# Vertical lines separating train/test and future
fig.add_vline(x=test.index[0], line=dict(color='gray', dash='dash'))
fig.add_vline(x=last_date, line=dict(color='gray', dash='dash'))

fig.update_layout(
    title=f"{stock_symbol} Stock Price Prediction",
    xaxis_title="Date",
    yaxis_title="Close Price",
    legend_title="Legend",
    hovermode='x unified'
)

st.plotly_chart(fig)
