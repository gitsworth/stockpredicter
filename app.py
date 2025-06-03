import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from datetime import datetime, timedelta

st.title("📊 Candlestick Stock Trend Predictor")

# Sidebar inputs
stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
num_years = st.sidebar.slider("Years of historical data", 1, 10, 5)
pred_days = st.sidebar.number_input("Days to predict", 1, 30, 7)

# Date range
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * num_years)

# Download OHLCV data
data = yf.download(stock_symbol, start=start_date, end=end_date)
if data.empty:
    st.error("⚠️ No data found. Please enter a valid stock symbol.")
    st.stop()

# Plot candlestick chart
fig = go.Figure(data=[
    go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Price'
    )
])

# Linear Regression Prediction Line (optional)
prices = data['Close']
if len(prices) >= 30:
    N = 5
    lagged_data = pd.DataFrame()
    for i in range(N, 0, -1):
        lagged_data[f'lag_{i}'] = prices.shift(i)
    lagged_data['target'] = prices
    lagged_data = lagged_data.dropna()

    train_size = int(0.8 * len(lagged_data))
    X_train = lagged_data.iloc[:train_size][[f'lag_{i}' for i in range(N, 0, -1)]]
    y_train = lagged_data.iloc[:train_size]['target']
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Recursive future prediction
    last_N = prices[-N:].values
    future_predictions = []
    current_features = last_N
    for _ in range(pred_days):
        next_pred = model.predict([current_features])[0]
        future_predictions.append(next_pred)
        current_features = np.roll(current_features, -1)
        current_features[-1] = next_pred

    # Add prediction line to chart
    future_dates = pd.date_range(start=prices.index[-1] + pd.Timedelta(days=1), periods=pred_days, freq='B')
    fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines+markers', name='Predicted Price', line=dict(color='blue', dash='dot')))
else:
    st.warning("Not enough data for prediction line.")

# Update layout
fig.update_layout(
    title=f"{stock_symbol} Candlestick Chart with Predictions",
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False,
    hovermode="x unified"
)

st.plotly_chart(fig)
