import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from datetime import datetime, timedelta

# App Title
st.title("📈 Stock Trend Predictor")
st.markdown("""
Predict future stock prices using a simple Linear Regression model.  
*Note: For educational purposes only — not for financial decisions.*
""")

# Sidebar
stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
num_years = st.sidebar.slider("Years of historical data", 1, 10, 5)
pred_days = st.sidebar.slider("Days to Predict", 1, 30, 10)

# Dates
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * num_years)

# Download data
data = yf.download(stock_symbol, start=start_date, end=end_date)

if data.empty:
    st.error("❌ No data found. Please check the stock symbol.")
    st.stop()

# Use 'Close' prices
prices = data['Close']

# Minimum data check
if len(prices) < 30:
    st.error("❌ Not enough data to make predictions.")
    st.stop()

# Create lagged dataset
N = 5
lagged_data = pd.DataFrame()
for i in range(N, 0, -1):
    lagged_data[f'lag_{i}'] = prices.shift(i)
lagged_data['target'] = prices
lagged_data.dropna(inplace=True)

# Train-test split
train_size = int(0.8 * len(lagged_data))
train_data = lagged_data.iloc[:train_size]
test_data = lagged_data.iloc[train_size:]

X_train = train_data.drop(columns='target')
y_train = train_data['target']
X_test = test_data.drop(columns='target')
y_test = test_data['target']

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Test prediction
y_pred_test = model.predict(X_test)

# Future prediction
last_N = prices[-N:].values
future_predictions = []
current_features = last_N

for _ in range(pred_days):
    current_features_reshaped = np.array(current_features).reshape(1, -1)  # ✅ FIXED HERE
    next_pred = model.predict(current_features_reshaped)[0]
    future_predictions.append(next_pred)
    current_features = np.roll(current_features, -1)
    current_features[-1] = next_pred

# Future dates
future_dates = pd.date_range(start=prices.index[-1] + pd.Timedelta(days=1), periods=pred_days, freq='B')

# Plot
fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(x=prices.index, y=prices, mode='lines', name='Historical', line=dict(color='blue')))

# Test prediction
fig.add_trace(go.Scatter(x=test_data.index, y=y_pred_test, mode='lines', name='Test Prediction', line=dict(color='green', dash='dot')))

# Future prediction
fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Future Prediction', line=dict(color='red', dash='dash')))

fig.update_layout(
    title=f"{stock_symbol} Stock Price Prediction",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    legend_title="Legend",
    hovermode='x unified'
)

st.plotly_chart(fig)
