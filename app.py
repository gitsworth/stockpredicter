import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
from datetime import datetime, timedelta

# Streamlit app title and description
st.title("Stock Trend Predictor")
st.write("This app predicts future stock prices based on historical data using a linear regression model.")
st.write("Select a stock symbol, the number of years of historical data, and the number of days to predict into the future.")
st.write("Note: This is a simple model and should not be used for actual investment decisions.")

# Sidebar inputs
stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL", help="Enter a valid stock ticker symbol (e.g., AAPL for Apple)")
num_years = st.sidebar.slider("Years of historical data", 1, 10, 5, help="Choose how many years of past data to analyze")
pred_days = st.sidebar.number_input("Days to predict", 1, 30, 30, help="Choose how many days into the future to predict")

# Calculate start and end dates
end_date = datetime.today()
start_date = end_date - timedelta(days=365 * num_years)

# Fetch historical data
data = yf.download(stock_symbol, start=start_date, end=end_date)
if data.empty:
    st.error("No data found for the given stock symbol.")
    st.stop()

# Extract 'Adj Close' prices
prices = data['Adj Close']

# Check for minimum data requirement
if len(prices) < 30:
    st.error("Not enough data available to make predictions. Please select a stock with more historical data.")
    st.stop()

# Create lagged features
N = 5  # Number of lagged days
lagged_data = pd.DataFrame()
for i in range(N, 0, -1):
    lagged_data[f'lag_{i}'] = prices.shift(i)
lagged_data['target'] = prices
lagged_data = lagged_data.dropna()

# Split data into train and test
train_size = int(0.8 * len(lagged_data))
train_data = lagged_data.iloc[:train_size]
test_data = lagged_data.iloc[train_size:]

X_train = train_data[[f'lag_{i}' for i in range(N, 0, -1)]]
y_train = train_data['target']
X_test = test_data[[f'lag_{i}' for i in range(N, 0, -1)]]
y_test = test_data['target']

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred_test = model.predict(X_test)

# Generate future predictions recursively
last_N = prices[-N:].values
future_predictions = []
current_features = last_N
for _ in range(pred_days):
    next_pred = model.predict([current_features])[0]
    future_predictions.append(next_pred)
    current_features = np.roll(current_features, -1)
    current_features[-1] = next_pred

# Generate future dates
last_date = prices.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=pred_days, freq='B')

# Create Plotly figure
fig = go.Figure()

# Historical data
fig.add_trace(go.Scatter(x=prices.index, y=prices, mode='lines', name='Historical', line=dict(color='blue')))

# Test predictions
fig.add_trace(go.Scatter(x=test_data.index, y=y_pred_test, mode='lines', name='Test Predictions', line=dict(color='green', dash='dash')))

# Future predictions
fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Future Predictions', line=dict(color='red', dash='dot')))

# Add vertical lines to mark transitions
test_start = test_data.index[0]
fig.add_vline(x=test_start, line=dict(color='gray', dash='dash'))
fig.add_vline(x=last_date, line=dict(color='gray', dash='dash'))

# Update layout
fig.update_layout(
    title=f"{stock_symbol} Stock Price Prediction",
    xaxis_title="Date",
    yaxis_title="Adjusted Close Price",
    legend_title="Legend",
    hovermode='x unified'
)

# Display the figure
st.plotly_chart(fig)
