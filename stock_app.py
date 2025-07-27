import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import matplotlib.pyplot as plt

st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")
st.title("📉 Stock Price Prediction using LSTM")

# Load pre-trained model (.h5)
model = load_model('C:/Users/karan/OneDrive/Documents/Stock price/stock_model.h5')

# Sidebar Inputs
stock = st.sidebar.text_input("Enter Stock Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", dt.date(2010, 1, 1))
end_date = dt.datetime.now()

# Load Data
@st.cache_data
def load_stock_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    df = df.reset_index()
    return df

df = load_stock_data(stock, start_date, end_date)

if df.empty:
    st.warning("No data found for the given stock symbol.")
    st.stop()

st.subheader(f"📄 Stock Data for {stock}")
st.dataframe(df.tail())

csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download CSV", csv, f"{stock}_data.csv", "text/csv")

# Plot Close Price
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df['Date'], df['Close'], label='Close Price', color='blue')
ax1.set_title(f"{stock} Close Price from {start_date} to Today")
ax1.set_xlabel("Date")
ax1.set_ylabel("Close Price")
ax1.legend()
st.pyplot(fig1)

# Prepare data
features = ['Close']
data = df[features]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Train/Test Split
train_size = int(len(scaled_data) * 0.70)
test_data = scaled_data[train_size - 100:]
close_index = features.index('Close')

# Sequence Creation
def create_sequences(data, seq_len=100, target_index=3):
    x, y = [], []
    for i in range(seq_len, len(data)):
        x.append(data[i-seq_len:i])
        y.append(data[i, target_index])
    return np.array(x), np.array(y)

x_test, y_test = create_sequences(test_data, target_index=close_index)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], len(features)))

# Predict test set
predicted_prices = model.predict(x_test)
predicted_prices = predicted_prices.reshape(-1, 1)

pred_array = np.zeros((len(predicted_prices), len(features)))
pred_array[:, close_index] = predicted_prices[:, 0]
y_test_array = np.zeros((len(y_test), len(features)))
y_test_array[:, close_index] = y_test

predicted_prices_rescaled = scaler.inverse_transform(pred_array)[:, close_index]
actual_prices_rescaled = scaler.inverse_transform(y_test_array)[:, close_index]

# Latest Prediction
st.subheader("📊 Latest Prediction")
st.metric(label="Next-Day Predicted Close Price", value=f"${predicted_prices_rescaled[-1]:.2f}")

# Plot Actual vs Predicted
fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(actual_prices_rescaled, label='Actual Price', color='green')
ax2.plot(predicted_prices_rescaled, label='Predicted Price', color='red')
ax2.set_title(f"{stock} Actual vs Predicted Prices")
ax2.set_xlabel("Time")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

# 🔮 Future Forecast
st.subheader("🔮 Predict Future Stock Prices")
n_days = st.slider("Select number of future days to predict", min_value=1, max_value=30, value=7)

future_input = scaled_data[-100:].reshape(1, 100, len(features))
future_predictions = []

for _ in range(n_days):
    pred = model.predict(future_input, verbose=0)[0][0]

    new_row = np.array([pred if i == close_index else future_input[0, -1, i] for i in range(future_input.shape[2])])
    next_input = np.vstack([future_input[0, 1:], new_row])
    future_input = np.expand_dims(next_input, axis=0)

    future_predictions.append(pred)

# Rescale future predictions
future_array = np.zeros((n_days, len(features)))
future_array[:, close_index] = future_predictions
future_prices = scaler.inverse_transform(future_array)[:, close_index]

# Display and Plot
future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_days)
future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Close Price": future_prices
})
st.write("### 📅 Future Predicted Prices")
st.dataframe(future_df)

fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(future_df["Date"], future_df["Predicted Close Price"], marker='o', linestyle='--', color='purple')
ax3.set_title(f"{stock} - Next {n_days} Days Predicted Close Prices")
ax3.set_xlabel("Date")
ax3.set_ylabel("Predicted Price")
st.pyplot(fig3)

csv_future = future_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download CSV", csv_future, f"{stock}_future_predictions.csv", "text/csv")
