import pandas as pd
import streamlit as st
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pickle

start_date = "1996-01-01"
end_date = "2023-05-01"
st.title("Reliance NSE Closing Stock Price Prediction")

model_name = st.sidebar.selectbox("Select Model for prediction", ("ARIMA", "Facebook Prophet", "Stacked LSTM"))

@st.cache_data
def get_dataset():
    # data = yf.download('RELIANCE.NS', period='max', auto_adjust=True)
    data = yf.download('RELIANCE.NS', start="1996-01-01", end="2023-05-01", auto_adjust=True)
    # Split data into training and testing sets
    size = int(len(data.Close) * 0.9)
    train_data = list(data[0:size]['Close'])
    test_data = list(data[size:]['Close'])
    return data, train_data, test_data

data, train_data, test_data = get_dataset()

st.subheader('Raw data')
st.write(data.tail())

st.subheader("Closing Price vs Time chart for entire dataset")
st.line_chart(data.Close, use_container_width=True)

# st.subheader("Closing Price vs Time chart with 100MA")
if model_name=="ARIMA":
    st.subheader(model_name)
    # Loading ARIMA model
    # with open('arima_model.pkl', 'rb') as f:
    #    arima_model = pickle.load(f)
    # Loading ARIMA predictions
    predictions = pd.read_csv("./arima_predictions.csv")

    st.subheader("ARIMA predictions mapping Actual Dataset")
    fig2 = plt.figure(figsize=(20,8), dpi=300)
    date_range = data[int(len(data.Close) * 0.9):].index
    plt.plot(date_range, predictions["actual_data"], color='blue', marker='.', label='Actual')
    plt.plot(date_range, predictions["predictions"], color='red', marker='.', linestyle='--', label='Predictions')
    plt.title("Reliance NSE Closing Stock Price Since 1996")
    plt.xlabel("Date")
    plt.ylabel("Closing Prices (Rs)")
    plt.grid()
    plt.legend()
    st.pyplot(fig2)
    st.text("RMSE: 41.57")
    
    st.subheader("Interactive Plot")
    predictions['Date'] = pd.to_datetime(predictions['Date'])
    # Set the 'Date' column as the index
    predictions = predictions.set_index('Date')
    st.line_chart(predictions, use_container_width=True)
    # st.line_chart(data=predictions, y=["actual_data", "predictions"])
    
    days = st.slider(label="Select days", value=14)
    st.subheader(f"ARIMA predictions for last {days} days")
    fig3 = plt.figure(figsize=(20,8), dpi=300)
    date_range = data[int(len(data.Close) * 0.9):].index
    plt.plot(date_range[-days:], predictions["actual_data"][-days:], color='blue', marker='.', label='Actual')
    plt.plot(date_range[-days:], predictions["predictions"][-days:], color='red', marker='.', linestyle='--', label='Predictions')
    # plt.title("Reliance NSE stock closing price forecast for last 15 days")
    plt.xlabel("Date")
    plt.ylabel("Closing Prices (Rs)")
    plt.grid()
    plt.legend()
    st.pyplot(fig3)
    rmse = np.sqrt(np.square(np.subtract(predictions["actual_data"][-days:], predictions["predictions"][-days:])).mean()).round(2)
    st.write("RMSE: ", rmse)

elif model_name=="Facebook Prophet":
    st.text("Prophet")
else:
    st.subheader(model_name)
    predictions = pd.read_csv("./arima_predictions.csv")

    st.subheader("ARIMA predictions mapping Actual Dataset")
    fig2 = plt.figure(figsize=(20,8), dpi=300)
    date_range = data[int(len(data.Close) * 0.9):].index
    plt.plot(date_range, predictions["actual_data"], color='blue', marker='.', label='Actual')
    plt.plot(date_range, predictions["predictions"], color='red', marker='.', linestyle='--', label='Predictions')
    plt.title("Reliance NSE Closing Stock Price Since 1996")
    plt.xlabel("Date")
    plt.ylabel("Closing Prices (Rs)")
    plt.grid()
    plt.legend()
    st.pyplot(fig2)
    st.text("RMSE: 41.57")
    
    st.subheader("Interactive Plot")
    predictions['Date'] = pd.to_datetime(predictions['Date'])
    # Set the 'Date' column as the index
    predictions = predictions.set_index('Date')
    st.line_chart(predictions, use_container_width=True)
    # st.line_chart(data=predictions, y=["actual_data", "predictions"])
    
    days = st.slider(label="Select days", value=14)
    st.subheader(f"ARIMA predictions for last {days} days")
    fig3 = plt.figure(figsize=(20,8), dpi=300)
    date_range = data[int(len(data.Close) * 0.9):].index
    plt.plot(date_range[-days:], predictions["actual_data"][-days:], color='blue', marker='.', label='Actual')
    plt.plot(date_range[-days:], predictions["predictions"][-days:], color='red', marker='.', linestyle='--', label='Predictions')
    # plt.title("Reliance NSE stock closing price forecast for last 15 days")
    plt.xlabel("Date")
    plt.ylabel("Closing Prices (Rs)")
    plt.grid()
    plt.legend()
    st.pyplot(fig3)
    rmse = np.sqrt(np.square(np.subtract(predictions["actual_data"][-days:], predictions["predictions"][-days:])).mean()).round(2)
    st.write("RMSE: ", rmse)
