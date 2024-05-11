# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf

# Page title
st.title('Stock Market Prediction App')

# Sidebar
st.sidebar.header('User Input Parameters')

# Function to get user input
def get_input():
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2010-01-01'))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime('today'))
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL for Apple)", 'AAPL')
    return start_date, end_date, stock_symbol

# Function to get historical stock data
def get_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

# Function to create and train the model
def create_model(data):
    # Prepare data
    data['Next Close'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Next Close']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    st.write('Mean Squared Error:', mse)

    return model

# Main function
def main():
    start, end, symbol = get_input()
    data = get_data(symbol, start, end)
    model = create_model(data)

    # Display the last 5 rows of the data
    st.subheader('Stock Data')
    st.write(data)

    # Prediction
    st.subheader('Prediction')
    open_price = st.number_input('Enter Open Price', value=data['Open'].iloc[-1])
    high_price = st.number_input('Enter High Price', value=data['High'].iloc[-1])
    low_price = st.number_input('Enter Low Price', value=data['Low'].iloc[-1])
    volume = st.number_input('Enter Volume', value=data['Volume'].iloc[-1])

    # Make prediction
    prediction = model.predict([[open_price, high_price, low_price, volume]])
    st.write('Predicted Close Price:', prediction[0])

if __name__ == '__main__':
    main()

