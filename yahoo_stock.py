# Import the necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.title("Yahoo Stock Time Series Forecasting")
# Function to load time series data
def load_data(file):
    data = pd.read_csv("C:/Users/black/Downloads/archive/yahoo_stock.csv")
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    return data

# Function to perform ARIMA forecasting
def arima_forecast( data, steps):
    close_data = data['Close']
    model = ARIMA(close_data, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Main function to run the app
def main ():
    st.title('Time Series Forecasting App')

    # Sidebar for file upload and forecast steps input
    st.sidebar.title('Settings')
    file = st.sidebar.file_uploader('Upload CSV file', type=['csv'])
    steps = st.sidebar.number_input('Forecast Steps', min_value=1, value=12)

    if file is not None:
        # Load data
        data = load_data(file)

        st.subheader("Data")
        st.write(data.head(5)) # displaying the first 5 rows

        # Plot original time series
        st.subheader('Original Time Series Plot')
        fig, ax = plt.subplots()  # Create Matplotlib figure and axis objects
        ax.plot(data.index, data['Close'], label ='Close Price', color='blue') 
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.legend()
        st.pyplot(fig) 

        # Add section divider
        st.markdown("---")

        # plot Open vs Close
        st.subheader('Open vs Close')
        fig, ax = plt.subplots()  # Create Matplotlib figure and axis objects
        ax.plot(data.index, data['Close'], label ='Close Price', color='blue') 
        ax.plot(data.index, data['Open'], label = 'Open Price', color='green', linestyle='--')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        st.pyplot(fig) 

        # Add interactive widget for forecasting
        st.sidebar.title("Forecasting Settings")
        steps = st.sidebar.slider("Number of Forecast Steps", min_value=1, max_value=100, value=12)

        # Perform forecasting
        forecast = arima_forecast(data, steps)

        # Plot Volumn
        st.subheader('Volume Plot')
        fig, ax = plt.subplots()
        ax.plot(data.index, data['Volume'], label='Volumn')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volume')
        ax.legend()
        st.pyplot(fig)

        # Add section divider
        st.markdown("---")

        # Plot forecasted values
        st.subheader('Forecasted Value')
        plt.plot(data.index, data, label='Original')
        forecast_dates = pd.date_range(start=data.index[-1], periods=steps+1, freq='MS')[1:]
        fig, ax = plt.subplots()
        ax.plot(data.index, data['Close'], label='Close Price', color='blue')
        ax.plot(forecast_dates, forecast, label='Forecast', color='red', linestyle='--')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        plt.legend()
        st.pyplot(fig)

        # Show forecasted values as a table
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
        st.subheader('Forecasted Values (Next {} Steps)'.format(steps))
        st.write(forecast_df)

# Run the app
if __name__ == '__main__':
    main()
