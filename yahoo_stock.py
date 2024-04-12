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
    model = ARIMA(data, order=(5,1,0))
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

        st.subheader('Time Series Data')
        st.write(data.head(5)) # displaying the first 5 rows

        # Plot original time series
        st.subheader('Original Time Series Plot')
        fig, ax = plt.subplots()  # Create Matplotlib figure and axis objects
        ax.plot(data.index, data.iloc[:, 0])  # Plot the time series data
        st.pyplot(fig) 

        # Perform forecasting
        forecast = arima_forecast(data, steps)

        # Plot forecasted values
        st.subheader('Forecasted Value')
        plt.plot(data.index, data, label='Original')
        plt.plot(pd.date_range(start=data.index[-1], periods=steps+1, freq='MS')[1:], forecast, label='Forecast', color='red')
        plt.legend()
        st.pyplot()

        # Show forecasted values as a table
        forecast_datas = pd.date_range(start=data.index[-1], periods=steps+1, freq='MS')[1:]
        forecast_df = pd.DataFrame({'Date': forecast_datas, 'Forecast': forecast})
        st.subheader('Forecasted Values (Next {} Steps)'.format(steps))
        st.write(forecast_df)

# Run the app
if __name__ == '__main__':
    main()
