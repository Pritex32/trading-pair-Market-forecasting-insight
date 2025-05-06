


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from PIL import Image
import mplcursors
import ccxt
import os
import pickle
import oandapyV20
import time
import requests
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.exceptions import V20Error
import requests
import datetime
from datetime import date,timedelta

# title of the app
import random


# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'

col1,col2=st.columns(2)
with col1:
     st.title('Pritex32 Market Price Forecasting insight') 
with col2:
     image='forex_streamlit_app/trading chart image.jfif'
     img = Image.open(image)          # load the image
     img = img.resize((100, 100))         # resize the image
     st.image(img)  

st.subheader('Author: Prisca Ukanwa')
#
# forex data download section with time and cache from2021 till present

forex_id = st.text_input('Enter forex pairs in capital letters (e.g. EUR_USD,GBP_USD)', value='GBP_USD')

def fetch_oanda_data_with_cache(instrument=forex_id, granularity='D', start_date='2021-01-01',
                                access_token='', account_id='', cache_file='oanda_cache.csv',
                                sleep_time=2, max_retries=5):
    client = API(access_token=access_token)
    end_dt = pd.Timestamp.utcnow()

    if os.path.exists(cache_file):
        cached_df = pd.read_csv(cache_file, parse_dates=['timestamp'])
        cached_df['timestamp'] = pd.to_datetime(cached_df['timestamp'], utc=True)
        st.write(f"Loaded {len(cached_df)} candles from cache. Latest cached date: {cached_df['timestamp'].max()}")
        start_dt = cached_df['timestamp'].max() + timedelta(seconds=1)
        all_data = cached_df.to_dict('records')
    else:
        st.write("No cache found. Starting from scratch...")
        start_dt = pd.to_datetime(start_date, utc=True)
        all_data = []

    prev_last_time = None

    while start_dt < end_dt:
        window_end = min(start_dt + timedelta(days=490), end_dt)
        params = {
            "granularity": granularity,
            "from": start_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "to": window_end.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "price": "M"
        }

        r = InstrumentsCandles(instrument=instrument, params=params)

        retries = 0
        while retries <= max_retries:
            try:
                response = client.request(r)
                break
            except (V20Error, requests.exceptions.RequestException) as e:
                retries += 1
                wait_time = sleep_time * retries
                st.warning(f"Request error: {e}")
                if retries > max_retries:
                    st.error("Max retries reached. Exiting loop.")
                    return pd.DataFrame(all_data)
                st.warning(f"Retrying in {wait_time} seconds... ({retries}/{max_retries})")
                time.sleep(wait_time)

        candles = response['candles']

        if not candles:
            st.write("No more candles returned by API.")
            break

        for c in candles:
            all_data.append({
                'timestamp': c['time'],
                'open': float(c['mid']['o']),
                'high': float(c['mid']['h']),
                'low': float(c['mid']['l']),
                'close': float(c['mid']['c']),
                'volume': int(c['volume'])
            })

        st.write(f"Fetched {len(candles)} candles from {params['from']} to {params['to']}")

        last_time = pd.to_datetime(candles[-1]['time'], utc=True)

        if prev_last_time is not None and last_time <= prev_last_time:
            st.warning("Last candle time has not moved forward. Stopping to avoid infinite loop.")
            break

        prev_last_time = last_time
        start_dt = last_time + timedelta(seconds=1)

        time.sleep(sleep_time)

    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

    df.to_csv(cache_file, index=False)
    st.success(f"Saved {len(df)} candles to cache file: {cache_file}")

    return df

# Replace with your actual OANDA access token and account ID
ACCESS_TOKEN = 'd917178f8075576c341cbe85848de18e-9575706fb366ffd63dbbd057ecc8d847'
ACCOUNT_ID = '101-004-31663011-001'

# Button to trigger fetching

df = fetch_oanda_data_with_cache(
        instrument=forex_id,
        granularity='D',
        start_date='2021-01-01',
        access_token=ACCESS_TOKEN,
        account_id=ACCOUNT_ID,
        cache_file=f'{forex_id.lower()}_2021_present.csv',
        sleep_time=2,
        max_retries=5
    )
      # show last few rows



# crypto section

# Streamlit input














# to show the data downloaded
col1,col2=st.columns(2)
with col1:
    st.write('Disclaimer: Not a financial advise')
         
with col2:
    st.subheader('Forex Data')
    with st.expander('View All Data',expanded=False):
        st.write('Live Date',df.tail(10))
        st.write('size of the data', df.shape)
    
# select target variable






st.sidebar.header('App Details')
st.sidebar.write('Welcome to the **Future Price Forecasting App**! This application is designed to assist traders and investors in making informed decisions by providing accurate price forecasts for stocks currency pairs, or other financial instruments.')
st.sidebar.write(""" With this tool, you can:
- **Minimize trading losses** by anticipating market trends.
- **Make strategic investment decisions** with data-driven forecasts.
- **Plan for the future** using customizable prediction periods.""")
st.sidebar.subheader('How to Use the App')
st.sidebar.write("""
1. **Enter the Pair ID**: Input the identifier for the  currency pair you want to forecast.
2. **warning**: dont refresh the software too much to avoid request limit.
3. **Charts**: Below are interactive plots for your visualizations .           
4. **Set the forecast Period**: Indicate how many days you want to forecast.
5. **Download the Forecast**: After generating the predictions, you can download the forecasted values in a convenient format by clicking view forecasted data.
6. ** Forecast Type**: We have 3 types of forecast, if all of them are saying the same things , you are on the right track.
                 """)


# contact developer buttion

# Create a button that links to your LinkedIn
st.sidebar.markdown(
    """
    <a href="https://www.linkedin.com/in/prisca-ukanwa-800a1117a/" target="_blank">
        <button style='background-color: #0072b1; color: white; padding: 10px 20px; border: none; border-radius: 5px;'>
            Contact Developer
        </button>
    </a>
    """,
    unsafe_allow_html=True
)


df.dropna(inplace=True)
df.drop_duplicates(inplace=True)


## convert date to index

# Convert the index to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)



rollmean=df['open'].rolling(50).mean() # the moving average

# plotting the Open column
st.subheader('Price vs moving Average')

fig=px.line(df, x=df.index, y='open',title='Open price yearly chart',labels={'open': 'Open Price'},text='open')
fig.add_scatter(x=df.index, y=rollmean, mode='lines', name='50 MA')
st.plotly_chart(fig)


# selecting target column
open_price=df[['open']]






scaler=MinMaxScaler(feature_range=(0,1))
# scaling the data
scaler_data=scaler.fit_transform(open_price)
# loading the model

# feature sequences
x=[]
y=[]
for i in range (60,len(scaler_data)):
      x.append(scaler_data[i-60:i])
      y.append(scaler_data[i])
x=np.array(x)
y=np.array(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# build the layers
from keras .models import Sequential
from keras.layers import Dense,Dropout,LSTM

lstm=Sequential()
lstm.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
lstm.add(Dropout(0.5))

lstm.add(LSTM(units=50,return_sequences=True))
lstm.add(Dropout(0.5))
lstm.add(LSTM(units=50,return_sequences=True))
lstm.add(Dropout(0.5))
lstm.add(LSTM(units=50,return_sequences=False))
lstm.add(Dropout(0.5))
lstm.add(Dense(units=1))

# compile
lstm.compile(optimizer='adam',loss='mean_absolute_error')

model=lstm.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=32,epochs=1)



pred=lstm.predict(x_test)

inv_pred=scaler.inverse_transform(pred)

inv_y_test=scaler.inverse_transform(y_test)


#plotting the predicted vs actual values
from statsmodels.tsa .api import SimpleExpSmoothing

st.subheader('Actual values vs model predicted to validate accuracy')
fig9=plt.figure(figsize=(15,10))
fit1=SimpleExpSmoothing(inv_pred).fit(smoothing_level=0.02,optimized=False)
sns.lineplot(fit1.fittedvalues,color='red',label='Predicted values')
fit2=SimpleExpSmoothing(inv_y_test).fit(smoothing_level=0.02,optimized=False)
sns.lineplot(fit2.fittedvalues,color='blue',label='Actual')
plt.legend()
st.pyplot(fig9)

from sklearn.metrics import mean_squared_error
# the mean squared error
rmse=np.sqrt(mean_squared_error(inv_y_test,inv_pred))
st.write('mean_squared_error',rmse)





# putting the date range for future prediction

st.markdown('<h1><mark>FORECAST</mark></h1>', unsafe_allow_html=True)


n_period = st.number_input(label='How many days of forecast?', min_value=1, value=30)



future_days = pd.date_range(start= df.index[-1] + pd.Timedelta(days=1), periods=n_period,freq='D')


        
    
# values to predict 

last_days=scaler_data[-60:]
last_days=last_days.reshape(1,60,1)
future_prediction=[]
for _ in range(n_period):
    nxt_pred=lstm.predict(last_days)
    future_prediction.append(nxt_pred[0,0])
    last_days = np.append(last_days[:, 1:, :], [[[nxt_pred[0, 0]]]], axis=1)
    last_days[0,-1,0]=nxt_pred[0,0]
    
forecast_array=np.array(future_prediction)
future_prediction=scaler.inverse_transform(forecast_array.reshape(-1,1))


# wether scaled values or x_test,it still the same thing




# dataframe for the forecated values
f_df = pd.DataFrame({'dates':future_days,
                     'open':future_prediction.flatten()}) 

f_df['dates']=pd.to_datetime(f_df['dates'])    

st.subheader('⬇️ stack Forecasted values')
with st.expander('View stack predicted values',expanded=False ):
    st.write('Click the Arrow ontop of this table to download this predictions')
    st.dataframe(f_df)

# Create Plotly Express line plot
fig2 = px.line(
    f_df,
    x='dates',
    y='open',
    title='stack Forecasted Results',
    labels={'open': 'Forecasted Values', 'dates': 'Date'},
    markers=True
)

# Customize marker and line colors
fig2.update_traces(
    line=dict(color='red', width=2),
    marker=dict(color='yellow', size=8, line=dict(color='red', width=1))
)

# Rotate x-axis labels and show grid
fig2.update_layout(
    xaxis_tickangle=-45,
    xaxis_title='Date',
    yaxis_title='open',
    legend_title='Legend',
    showlegend=True
)
st.plotly_chart(fig2)



# bidirectional forecast
from keras.layers import Bidirectional
# values to predict 
bi_lstm=Sequential()
bi_lstm.add(Bidirectional(LSTM(units=100,return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2]))))
bi_lstm.add(Bidirectional(LSTM(units=100,return_sequences=False)))
bi_lstm.add(Dropout(0.3))
bi_lstm.add(Dense(units=1))

bi_lstm.compile(optimizer='adam',loss='mean_squared_error')
model_bi=bi_lstm.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=32,epochs=1)

bi_last_days=scaler_data[-60:]
bi_last_days=bi_last_days.reshape(1,60,1)
future_prediction_bi=[]
for _ in range(n_period):
    nxt_pred=bi_lstm.predict(bi_last_days)
    future_prediction_bi.append(nxt_pred[0,0])
    bi_last_days = np.append(bi_last_days[:, 1:, :], [[[nxt_pred[0, 0]]]], axis=1)
    bi_last_days[0,-1,0]=nxt_pred[0,0]
    
forecast_array=np.array(future_prediction_bi)
future_prediction_bi=scaler.inverse_transform(forecast_array.reshape(-1,1))


# wether scaled values or x_test,it still the same thing




# dataframe for the forecated values
bi_df = pd.DataFrame({'dates':future_days,
                     'open':future_prediction_bi.flatten()}) 

bi_df['dates']=pd.to_datetime(bi_df['dates'])    
col1,col2=st.columns(2)
with col1:
    st.subheader('Bidirectional Forecasted values')
    st.write('Click the Arrow ontop of this table to download this predictions')
    with st.expander('View predicted values',expanded=False):
        st.dataframe(bi_df)

# Create Plotly Express line plot
with col2:
    fig45 = px.line(
    bi_df,
    x='dates',
    y='open',
    title='Bidirectional Forecasted Results',
    labels={'open': 'Forecasted Values', 'dates': 'Date'},
    markers=True)

# Customize marker and line colors
    fig45.update_traces(
    line=dict(color='green', width=2),
    marker=dict(color='red', size=8, line=dict(color='red', width=1)))

# Rotate x-axis labels and show grid
    fig45.update_layout(
    xaxis_tickangle=-45,
    xaxis_title='Date',
    yaxis_title='open',
    legend_title='Legend',
    showlegend=True)
    st.plotly_chart(fig45)



## CNN_LSTM model
from keras .layers import Conv1D,MaxPooling1D,Flatten

cnn_lstm=Sequential()
cnn_lstm.add(Conv1D(filters=64,kernel_size=3,activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
cnn_lstm.add(MaxPooling1D(2))
cnn_lstm.add(LSTM(units=50))
cnn_lstm.add(Dropout(0.3))
cnn_lstm.add(Dense(units=1))

cnn_lstm.compile(optimizer='adam',loss='mse')

cnn_model=cnn_lstm.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=32,epochs=1)


cnn_lstm_last_days=scaler_data[-60:]
cnn_lstm_last_days=cnn_lstm_last_days.reshape(1,60,1)
future_prediction_cnn=[]
for _ in range(n_period):
    nxt_pred1=bi_lstm.predict(cnn_lstm_last_days)
    future_prediction_cnn.append(nxt_pred1[0,0])
    cnn_lstm_last_days = np.append(cnn_lstm_last_days[:, 1:, :], [[[nxt_pred1[0, 0]]]], axis=1)
    
    
forecast_array_cnn=np.array(future_prediction_cnn)
future_prediction_cnn1=scaler.inverse_transform(forecast_array_cnn.reshape(-1,1))


# wether scaled values or x_test,it still the same thing




# dataframe for the forecated values
cnn_lstm_df = pd.DataFrame({'dates':future_days,
                     'open':future_prediction_cnn1.flatten()}) 

cnn_lstm_df['dates']=pd.to_datetime(cnn_lstm_df['dates'])    
col1,col2=st.columns(2)
with col1:
    st.subheader('CNN_LSTM Forecasted values')
    st.write('Click the Arrow ontop of this table to download this forecast')
    with st.expander('View cnn_lstm forecasted values',expanded=False):
        st.dataframe(cnn_lstm_df)

# Create Plotly Express line plot
with col2:
    fig456 = px.line(
    cnn_lstm_df,
    x='dates',
    y='open',
    title='cnn_lstm Forecasted Result',
    labels={'open': 'Forecasted Values', 'dates': 'Date'},
    markers=True)

# Customize marker and line colors
    fig456.update_traces(
    line=dict(color='brown', width=2),
    marker=dict(color='orange', size=8, line=dict(color='red', width=1)))

# Rotate x-axis labels and show grid
    fig456.update_layout(
    xaxis_tickangle=-45,
    xaxis_title='Date',
    yaxis_title='open',
    legend_title='Legend',
    showlegend=True)
    st.plotly_chart(fig456)












# forecasted vs actual section

st.subheader('stacked Forecasted value and original data')


# plotting the entire df with the forecated values
# Create figure and axis
fig3 = plt.figure(figsize=(15, 8))
sns.lineplot(data=df, x=df.index, y='open', label='Original values')# Plot original values
# Plot forecasted values
sns.lineplot(data=f_df, x='dates', y='open', label='Forecasted values', color='red')  
plt.title('Days of Predictions, Full View')# Set plot title and formatting
plt.xticks(rotation='vertical')
plt.grid(True)
plt.legend()
# Make the plot interactive
cursor= mplcursors.cursor(hover=True)
cursor.connect("add", lambda sel: sel.annotation.set_text(f"X: {sel.target[0]}\nY: {sel.target[1]}"))
plt.show()
# Display in Streamlit
st.pyplot(fig3)












df= df.copy()
df.index = pd.to_datetime(df.index)
# Resample Open price

monthly_open = df['open'].resample('M').mean().dropna()
weekly_open = df['open'].resample('W').mean().dropna()
daily_open = df['open'].resample('D').mean().dropna()
four_hour_open = df['open'].resample('4H').mean().dropna() # null values are droped becoz when resampling the data produces null values
hourly_open = df['open'].resample('1H').mean().dropna()

colors = {
    'monthly': '#FF4136',  # red
    'weekly': '#2ECC40',   # green
    'daily': '#0074D9',    # blue
    '4h': '#FF851B',       # orange
    '1h': '#B10DC9'        # purple
}

col1,col2=st.columns(2)
with col1:
    st.subheader('monthly chart')
    fig_monthly = px.line(x=monthly_open.index, y=monthly_open.values, title='Monthly Average Open Price')
    fig_monthly.update_traces(line_color=colors['monthly'])
    st.plotly_chart(fig_monthly)

with col2:
    st.subheader('Weekly Chart')
    fig_weekly = px.line(x=weekly_open.index, y=weekly_open.values, title='Weekly Average Open Price', labels={'x': 'Date', 'y': 'Open Price'})
    fig_weekly.update_traces(line_color=colors['weekly'])
    st.plotly_chart(fig_weekly)

col1,col2=st.columns(2)
with col1:
    st.subheader('Daily Chart')
    fig_daily = px.line(x=daily_open.index, y=daily_open.values, title='Daily Average Open Price', labels={'x': 'Date', 'y': 'Open Price'})
    fig_daily.update_traces(line_color=colors['daily'])
    st.plotly_chart(fig_daily)

# 4-Hour chart
with col2:
    st.subheader('4-Hour Chart')
    fig_4h = px.line(x=four_hour_open.index, y=four_hour_open.values, title='4-Hour Average Open Price', labels={'x': 'Date', 'y': 'Open Price'})
    fig_4h.update_traces(line_color=colors['4h'])
    st.plotly_chart(fig_4h)

# Hourly chart
st.subheader('Hourly Chart')
fig_hourly = px.line(x=hourly_open.index, y=hourly_open.values, title='Hourly Average Open Price', labels={'x': 'Date', 'y': 'Open Price'})
fig_hourly.update_traces(line_color=colors['1h'])
st.plotly_chart(fig_hourly)

# plotting open and close chart
col1,col1=st.columns(2)
with col1:
     volume=plt.figure(figsize=(15,10))
     sns.lineplot(df['Volume'],color='grey',marker='*',markerfacecolor='white',label='Volume prices over time')
     st.pyplot(volume)
    
    
with col2:
     high=plt.figure(figsize=(15,10))
     sns.lineplot(df['low'],label='Low prices over time',color='black')
     st.pyplot(high)

# Addding footer

st.markdown("""
    <hr style="border:1px solid #ddd;">
    <div style="text-align: center;">
        © 2025 pritex. All rights reserved. | Pristomac forecasting software
    </div>
""", unsafe_allow_html=True)


# Ensure date input is valid before filtering
# selecting the open column from the main df

# plotting the entire df by filtering  to get a close view


