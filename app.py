import os
from isort import file
from matplotlib import ticker
from numpy import real
import streamlit as st
import pandas as pd
import joblib

from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from vnstock import listing_companies, stock_historical_data
from RNN import RNNRegressor
from utils import *
from LSTM import LSTM, predict_future
from LSTM1 import LSTM1

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F

st.title('Stock Price Predictions')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')


def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize','Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()



@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

company_data = listing_companies(live=True)
cp_list = company_data['ticker'].tolist()

# option = st.sidebar.text_input('Enter a Stock Symbol', value='VIC')
option = st.sidebar.selectbox('Enter a Stock Symbol', cp_list)
option = option.upper()

today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=3000)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)
if st.sidebar.button('Comfirm'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
       
    else:
        st.sidebar.error('Error: End date must fall after start date')

start_date = start_date.strftime('%Y-%m-%d')
end_date = end_date.strftime('%Y-%m-%d')


data = stock_historical_data(symbol=option, start_date=start_date, end_date=end_date, resolution="1D", type="stock", beautify=True, decor=True, source='DNSE')
# data = download_data(option, start_date, end_date)

data['Change'] = data['Close'].pct_change()
data.iloc[0, data.columns.get_loc('Change')] = 0

data = expand_data(data)

ticker = option
filename = option + '.csv'
filename = os.path.join('Data', filename)
data.to_csv(filename, index = False)

scaler = StandardScaler()

def tech_indicators():
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Close - open', 'BB', 'MACD', 'RSI', 'SMA', 'EMA'])

    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # Creating a new dataframe
    bb = bb[['Close', 'bb_h', 'bb_l']]
    # MACD
    macd = MACD(data.Close).macd()
    # RSI
    rsi = RSIIndicator(data.Close).rsi()
    # SMA
    sma = SMAIndicator(data.Close, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(data.Close).ema_indicator()

    if option == 'Close - open':
        temp = data[['Close', 'Open']]
        st.write('Close - Open Price')
        st.line_chart(temp)
    elif option == 'BB':
        st.write('BollingerBands')
        st.line_chart(bb)
    elif option == 'MACD':
        st.write('Moving Average Convergence Divergence')
        st.line_chart(macd)
    elif option == 'RSI':
        st.write('Relative Strength Indicator')
        st.line_chart(rsi)
    elif option == 'SMA':
        st.write('Simple Moving Average')
        st.line_chart(sma)
    else:
        st.write('Expoenetial Moving Average')
        st.line_chart(ema)


def dataframe():
    st.header('Company Data')
    st.dataframe(company_data)
    st.header('Stock Data')
    st.dataframe(data)


def predict():
    mode = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'KNeighborsRegressor', 'LSTM'])
    num = st.number_input('How many days forecast?', value=5)
    num = int(num)
    if st.button('Predict'):
        # if mode == 'LSTM':
        #     if os.path.exists(os.path.join('Model', ticker, mode + '.pth')):
                
        #         temp = os.path.join('Model', ticker, mode + '.pth')
        #         model = model.load_state_dict(torch.load(temp))
        #         next_days(model,num)
                
        if os.path.exists(os.path.join('Model', ticker, mode + '.pkl')):
            model = joblib.load(os.path.join('Model', ticker, mode + '.pkl'))
            next_days(model,num)
        else: 
            model_engine( num, mode)

def model_engine( num, mode):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
   
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.02)
    
    if mode == 'LinearRegression':
        model = LinearRegression()
    elif mode == 'RandomForestRegressor':
        model = RandomForestRegressor()
    elif mode == 'KNeighborsRegressor':
        model = KNeighborsRegressor()

    
    # training the model

    if mode == 'LSTM':
        y_test, preds, model, forecast_pred = LSTM1(data,num  )
    else:
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        forecast_pred = model.predict(x_forecast)
    
        # saving the model
        folder_path = os.path.join('Model', ticker)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        model_name = mode +'.pkl'
        model_name = os.path.join(folder_path, model_name)
        joblib.dump(model, model_name)
        
    # visualizing the data
    df = pd.DataFrame({'Actual': y_test, 'Predicted': preds})
    st.line_chart(df)
    
    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)} \
            \nMSE: {mean_squared_error(y_test, preds)}')
 
 
    # predicting stock price based on the number of days

    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1
        
def next_days(model,num):
    
   # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
   
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.02)
    preds = model.predict(x_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': preds})
    st.line_chart(df)
    
    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)} \
            \nMSE: {mean_squared_error(y_test, preds)}')
 
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1
    
if __name__ == '__main__':
    main()
