#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:13:11 2023

@author: felipezenteno
"""

import pandas as pd
import yfinance as yf
import datetime
import streamlit as st
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose


def download_data(stock, start, end):
    data = yf.download(stock, start, end)
    return data

@st.cache_data
def get_sp500():
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(sp500_url)
    sp500_components = sp500_table[0]
    sp500_dict = dict(zip(sp500_components['Security'].tolist(),
                          sp500_components['Symbol'].tolist()))
    sp500_dict2 = dict(zip(sp500_components['Symbol'].tolist(),
                          sp500_components['Security'].tolist()))
    list_names = sp500_components['Security'].tolist() + sp500_components['Symbol'].tolist()
    sp500_components.rename(columns={"GICS Sector": "Industries"}, inplace=True)
    return sp500_components, sp500_dict, sp500_dict2, list_names

def convert_market_cap(market_cap_str):
    if 'B' in market_cap_str:
        return int(float(market_cap_str.replace(',', '').replace('B', '')) * 1e9)
    else:
        return (float(market_cap_str)) 
    
@st.cache_data
def get_MarketCap():
    url = "https://stockanalysis.com/list/sp-500-stocks/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')
    data = []
    headers = []
    for row in table.find_all('tr'):
        cells = row.find_all(['th', 'td'])
        if cells:
            if not headers:
                headers = [cell.text.strip() for cell in cells]
            else:
                data.append([cell.text.strip() for cell in cells])
    df = pd.DataFrame(data, columns=headers)
    df = df[['Symbol', 'Market Cap']]
    df.rename(columns={"Market Cap": "MarketCap"}, inplace=True)
    df['MarketCap'] = df['MarketCap'].apply(convert_market_cap)
    return df

def plot_by_industry(df):
    grouped_df = df.groupby(['Industries', 'Symbol']).sum().reset_index()
    grouped_df = grouped_df.sort_values(by='MarketCap', ascending=False)
    fig = px.bar(grouped_df, x='Industries', y='MarketCap', color='Symbol', title='MarketCap by Industry and Symbol')
    fig.update_layout(showlegend=False)
    return fig

def get_PriceCapVol(ticker_symbol,Period):
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=f'{Period}y')
    current_price = data['Close'].iloc[-1]
    market_cap = ticker.info['marketCap']
    volume = data['Volume'].iloc[-1]
    price_period = data['Close'].iloc[0]
    return current_price, market_cap, volume, price_period


def get_variation(current_price, old_price):
    var_por = ((current_price - old_price) / old_price) * 100
    return var_por


def format_number(number):
    if abs(number) >= 1_000_000_000_000:
        return f"{number / 1_000_000_000_000:.2f}T"
    elif abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f}B"
    elif abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    else:
        return str(number)

def get_plot(stock_ticker, start, end, days):
    df = download_data(stock_ticker, start, end)
    df['Mid'] = (df['High']+df['Low'])/2 #The use of mid or close is arbitrary.
    decomposition = seasonal_decompose(df['Mid'], period=365)
    df['Trend'] = decomposition.trend
    df['Seasonal'] = decomposition.seasonal
    df['Residual'] = decomposition.resid 
    df['MA30'] = df['Mid'].rolling(30).mean()
    df['MA100'] = df['Mid'].rolling(100).mean()
    days = 10
    alpha = 2/(days+1)
    df['EWM10'] = df['Mid'].ewm(span=days).mean()
    df[['Mid','Trend','Seasonal','Residual',"MA30","MA100","EWM10"]].plot(figsize=(20,8))
    #Check Volume
    #df2 = df[['Close','Volume']].plot(subplots=True)
    df = df[['Mid','Trend','Seasonal','Residual',"MA30","MA100","EWM10"]]
    return df


#%% Streamlit
def main():
    st.set_page_config(layout="wide")
    sp500_com, dic_sp500, dic_sp500_2, list_500 = get_sp500()
    marketcap = get_MarketCap()
    sp500_df = pd.merge(marketcap, sp500_com, on='Symbol', how='inner')
    st.title("Stock App (S&P500)")
    st.markdown(""" Stock App is a comprehensive tool for monitoring the S&P 500. Powered by yfinance for data retrieval, Plotly for visualization, and web scraping for insights, it offers stock data, interactive charts, and a useful trending analysis plot to identify market patterns and trends.""")

    col1,col2,col3,col4,col5 = st.columns(5)
    with col2:
        stock_selected = st.selectbox('**Stock**', list_500)
    with col4:
        time_period = [10, 5, 3]
        time_period_selected = st.selectbox('**Period** (Years)', time_period)
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=time_period_selected*365)

    if stock_selected in (sp500_com['Security'].tolist()):
        stock_security = stock_selected
        stock_ticker = dic_sp500[stock_selected]
    if stock_selected in (sp500_com['Symbol'].tolist()):
        stock_security = dic_sp500_2[stock_selected] 
        stock_ticker = stock_selected
    st.markdown(f"<h1 style='text-align: center;'>{stock_security}  ({stock_ticker}) </h1>", unsafe_allow_html=True)
#                       METRICS
    current_price, market_cap, volume, old_price  = get_PriceCapVol(stock_ticker, time_period_selected)
    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.subheader("")
        variacion = "{:,.2f}".format(get_variation(current_price, old_price))
        st.metric("Current Price", "${:,.2f}".format(current_price),f'{variacion}%')
    with middle_column:
        st.subheader("")
        st.metric("Market Cap", f'$ {format_number(market_cap)}')
    with right_column:
        st.subheader("")
        st.metric("Volume", f'${format_number(volume)}')
        
    st.header(f'Trend Analysis  ({time_period_selected} Years)')
    st.line_chart(get_plot(stock_ticker, start, end, 10))
    st.title(" Industries Plots in S&P500")
    
    left_column,x, right_column = st.columns([6,1,8])
    with left_column:  
        #fig = px.pie(sp500_df, values='MarketCap', names='Industries', title='Market Cap by Industry')
        #st.plotly_chart(fig)
        
        fig = go.Figure(data=[go.Pie(labels=sp500_df.Industries,
                                     values=sp500_df.MarketCap,
                                     textinfo='label+percent',
                                     insidetextorientation='radial')])
        fig.update_layout(title_text='MarketCap by Industries')
        fig.update_layout(showlegend=False)
        st.write(fig)
    with right_column:
        st.plotly_chart(plot_by_industry(sp500_df))

    st.markdown("##")
    st.markdown("##")
    st.write("© Copyright 2024 Felipe Zenteno  All rights reserved.")
if __name__ == '__main__':
    main()
    
