import yfinance as yf
import pandas as pd
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import plotly.express as px
import requests
from bs4 import BeautifulSoup

@st.cache_data
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

def get_plot(df):
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
    df = df.dropna()
    return df

def get_data(data):
# Calculate moving averages and other indicators
    data['MA_10'] = data['Close'].rolling(window=10).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_100'] = data['Close'].rolling(window=100).mean()
    decomposition = seasonal_decompose(data['Close'], period=365)
    data['Seasonal'] = decomposition.seasonal
    days = 10
    data['EWM10'] = data['Close'].ewm(span=days).mean()
    data = data[["Close", 'Seasonal', "MA_10", "MA_50", "MA_100", "EWM10"]]
    data = data.dropna()
    data.index = pd.to_datetime(data.index)
    return data


# Train-test split
def train_model(data, columns): #
# Prepare data for training and testing
    X = data[columns]
    y = data['Close'].shift(-1).dropna()
    X = X[:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
    # Train a model
    model = LinearRegression()
    model.fit(X_train, y_train)
# Extract model parameters
    coefficients = model.coef_
    intercept = model.intercept_
    return coefficients, intercept

def simulate_model(intercept, coefficients, X, initial_balance, threshold_percentage):
    # Initialize balance and position
    # Initialize balance and position
    X = X[['Close', 'Seasonal', 'MA_10', 'MA_50', 'MA_100', 'EWM10']]
    balance = initial_balance
    position = 0  # Number of shares
    trades = []
    
    # Iterate over each data point in the entire dataset
    for i in range(len(X)):
        current_date = X.index[i]
        current_price = round(X.iloc[i]['Close'], 2)
        
        # Calculate predicted price using model parameters
        predicted_price = intercept + sum(coefficients * X.iloc[i])
        predicted_price = round(predicted_price, 2)
    
        # Calculate threshold values
        upper_threshold = current_price * (1 + threshold_percentage)
        lower_threshold = current_price * (1 - threshold_percentage)
    
        # Buy decision
        if predicted_price > upper_threshold and balance >= current_price:
            shares_to_buy = int(balance // current_price)  # Buy whole shares only
            if shares_to_buy > 0:
                position += shares_to_buy
                balance -= shares_to_buy * current_price
                trades.append([current_date, 'Buy', shares_to_buy, current_price, predicted_price, int(balance + position * current_price)])
    
        # Sell decision
        elif predicted_price < lower_threshold and position > 0:
            balance += position * current_price
            trades.append([current_date, 'Sell', position, current_price, predicted_price, int(balance)])
            position = 0
    
    # Convert trades list to DataFrame
    trades_df = pd.DataFrame(trades, columns=['Date', 'Action', 'Stocks', 'Price', 'Predict_Price', 'Balance'])
    
    # Calculate buy-and-hold final balance
    buy_and_hold_shares = initial_balance // X.iloc[0]['Close']
    buy_and_hold_final_balance = buy_and_hold_shares * X.iloc[-1]['Close']
    
    # Determine if the trading strategy was "Good" or "Bad"
    final_balance = balance + (position * X.iloc[-1]['Close'])
    profit = final_balance - initial_balance
    today_price = round(X.iloc[-1]['Close'],2)
    first_price = round(X.iloc[0]['Close'],2)
    prices_list = [today_price, first_price]
    
    return trades_df, initial_balance, final_balance, profit, balance, trades, buy_and_hold_final_balance, predicted_price, prices_list


def plot_operations(data, trades):
# Add stock price line plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    
    # Add scatter plot for trades
    for trade in trades:
        if trade[1] == 'Buy':
            fig.add_trace(go.Scatter(x=[trade[0]], y=[trade[3]], mode='markers', marker=dict(symbol='triangle-up', color='green'), name='Buy'))
        elif trade[1] == 'Sell':
            fig.add_trace(go.Scatter(x=[trade[0]], y=[trade[3]], mode='markers', marker=dict(symbol='triangle-down', color='red'), name='Sell'))
    
    if len(trades) > 1:
         #Update legend to show only first three items
        fig.update_traces(showlegend=False)  # Hide all legends initially
        fig.data[0].showlegend = True  # Show legend for the first trace (Close Price)
        fig.data[1].showlegend = True  # Show legend for the second trace (Buy)
        fig.data[2].showlegend = True  # Show legend for the third trace (Sell)
    
    # Update layout
    fig.update_layout(
        title='Stock Price and Trade Actions',
        xaxis_title='Date',
        yaxis_title='Price',
        showlegend=True,
    )
    # Show plot
    return fig

#%% Streamlit
def main():
    st.set_page_config(layout="wide")
    sp500_com, dic_sp500, dic_sp500_2, list_500 = get_sp500()
    marketcap = get_MarketCap()
    sp500_df = pd.merge(marketcap, sp500_com, on='Symbol', how='inner')
    st.title("Stock App (S&P500)")
    st.markdown("""Stock App is a sophisticated tool for monitoring the S&P 500. Utilizing yfinance for data retrieval and Plotly for visualizations, our app offers comprehensive stock data and interactive charts to identify market trends. \n\n
                \n\nAdditionally, Stock App features a prediction model using **linear regression** with **sklearn**, allowing you to simulate trading actions and compare them to the buy-and-hold strategy.\n\n
                \n\nEnhance your investment decisions with cutting-edge insights from Stock App.
                """)
    col1,col2,col3,col4,col5 = st.columns(5)
    with col2:
        list_500.append('<select>')
        default_ix = list_500.index('MET')
        stock_selected = st.selectbox('**Stock**', list_500, index=default_ix)
    with col3:
        initial_amount_selected = st.number_input("Initial Amount (USD)", value=1000, placeholder="e.g 1000")
    with col4:
        time_period = ['<select>', 5,6,7,8,9,10]
        default_ix = time_period.index(10)
        time_period_selected = st.selectbox('**Period** (Years)', time_period, index=default_ix)
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
    data = download_data(stock_ticker, start, end)
    new_data = get_data(data)
    coefficients, intercept = train_model(new_data, ['Close', 'Seasonal', 'MA_10', 'MA_50', 'MA_100', 'EWM10'])
    threshold_percentage_selected = 1/1000 #It can be an input
    trades_df,initial_balance, final_balance, profit, balance, trades, buy_and_hold_final_balance, predicted_price, prices_list = simulate_model(intercept,coefficients, data, initial_amount_selected, threshold_percentage_selected)
    with left_column:
        st.subheader("")
        variacion = "{:,.2f}".format(get_variation(prices_list[0], prices_list[1]))
        st.metric("Current Price", "${:,.2f}".format(prices_list[0]),f'{variacion}%')
    with middle_column:
        st.subheader("")
        st.metric("Market Cap", f'$ {format_number(market_cap)}')
    with right_column:
        st.subheader("")
        st.metric("Volume", f'${format_number(volume)}')
        
#We will show the 
    st.header('Operations using the Algorithym Advices')
    col1,col2,col3, col4, col5 = st.columns([10,1,7,1,7])
    with col1:
        st.plotly_chart(plot_operations(data, trades))
    with col3:
        trades_df['Date'] = pd.to_datetime(trades_df['Date']) 
        # Extract only the date part
        trades_df['Date'] = trades_df['Date'].dt.date
        st.dataframe(trades_df)
    with col5:
        var_final_balance = "{:,.2f}".format(get_variation(final_balance, initial_amount_selected))
        st.metric("Final Balance", f'$ {format_number(round(final_balance,2))}', f'{var_final_balance}%')
        st.metric("Profit", f'$ {format_number(round(profit,2))}')
        var_buy_hold = "{:,.2f}".format(get_variation(buy_and_hold_final_balance, initial_amount_selected))
        st.metric("Buy and Hold Balance", f'$ {format_number(round(buy_and_hold_final_balance,2))}', f'{var_buy_hold}%')
        st.metric("Predicted Price Today", f'$ {format_number(round(predicted_price,2))}')



    

    st.header(f'Trend Analysis  ({time_period_selected} Years)')
    st.line_chart(get_plot(data))
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


