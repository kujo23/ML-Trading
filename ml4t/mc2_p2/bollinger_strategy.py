import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

def symbol_to_path(symbol, base_dir=os.path.join("..", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates, addSPY=True):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def get_rolling_mean(values,window):
    #Return rolling mean of given values, using window size
    return pd.rolling_mean(values,window)

def get_rolling_std(values,window):
    #Return rolling std dev of given values, using window size
    return pd.rolling_std(values,window)

def get_bollinger_bands(rm,rstd):
    #Return Bollinger bands
    upper_band = rm + rstd*2
    lower_band = rm - rstd*2
    return upper_band,lower_band


def test_run():
    """Driver function."""
    # Define input parameters

    start_date = '2007-12-31'
    end_date = '2009-12-31'
    dates = pd.date_range(start_date,end_date)

    symbols = ['IBM']
    df = get_data(symbols,dates)

    rm = get_rolling_mean(df['IBM'],window = 20)
    rstd = get_rolling_std(df['IBM'],window = 20)

    upper_band, lower_band = get_bollinger_bands(rm,rstd)

    short_entries = []
    short_exits = []
    long_entries = []
    long_exits = []
    signals = []

    entryfound = False
    for i in range(len(upper_band.index)):
        if df['IBM'][i-1] > upper_band.loc[upper_band.index[i-1]] and df['IBM'][i] < upper_band.loc[upper_band.index[i]] and not entryfound:
            short_entries.append(str(upper_band.index[i]))
            entryfound = True
            signals.append([str(upper_band.index[i]).split()[0],'SELL'])
        elif df['IBM'][i-1] > rm.loc[upper_band.index[i-1]] and df['IBM'][i] < rm.loc[upper_band.index[i]] and entryfound:   
            short_exits.append(upper_band.index[i])
            entryfound = False
            signals.append([str(upper_band.index[i]).split()[0],'BUY'])
        elif df['IBM'][i-1] < lower_band.loc[upper_band.index[i-1]] and df['IBM'][i] > lower_band.loc[upper_band.index[i]] and not entryfound:
            long_entries.append(upper_band.index[i])
            entryfound = True
            signals.append([str(upper_band.index[i]).split()[0],'BUY',])
        elif df['IBM'][i-1] < rm.loc[upper_band.index[i-1]] and df['IBM'][i] > rm.loc[upper_band.index[i]] and entryfound:
            long_exits.append(upper_band.index[i])
            entryfound = False
            signals.append([str(upper_band.index[i]).split()[0],'SELL'])

    ordersfile = open('orders.csv','w')
    ordersfile.write("Date,Symbol,Order,Shares\n")
    for signal in signals:
        ordersfile.write("%s,IBM,%s,100\n"%(signal[0],signal[1]))

    ordersfile.close()


    #Plotting of the Graph

    ax = df['IBM'].plot(title='Bollinger Bands',label='IBM')
    rm.plot(label='SMA',ax = ax)
    upper_band.plot(label = 'Upper Bollinger Bands',ax = ax, color='c')
    lower_band.plot(label = 'Lower Bollinger Bands',ax = ax, color='c')

    ymin,ymax = ax.get_ylim()
    plt.vlines(long_entries,ymin,ymax,color='g')
    plt.vlines(long_exits,ymin,ymax)
    plt.vlines(short_entries,ymin,ymax,color='r')
    plt.vlines(short_exits,ymin,ymax)

    ax.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    test_run()
