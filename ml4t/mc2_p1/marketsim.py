"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import os, csv

from util import get_data, plot_data
from portfolio.analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data

def compute_portvals(start_date, end_date, orders_file, start_val):
    """Compute daily portfolio value given a sequence of orders in a CSV file.

    Parameters
    ----------
        start_date: first date to track
        end_date: last date to track
        orders_file: CSV file to read orders from
        start_val: total starting cash available

    Returns
    -------
        portvals: portfolio value for each trading day from start_date to end_date (inclusive)
    """
    # TODO: Your code here

    reader = csv.reader(open(orders_file,'rU'),delimiter= ',')
    next(reader)
    dates = pd.date_range(start_date, end_date)
    symbols = []
    for row in reader:
        symbols.append(row[1])

    symbols = list(set(symbols))

    df_prices = get_data(symbols,dates)
    df_prices['CASH'] = 1.00

    # Calculate the Trading table
    df_trade = df_prices.copy()
    df_trade[df_trade != 0] = 0.0

    reader = csv.reader(open(orders_file,'rU'),delimiter= ',')
    next(reader)
    for row in reader:
        date, symbol, order, share = row[0], row[1], row[2], row[3]

        if (order == 'BUY'):
            df_trade[symbol][date] = df_trade[symbol][date] + np.float64(share)
        else:
            df_trade[symbol][date] = df_trade[symbol][date] - 1.0*np.float64(share)

    df_temp = -1.0*df_prices*df_trade
    df_trade['CASH'] = df_temp.sum(axis=1)


    # Calculate the Holdings table
    df_holdings = df_trade.copy()
    df_holdings['CASH'][0] = start_val

    for i in range(1,len(df_holdings['CASH'])):
        df_holdings['CASH'][i] = df_holdings['CASH'][i-1] + df_trade['CASH'][i-1]
        for symbol in symbols :
            df_holdings[symbol][i] = df_holdings[symbol][i] + df_holdings[symbol][i-1]

    df_holdings['CASH'][1:-1] = df_holdings['CASH'][2:]

    # Calculate the portvals
    df_temp = df_holdings*df_prices
    portvals = df_temp.sum(axis=1)
    portvals[0] = start_val
    portvals[-1] = df_trade['CASH'][-1] + portvals[-1]

    return portvals


def test_run():
    """Driver function."""
    # Define input parameters
    start_date = '2010-01-01'
    end_date = '2010-12-31'
    orders_file = os.path.join("orders", "orders_outsample.csv")
    start_val = 10000

    # Process orders
    portvals = compute_portvals(start_date, end_date, orders_file, start_val)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series

    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    # Simulate a $SPX-only reference portfolio to get stats
    prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['$SPX']]  # remove SPY
    portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    # Compare portfolio against $SPX
    print "Data Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of $SPX: {}".format(sharpe_ratio_SPX)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of $SPX: {}".format(cum_ret_SPX)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of $SPX: {}".format(std_daily_ret_SPX)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of $SPX: {}".format(avg_daily_ret_SPX)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

    # Plot computed daily portfolio value
    df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', 'SPY'], axis=1)
    plot_normalized_data(df_temp, title="IBM Data Out Sample Backtest")


if __name__ == "__main__":
    test_run()
