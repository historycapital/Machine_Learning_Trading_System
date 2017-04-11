"""
A code to generate best trading strategy.  (c) 2017 BAOFENG ZHANG
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import datetime as dt
import os
import math
from util import get_data, plot_data
from indicators import *

###The performance of a portfolio starting with $100,000 cash, investing in 200 shares of AAPL and holding that position
def build_benchmark(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), syms=['AAPL'], start_val = 100000):
    
    price, price_all,price_SPY = compute_prices(sd = sd,ed = ed)
    benchmark = price.copy()
    cash = start_val - 200 * benchmark.values[0,:].sum()
    benchmark = 200 * benchmark + cash
    print benchmark
    return benchmark

def compute_portfolio_stats(prices,rfr = 0.0, sf = 252.0,sv=1):
    #cr: Cumulative return
    #adr: Average daily return
    #sddr: Standard deviation of daily return
    #sr: Sharpe Ratio
    #added on 1/29/2017 code to compute daily portforlio values
    normed_price = prices / prices.ix[0,:]
    #alloced_price = normed_price
    #pos_vals = alloced_price * sv
    #port_val = pos_vals#.sum(axis=1)
    #addition ended on 1/29/2017
    #daily_returns = prices / prices(1) - 1
    #daily_returns = daily_returns[1:]
    #cr = prices[-1] / prices[0] - 1.0
    
    port_val = normed_price.sum(axis=1)

    daily_returns = (port_val / port_val.shift(1)) - 1
    daily_returns = daily_returns[1:]
    cr   = (port_val[-1] / port_val[0]) - 1.0
    adr  = daily_returns.mean()
    sddr = daily_returns.std()
    sr   = np.sqrt(sf) * (adr - rfr) / sddr
    return cr,adr,sddr,sr
#add ended on 3/3/2017
    
#use the best strategy to make some kind of trading decision for each day
### Use the three indicators to make some kind of trading decision for each day
def build_best_orders(syms = ['AAPL'], lookback = 21, start_val = 100000):
        
        
    #benchmark = build_benchmark()
    #print benchmark
    price, price_all, price_SPY = compute_prices()
    
    daily_rets = price.copy()
    daily_rets.values[:-1,:] = price.values[:-1,:] - price.values[1:,:]
    daily_rets.values[-1,:] = 0 
    #num = daily_rets._get_numeric_data()
    #num[num < 0] = -1 * num[num < 0]
    daily_rets[daily_rets < 0] = -1 * daily_rets[daily_rets < 0] 

    for i in range(daily_rets.shape[0]):
        if i > 0:
            daily_rets.values[i,:] = daily_rets.values[i,:] + daily_rets.values[i-1,:]
    #daily_rets.values[1:,:] = daily_rets.values[1:,:] + daily_rets.values[:-1,:]
    daily_rets = 200 * daily_rets
    daily_rets = start_val + daily_rets
    #print daily_rets
    return daily_rets
    
def plot_best_strategy(price,benchmark,file_name):
    
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(title = 'AAPL with Best Strategy')
    ax.plot(price,label = 'AAPL Price',color = 'blue')
    ax.plot(benchmark,label = 'Benchmark', color = 'black')
    ax.set_ylabel('Price')
    ax.set_xlabel('Date')
    ax.set_ylim([0,4])
    #start, end = ax.get_xlim()
    #ax.xaxis.set_ticks(np.arange(start,end,90))
    #ax.legend(loc = 'upper left',bbox_to_anchor = (0.55,1.0), prop = {'size':7})
    ax.legend(loc = 'upper left',prop = {'size':10})
    plt.xticks(rotation=30)
    plt.savefig(file_name)
    #plt.show()
    
    
def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    benchmark = build_benchmark()
    portvals = build_best_orders()
    benchmark_n = normalization_indicator(benchmark)
    portvals_n = normalization_indicator(portvals)
    plot_best_strategy(portvals_n,benchmark_n,'best_strategy.png')
    #print portvals.ix[-1,:]
    
    #daily_returns = (port_val / port_val.shift(1)) - 1
    #daily_returns = daily_returns[1:]
    #cr   = (port_val[-1] / port_val[0]) - 1.0
    
    #daily_returns = portvals_n / portvals_n.shift(1) - 1
    #daily_returns = daily_returns[1:]
    #cr = 1.0#portvals_n[-1] / portvals_n[0] - 1.0
    #adr  = daily_returns.mean()
    #sddr = daily_returns.std()
    #rfr = 0.0
    #sf = 252
    #sr   = np.sqrt(sf) * (adr - rfr) / sddr
    
    cr, adr, sddr, sr = compute_portfolio_stats(portvals)
    crSPY,adrSPY,sddrSPY, srSPY = compute_portfolio_stats(benchmark)
    #add ended on 3/3/2017
    #cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]#cr, adr, sddr, sr#
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = cr, adr, sddr, sr
    
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = crSPY,adrSPY,sddrSPY, srSPY#
    
    # Compare portfolio against $SPX
    #print "Date Range: {} to {}".format(start_date, end_date)
    #print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of Benchmark : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of Benchmark : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of Benchmark : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of Benchmark : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals.ix[-1])
    
if __name__ == "__main__":
    test_code()