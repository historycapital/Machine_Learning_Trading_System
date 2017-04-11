"""
A simple wrapper for market simulator.  (c) 2017 BAOFENG ZHANG
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
from best_strategy import *
from indicators import *

def author():
    return 'bzhang367'

def compute_portvals(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), orders_file = "./order.csv", start_val = 100000):
    # this is the function the autograder will call to test your code
    #parse the data from orders.csv and save in the frame orders.
    orders = pd.read_csv(orders_file, index_col ='Date', 
             parse_dates = True, 
             #usecols = ['Date','Symbol','Order','Shares'], 
             na_values = ['nan'])
    #sort the dates
    orders = orders.sort_index()
    #apply 21 day rule
    #for i in range(orders.shape[0]):
    #    date = orders.index[i]
    #    sym = orders.ix[i]['Symbol']
    #    if orders.ix[i,'Order'] == 'BUY' 
    #get symbols from orders
    syms = orders['Symbol'].unique().tolist()

    #sd = orders.index[0]
    #ed = orders.index[-1]
    #sd = dt.datetime(2010,1,1)
    #ed = dt.datetime(2011,12,31)

    dates = pd.date_range(sd, ed)    
    prices_all = get_data(syms, dates)  # automatically adds SPY    
    prices = prices_all[syms]  # only portfolio symbols
    #prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    prices['Cash'] = 1.0

    #create a dataframe trades to calculate the trading information
    #set the initial values for different stock to 0
    trades = prices.copy()
    for sym in syms:
        trades[sym] = 0
    trades['Cash'] = 0
    buyline = []
    sellline= []

    for i in range(orders.shape[0]):
        date = orders.index[i]
        sym  = orders.ix[i]['Symbol']
        if orders.ix[i,'Order'] == 'BUY':
            buyline.append(date)
            trades.ix[date,sym] += orders.ix[i,'Shares']
            trades.ix[date,'Cash'] -= prices.ix[date,sym] * orders.ix[i,'Shares']
            #print(prices.ix[date,sym])
        elif orders.ix[i,'Order'] == 'SELL':
            sellline.append(date)
            trades.ix[date,sym] -= orders.ix[i,'Shares']
            trades.ix[date,'Cash'] += prices.ix[date,sym] * orders.ix[i,'Shares']

    #create a dataframe holdings to calculate the holdings for each day
    holdings = trades.copy()
    holdings.ix[0,'Cash'] += start_val

    for j in range(1,holdings.shape[0]):
        s = 0
        t = 0
        leverage = 0 
        for sym in syms:
            holdings.ix[j,sym] += holdings.ix[j-1,sym]
            s += abs(holdings.ix[j,sym] * prices.ix[j,sym])
            t += holdings.ix[j,sym] * prices.ix[j,sym]
        holdings.ix[j,'Cash'] = holdings.ix[j,'Cash'] + holdings.ix[j-1,'Cash']
        #leverage = s/(t + holdings.ix[j,'Cash'])
        #if leverage > 1.5 :
        #    holdings.ix[j,:] = holdings.ix[j-1,:]

    #create a datafram values to calculate the values for all holdings
    values = prices * holdings

    values['portfolio'] = values.sum(axis = 1)

    portvals = values[['portfolio']]
    
    return portvals, buyline,sellline


def compute_portfolio_stats(prices,rfr = 0.0, sf = 252.0,sv=1):
    #cr: Cumulative return
    #adr: Average daily return
    #sddr: Standard deviation of daily return
    #sr: Sharpe Ratio
    #added on 1/29/2017 code to compute daily portforlio values
    normed_price = prices / prices.ix[0,:]
    alloced_price = normed_price
    pos_vals = alloced_price * sv
    port_val = pos_vals#.sum(axis=1)
    #addition ended on 1/29/2017

    daily_returns = (port_val / port_val.shift(1)) - 1
    daily_returns = daily_returns[1:]
    cr   = (port_val[-1] / port_val[0]) - 1.0
    adr  = daily_returns.mean()
    sddr = daily_returns.std()
    sr   = np.sqrt(sf) * (adr - rfr) / sddr
    return cr,adr,sddr,sr
#add ended on 3/3/2017

def plot_rule_based_strategy(price,benchmark,buyline,sellline,file_name):
    
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(title = 'AAPL with Rule-based Strategy')
    ax.plot(price,label = 'AAPL Price',color = 'blue')
    ax.plot(benchmark,label = 'Benchmark', color = 'black')
    ax.set_ylabel('Price')
    ax.set_xlabel('Date')
    #ax.set_ylim([0,4])
    #start, end = ax.get_xlim()
    #ax.xaxis.set_ticks(np.arange(start,end,90))
    ax.legend(loc = 'upper center',bbox_to_anchor = (0.55,1.0), prop = {'size':7})
    #ax.legend(loc = 'upper center',prop = {'size':7})
    plt.xticks(rotation=30)
    for l in buyline:
        plt.axvline(x = l,color = 'green')
    for sl in sellline:
        plt.axvline(x = sl, color = 'red')
    
    plt.savefig(file_name)
    #plt.show()

def plot_RL_based_strategy(RL_portvals,rule_portvals, benchmark,buyline,sellline,file_name):
    
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set(title = 'AAPL with Rule-based Strategy')
    ax.plot(RL_portvals,label = 'RL_based Portfolio',color = 'green')
    ax.plot(rule_portvals, label = 'Rule_based Portfolio', color = 'blue')
    ax.plot(benchmark,label = 'Benchmark', color = 'black')
    ax.set_ylabel('Price')
    ax.set_xlabel('Date')
    #ax.set_ylim([0,4])
    #start, end = ax.get_xlim()
    #ax.xaxis.set_ticks(np.arange(start,end,90))
    ax.legend(loc = 'upper center',bbox_to_anchor = (0.55,1.0), prop = {'size':7})
    #ax.legend(loc = 'upper center',prop = {'size':7})
    plt.xticks(rotation=30)
    for l in buyline:
        plt.axvline(x = l,color = 'green')
    for sl in sellline:
        plt.axvline(x = sl, color = 'red')
    
    plt.savefig(file_name)
    
def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./order.csv"
    sv = 100000

    # Process orders
    #portvals,prices_SPY = compute_portvals(orders_file = of, start_val = sv)
    portvals,buyline,sellline = compute_portvals(orders_file = of, start_val = sv)
    #print(portvals)
    if isinstance(portvals, pd.DataFrame):
        #prices_SPY = portvals[portvals.columns[1]]
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    #print(portvals)
    ###plot the 
    benchmark = build_benchmark()
    benchmark = benchmark[benchmark.columns[0]]
    benchmark_n = normalization_indicator(benchmark)
    portvals_n = normalization_indicator(portvals)
    plot_rule_based_strategy(portvals_n, benchmark_n, buyline,sellline,'rule_based.png')
    
    
    ###simulate the RL_simulation
    portvals_RL, buy_RL,sell_RL = compute_portvals(orders_file = './order_RL.csv', start_val = sv)
    portvals_RL = portvals_RL[portvals_RL.columns[0]]
    portvals_RL_n = normalization_indicator(portvals_RL)
    plot_RL_based_strategy(portvals_RL_n,portvals_n,benchmark_n,buy_RL,sell_RL,'RL_based.png')
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = portvals.index[0]
    end_date = portvals.index[-1]
    
    #start_date = dt.datetime(2008,1,1)
    #end_date = dt.datetime(2008,6,1)
    #added on 3/3/2017
    #symbols = ['IBM', 'X', 'GLD']
    # Assess the portfolio
    
    ###Part 6 Comparative analysis
    portvals_out, buy_out,sell_out = compute_portvals(sd = dt.datetime(2010,1,1), ed = dt.datetime(2011,12,31), orders_file = './order_out.csv', start_val = sv)
    portvals_out = portvals_out[portvals_out.columns[0]]
    portvals_out_n = normalization_indicator(portvals_out)
    
    benchmark_out = build_benchmark(sd = dt.datetime(2010,1,1), ed = dt.datetime(2011,12,31))
    benchmark_out = benchmark_out[benchmark_out.columns[0]]
    benchmark_out_n = normalization_indicator(benchmark_out)
    
    portvals_RL_out, buy_RL_out,sell_RL_out = compute_portvals(sd = dt.datetime(2010,1,1), ed = dt.datetime(2011,12,31), orders_file = './order_RL_out.csv', start_val = sv)
    portvals_RL_out = portvals_RL_out[portvals_RL_out.columns[0]]
    portvals_RL_out_n = normalization_indicator(portvals_RL_out)
    plot_RL_based_strategy(portvals_RL_out_n,portvals_out_n,benchmark_out_n,buy_RL_out,sell_RL_out,'RL_based_out.png')
    
    cr, adr, sddr, sr = compute_portfolio_stats(portvals_RL_out)
    crSPY,adrSPY,sddrSPY, srSPY = compute_portfolio_stats(portvals_out)
    crbench, adrbench, sddrbench,srbench = compute_portfolio_stats(benchmark_out)
    #add ended on 3/3/2017
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = cr, adr, sddr, sr#[0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = crSPY,adrSPY,sddrSPY, srSPY#[0.2,0.01,0.02,1.5]
    cum_ret_bench,avg_daily_ret_bench,std_daily_ret_bench,sharpe_ratio_bench = crbench, adrbench, sddrbench, srbench
    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of RL_based Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of Rule_based Fund: {}".format(sharpe_ratio_SPY)
    print "Sharpe Ratio of Benchmark: {}".format(sharpe_ratio_bench)
    print
    print "Cumulative Return of RL_based Fund: {}".format(cum_ret)
    print "Cumulative Return of Rule_based Fund: {}".format(cum_ret_SPY)
    print "Cumulative Return of Benchmark: {}".format(cum_ret_bench)
    print
    print "Standard Deviation of RL_based Fund: {}".format(std_daily_ret)
    print "Standard Deviation of Rule_based Fund: {}".format(std_daily_ret_SPY)
    print "Standard Deviation of Benchmark: {}".format(std_daily_ret_bench)
    print
    print "Average Daily Return of RL_based Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of Rule_based Fund: {}".format(avg_daily_ret_SPY)
    print "Average Daily Return of benchmark: {}".format(avg_daily_ret_bench)
    print
    print "Final RL_based Portfolio Value: {}".format(portvals_RL_out[-1])
    print "Final Rule_based Portfolio Value: {}".format(portvals_out[-1])
    print "Final Benchmark Portfolio Value: {}".format(benchmark_out[-1])
    
    
if __name__ == "__main__":
    test_code()
