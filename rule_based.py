"""Manual Rule-Based Trader"""
"""
  (c) 2017 BAOFENG ZHANG
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
from best_strategy import build_benchmark
import csv



### Use the three indicators to make some kind of trading decision for each day
def build_orders(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31),syms = ['AAPL'], lookback = 21, file_name = 'order.csv'):
        
        
    price, price_all,price_SPY = compute_prices(sd = sd, ed = ed)
        
    sma = price_sma_indicator(price_all)
    bbp,top_band,bottom_band = bollinger_band_indicator(price_all)
    rsi = rsi_indicator(price_all)
    stoch,stochd = stochastic_indicator(price_all)
    #rsi_SPY = rsi_indicator(price_SPY)
    
    # Orders starts as a NaN array of the same shape/index as price
    orders = price.copy()
    orders.ix[:,:] = np.NaN

    #Create a copy of RSI but with the SPY column copied to all columns
    spy_rsi = rsi.copy()
    spy_rsi.values[:,:] = spy_rsi.ix[:,['SPY']]

    #Create a copy of STOCH but with the SPY column copied to all columns
    spy_stoch = stoch.copy()
    spy_stoch.values[:,:] = spy_rsi.ix[:,['SPY']]

    ###create a binary(0-1) array showing when price is above SMA-21
    sma_cross = pd.DataFrame(0, index=sma.index, columns=sma.columns)
    sma_cross[sma >= 1] = 1

    ###Turn that array into one that only shows the crossings (-1 == croos down, +1 == cross up)
    sma_cross[1:] = sma_cross.diff()
    sma_cross.ix[0] = 0

    ##now we can calculate the results of entire strategy at once
    #Apply our entry order conditions all at once. This represents our TARGET SHARES
    #at this moment in time, not an actual order
    orders[(sma < 0.95) & (bbp < 0) & (stoch < 20) & (rsi < 30) & (spy_rsi > 30) & (spy_stoch > 20)] = 200
    orders[(sma > 1.05) & (bbp > 1) & (stoch > 80) & (rsi > 70) & (spy_rsi < 70) & (spy_stoch < 80)] = -200
    #orders[(rsi < 30) & (spy_rsi > 30) ] = 200
    #orders[(rsi > 70) & (spy_rsi < 70) ] = -200
    #orders[(bbp < 0) ] = 200
    #orders[(bbp > 1) ] = -200
               
    #Apply our exit order conditions all at once. Again, this represents Target Shares.
    orders[(sma_cross != 0)] = 0
        
    #we now have -200, 0, or +200 target shares on all days that "we care about". (i.e. those
    #days when our strategy tells us something) all other days are NaN, meaning "hold whatever
    #you have".
        
    ###NaN meant "stand pat", so we should forward fill those
    #Forward fill NaNs with previous values, then fill remaining NaNs with 0
    orders.ffill(inplace = True)
    orders.fillna(0, inplace = True)
        
    #we now have a dataframe with our target shares on every day, including holding periods.
    ###But we wanted orders, not target holdings!
    #Now take the diff, which will give us an order to place only when the target shares changed.
    orders[1:] = orders.diff()
    orders.ix[0] = 0

    ###and now we have our orders array, just as we wanted it, with no iteration

    ###Dump the orders to stdout (redirect to a file if you wish)
        
    ###we can at least drop the SPY column
    #del orders['SPY']
    #syms.remove('SPY')
        
    ###And more importantly, drop all rows with no non-zero values(i.e. no orders)
    orders = orders.loc[(orders != 0).any(axis = 1)]
        
    ###Now we have only the days that have orders. That's better, at least!
    order_list = []
    for day in orders.index:
        for sym in syms:
            if orders.ix[day,sym] > 0:
                order_list.append([day.date(), sym, 'BUY', 200])
            elif orders.ix[day,sym] < 0:
                order_list.append([day.date(), sym, 'SELL', 200])
        
        
    with open(file_name,'wb') as csvfile:
        fieldnames = ['Date','Symbol','Order','Shares']
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()
        for order in order_list:
            order_tmp = ",".join(str(x) for x in order)
            csvfile.write(order_tmp + "\n")
            
        

if __name__ =="__main__":
    #build_orders()
    #price,price_all,price_SPY = compute_prices()
    build_orders()
    build_orders(sd = dt.datetime(2010,1,1), ed = dt.datetime(2011,12,31),file_name = 'order_out.csv')
    
    #test_run()
    print 'good job'