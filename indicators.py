"""
calculate different technical indicators  (c) 2017 BAOFENG ZHANG
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import datetime as dt
import os
import math
from util import get_data, plot_data


    
def compute_prices(sd = dt.datetime(2008,1,1),ed = dt.datetime(2009,12,31),lookback = 21,syms = ['AAPL'], gen_plot = False):
    #Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd,ed)
        
    prices_all = get_data(syms,dates) #automatically adds SPY
    prices = prices_all[syms] #only portfolio symbols
    prices_SPY = prices_all['SPY'] #only SPY, for comparision later
    
    
    
    return prices, prices_all, prices_SPY
    
###calculate the stochastic indicator
def stochastic_indicator(price, syms = ['AAPL'], lookback = 21):
    ###
    stoch_max = price.rolling(window = lookback, min_periods = lookback).max()
    stoch_min = price.rolling(window = lookback, min_periods = lookback).min()
    
    stochk = 100 * (price - stoch_min) / (stoch_max - stoch_min)
    stochd = stochk.rolling(window = 3, min_periods = 3).mean()
    stochk = stochk.fillna(method = 'bfill')
    stochd = stochd.fillna(method = 'bfill')
    
    return stochk, stochd
    

#create sma array, zero it out    
def sma_indicator(price, syms = ['AAPL'], lookback = 21):
    ###calculate SMA-21 for the entire date range for all symbols
    sma = price.rolling(window = lookback, min_periods = lookback).mean()
    sma = sma.fillna(method='bfill')
    
    return sma
    
###Turn SMA into Price/SMA ratio
def price_sma_indicator(price):
        
    sma = sma_indicator(price)
    price_sma = price / sma
    
    return price_sma  

    
###calculate bollinger bands
def bollinger_band_indicator(price, syms = ['AAPL'], lookback = 21):
    
    
        
    sma = sma_indicator(price)
    #price = compute_prices()
        
    ###calculate bolling bands(21 day) over the entire period
    rolling_std = price.rolling(window = lookback, min_periods = lookback).std()
    #rolling_std = pd.rolling_std(price, window = lookback, min_periods = lookback)
    top_band = sma + (2 * rolling_std)
    bottom_band = sma - (2 * rolling_std)
        
    bbp = (price - bottom_band) / (top_band - bottom_band)
    bbp = bbp.fillna(method = 'bfill')
    
    return bbp,top_band, bottom_band
    
###Calculate relative strength, then RSI
def rsi_indicator(price, syms = ['AAPL'], lookback = 21):
        
    #price, price_SPY = compute_prices()
    rsi = price.copy() 
    
    daily_rets = price.copy()
    daily_rets.values[1:,:] = price.values[1:,:] - price.values[:-1,:]
    daily_rets.values[0,:] = np.nan

    ###final vectorize code
    up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
    down_rets = -1 * daily_rets[daily_rets < 0].fillna(0).cumsum()
        
    up_gain = price.copy()
    up_gain.ix[:,:] = 0
    up_gain.values[lookback:,:] = up_rets.values[lookback:,:] - up_rets.values[:-lookback,:]

    down_loss = price.copy()
    down_loss.ix[:,:] = 0
    down_loss.values[lookback:,:] = down_rets.values[lookback:,:] - down_rets.values[:-lookback,:]

    #Now we can  calculate the RS and RSI all at once
    rs = (up_gain / lookback) / (down_loss / lookback)
    rsi = 100 - (100 / (1 + rs))
    rsi.ix[:lookback,:] = np.nan
    
        
    #Inf results mean down_loss was 0. Those should be RSI 100
    rsi[rsi == np.inf] = 100
    rsi = rsi.fillna(method = 'bfill')
    
    return rsi
    
### Use the three indicators to make some kind of trading decision for each day
def build_orders(syms = ['AAPL'], lookback = 21):
        
        
    price, price_all,price_SPY = compute_prices()
        
    sma = price_sma_indicator(price_all)
    bbp,top_band,bottom_band = bollinger_band_indicator(price_all)
    rsi = rsi_indicator(price_all)
    
        
    # Orders starts as a NaN array of the same shape/index as price
    orders = price.copy()
    orders.ix[:,:] = np.NaN

    #Create a copy of RSI but with the SPY column copied to all columns
    spy_rsi = rsi.copy()
    spy_rsi.values[:,:] = spy_rsi.ix[:,['SPY']]

    ###create a binary(0-1) array showing when price is above SMA-21
    sma_cross = pd.DataFrame(0, index=sma.index, columns=sma.columns)
    sma_cross[sma >= 1] = 1

    ###Turn that array into one that only shows the crossings (-1 == croos down, +1 == cross up)
    sma_cross[1:] = sma_cross.diff()
    sma_cross.ix[0] = 0

    ##now we can calculate the results of entire strategy at once
    #Apply our entry order conditions all at once. This represents our TARGET SHARES
    #at this moment in time, not an actual order
    orders[(sma < 0.95) & (bbp < 0) & (rsi < 30) & (spy_rsi > 30)] = 200
    orders[(sma > 1.05) & (bbp > 1) & (rsi > 70) & (spy_rsi < 70)] = -200
               
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
    #Dump the orders to stdout. (redirect to a file if you wish)
    for order in order_list:
        print "  ".join(str(x) for x in order)

def standardization_indicator(df):
    df_stand = (df - df.mean()) / df.std()
    return df_stand

def normalization_indicator(df):
    df_norm = df / df.ix[0,:]
    return df_norm

def plot_indicator(df, title = "Stock Prices"):
    
    #sma = sma_indicator()
    
    ax = df.plot(title = title, fontsize = 12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    #plt.show()

def plot_indicator_two_graph(df1,df2,df3,df4,file_name):
    
    plt.clf()
    fig,axes = plt.subplots(nrows=3,ncols=1)
    axes[0].plot(df1)
    axes[0].tick_params(labelbottom='off')
    axes[1].plot(df2)
    axes[1].tick_params(labelbottom='off')
    axes[2].plot(df3)
    plt.xticks(rotation=30)
    

def plot_stoch_indicator(price,sma,stochk,file_name):
    
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.set(title = 'AAPL with Stochastic Indicator')
    ax.plot(price,label = 'AAPL Price')
    ax.plot(sma,label = 'SMA(21)')
    #ax.plot(upper_band, label = 'upper_band', color='r')
    #ax.plot(bottom_band,label = 'bottom_band',color='r')
    ax.set_ylabel('Price')
    #ax.set_ylim([0,2])
    #start, end = ax.get_xlim()
    #ax.xaxis.set_ticks(np.arange(start,end,90))
    L = plt.legend()
    L.get_texts()[0].set_text('AAPL Price')
    L.get_texts()[0].set_text('SMA(21)')
    #L.get_texts()[1].set_text('Upper_band')
    #L.get_texts()[2].set_text('Lower_band')
    ax.tick_params(labelbottom='off')
    ax.legend(loc = 'upper center',bbox_to_anchor = (0.55,1.0), prop = {'size':7})
    ###create font properties
    #fontP = FontProperties()
    #fontP.set_size('small')
    #plt.legend([ax], "title",prop = fontP)
    
    ax1 = fig.add_subplot(212)
    ax1.plot(stochk,label = 'stochk',color = 'r')
    #ax1.plot(stochd,label = 'stochd', color = 'green')
    ax1.set_ylabel('Stochastic Oscillator')
    ax1.set_xlabel('Date')
    ax1.axhline(20.0, color = 'black')
    ax1.axhline(80.0, color = 'black')
    #start1, end1 = ax1.get_xlim()
    #ax1.xaxis.set_ticks(np.arange(start,end,90))
    fig.subplots_adjust(hspace=0.05)#,left = 0.2, right = 1.0, top = 0.2, bottom = 0.1)
    plt.xticks(rotation=30)
    plt.savefig(file_name)
    #plt.show()
    
def plot_price_sma_indicator(price,sma,price_sma, file_name):
    
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.set(title = 'AAPL with Price/SMA Indicator')
    ax.plot(price, label = 'AAPL Price')
    ax.plot(sma, label = 'SMA')
    
    #ax = price.plot(title = 'AAPL with Price/SMA Indicator', label = 'price')
    #sma.plot(label = 'SMA',ax = ax)
    #price_sma.plot(label = 'Price/SMA', ax = ax)
    #ax.set_xlabel('Date')
    ax.set_ylabel('Price/SMA Indicator')
    #ax.set_ylim([0,2])
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start,end,90))
    
    L = plt.legend()
    L.get_texts()[0].set_text('AAPL Price')
    L.get_texts()[1].set_text('SMA')
    #L.get_texts()[2].set_text('Price/SMA')
    ax.tick_params(labelbottom='off')
    ax.legend(loc = 'upper center',bbox_to_anchor = (0.55,1.0), prop = {'size':7})
    
    ax1 = fig.add_subplot(212)
    ax1.plot(price_sma,label = 'Price/SMA',color = 'r')
    ax1.set_ylabel('Price/SMA')
    ax1.set_xlabel('Date')
    ax1.axhline(0.95, color = 'black')
    ax1.axhline(1.05, color = 'black')
    fig.subplots_adjust(hspace=0.1)
    
    plt.xticks(rotation=30)
    plt.savefig(file_name)
    #plt.show()
    
def plot_bollinger_band_indicator(price,sma,upper_band, bottom_band,bbp, file_name):
    
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.set(title = 'AAPL with Bollinger Band Indicator')
    ax.plot(price,label = 'AAPL Price')
    ax.plot(sma,label = 'SMA(21)')
    ax.plot(upper_band, label = 'upper_band', color='r')
    ax.plot(bottom_band,label = 'bottom_band',color='r')
    ax.set_ylabel('Price')
    #ax.set_ylim([0,2])
    #start, end = ax.get_xlim()
    #ax.xaxis.set_ticks(np.arange(start,end,90))
    L = plt.legend()
    L.get_texts()[0].set_text('AAPL Price')
    L.get_texts()[1].set_text('SMA(21)')
    L.get_texts()[2].set_text('Upper_band')
    L.get_texts()[3].set_text('Lower_band')
    ax.tick_params(labelbottom='off')
    ax.legend(loc = 'upper center',bbox_to_anchor = (0.55,1.0), prop = {'size':7})
    ###create font properties
    #fontP = FontProperties()
    #fontP.set_size('small')
    #plt.legend([ax], "title",prop = fontP)
    
    ax1 = fig.add_subplot(212)
    ax1.plot(bbp,label = 'bbp',color = 'r')
    ax1.set_ylabel('Bollinger Band (%BB)')
    ax1.set_xlabel('Date')
    ax1.axhline(0.0, color = 'black')
    ax1.axhline(1.0, color = 'black')
    fig.subplots_adjust(hspace=0.1)
    #start, end = ax.get_xlim()
    #ax1.xaxis.set_ticks(np.arange(start,end,90))
    plt.xticks(rotation=30)
    plt.savefig(file_name)
    #plt.show()

def plot_rsi_indicator(price,sma,rsi, file_name):
    
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.set(title = 'AAPL with RSI Indicator')
    ax.plot(price,label = 'AAPL Price')
    ax.plot(sma,label = 'SMA(21)')
    #ax.plot(upper_band, label = 'upper_band', color='r')
    #ax.plot(bottom_band,label = 'bottom_band',color='r')
    ax.set_ylabel('Price')
    #ax.set_ylim([0,2])
    #start, end = ax.get_xlim()
    #ax.xaxis.set_ticks(np.arange(start,end,90))
    L = plt.legend()
    L.get_texts()[0].set_text('AAPL Price')
    L.get_texts()[1].set_text('SMA(21)')
    #L.get_texts()[1].set_text('Upper_band')
    #L.get_texts()[2].set_text('Lower_band')
    ax.tick_params(labelbottom='off')
    ax.legend(loc = 'upper center',bbox_to_anchor = (0.55,1.0), prop = {'size':7})
    ###create font properties
    #fontP = FontProperties()
    #fontP.set_size('small')
    #plt.legend([ax], "title",prop = fontP)
    
    ax1 = fig.add_subplot(212)
    ax1.plot(rsi,label = 'rsi',color = 'r')
    ax1.set_ylabel('Relative Strength Index')
    ax1.set_xlabel('Date')
    ax1.axhline(30.0, color = 'black')
    ax1.axhline(70.0, color = 'black')
    #start1, end1 = ax1.get_xlim()
    #ax1.xaxis.set_ticks(np.arange(start,end,90))
    fig.subplots_adjust(hspace=0.1)
    plt.xticks(rotation=30)
    plt.savefig(file_name)
    #plt.show()

def test_run():
    
    #df = price_sma_indicator()
    #df = bollinger_band_indicator()
    price,price_all, price_SPY = compute_prices()
    sma = sma_indicator(price)
    price_sma = price_sma_indicator(price)
    
    #price_st = standardization_indicator(price)
    #price_SPY_st = standardization_indicator(price_SPY)
    #sma_st = standardization_indicator(sma)
    #plot_price_sma_indicator(price_st,price_SPY_st,sma_st,'sma.png')
    
    #plot price/SMA figure
    price_n = normalization_indicator(price)
    price_sma_n = normalization_indicator(price_sma)
    sma_n = normalization_indicator(sma)
    plot_price_sma_indicator(price_n,sma_n,price_sma,'sma.png')
    
    #plot bolling_band figure
    bbp, upper_band, bottom_band = bollinger_band_indicator(price)
    bbp_n = normalization_indicator(bbp)
    upper_band_n = normalization_indicator(upper_band)
    bottom_band_n = normalization_indicator(bottom_band)
    plot_bollinger_band_indicator(price,sma,upper_band,bottom_band,bbp,'bbp.png')
    
    #plot relative strength index
    rsi = rsi_indicator(price)
    plot_rsi_indicator(price_n,sma_n,rsi,'rsi.png')
    
    
    ###standardization sma, bb, rsi
    rsi_n = normalization_indicator(rsi)
    sma_st = standardization_indicator(sma)
    price_sma_st = standardization_indicator(price_sma)
    bbp_st = standardization_indicator(bbp)
    rsi_st = standardization_indicator(rsi)
    #rsi_st = (rsi - rsi.mean())/rsi.std()
    #print rsi.mean()
    #print rsi.std()
    """
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(sma_st)
    axes.plot(price_sma_st)
    #print bbp
    axes.plot(bbp_st)
    axes.plot(rsi_st)
    #print rsi
    """
    stochk,stochd = stochastic_indicator(price)
    plot_stoch_indicator(price_n,sma_n,stochk,'stoch.png')
    #bbp = bollinger_band_indicator(price)
    #bbp = rsi_indicator(price)
    #df = df[21:,]

    #df = df.fillna(method='bfill')
    #df, df_SPY = compute_prices()
    #df = df/df.ix[0,:]
    #print df
    #price_SPY_st = standardization_indicator(price_SPY)
    #price_st = standardization_indicator(price)
    #sma_st = standardization_indicator(sma)
    #bbp_st = standardization_indicator(bbp)
    #price = price/price.ix[0,:]
    #df = df/df.ix[0,:]
    #plot_indicator_two_graph(price_st,price_SPY_st,bbp_st,'rsi.png')
    #plot_indicator(df)
    build_orders()
    
    
    
    
        

if __name__ =="__main__":
    #build_orders()
    #price,price_all,price_SPY = compute_prices()
    #build_orders()
    test_run()
    print 'good job'
        
        
        
        
    
    
        