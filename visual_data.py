"""
A simple wrapper for visulization scatter data.  (c) 2017 BAOFENG ZHANG
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
import RTLearner as rtl
import BagLearner as bl
from ML_based import *




def visualization_data(sd = dt.datetime(2008,1,1),ed = dt.datetime(2009,12,31), syms = ['AAPL'], lookback = 21):
    
    price, price_all,price_SPY = compute_prices(sd,ed)
        
    bbp,top_band,bottom_band = bollinger_band_indicator(price)
    stoch, stochd = stochastic_indicator(price)
    rsi = rsi_indicator(price)
             
    rsi_rule = rsi.copy()
    rsi_rule.ix[:,:] = 0
    rsi_rule[(rsi > 70) & (stoch > 80)] = -1
    rsi_rule[(rsi < 30) & (stoch < 20)] = 1
    
    stoch_sd = standardization_indicator(stoch)
    rsi_sd = standardization_indicator(rsi)
    rsi_sd = rsi_sd.rename(columns = {'AAPL':'rsi'})
    stoch_sd = stoch_sd.rename(columns = {'AAPL':'stoch'})
    rsi_rule = rsi_rule.rename(columns = {'AAPL': 'color'})
    data_rule = pd.concat([stoch_sd,rsi_sd,rsi_rule], axis = 1)
    #print data_rule
    
    fig, ax = plt.subplots()
    colors = {1 : 'green', 0:'black',-1:'red'}
    ax.scatter(data_rule['stoch'], data_rule['rsi'], c=data_rule['color'].apply(lambda x: colors[x]))
    ax.set_xlabel('Stochastic Oscillator (stoch)')
    ax.set_ylabel('Relative Strength Index (rsi)')
    ax.set_ylim([-1.5,1.5])
    ax.set_xlim([-1.5,1.5])
    ax.set(title = 'Rule_based Strategy for Stoch and RSI')
    plt.gca().set_aspect('equal',adjustable = 'box')
    plt.savefig('rule_scatter.png')
    
    
    train_data = build_RL_data()
    train_data = train_data.rename(columns = {'Ydata':'color'})
    fig1, ax1 = plt.subplots()
    colors1 = {1 : 'green', 0:'black',-1:'red'}
    ax1.scatter(train_data['stoch'], train_data['rsi'], c=train_data['color'].apply(lambda x: colors1[x]))
    ax1.set_xlabel('Stochastic Oscillator (stoch)')
    ax1.set_ylabel('Relative Strength Index (rsi)')
    ax1.set_ylim([-1.5,1.5])
    ax1.set_xlim([-1.5,1.5])
    ax1.set(title = 'ML_based Strategy for Stoch and RSI')
    plt.gca().set_aspect('equal',adjustable = 'box')
    plt.savefig('ML_scatter.png')
    
    train_data = build_RL_data()
    test_data = build_RL_data(sd = dt.datetime(2010,1,1),ed = dt.datetime(2011,12,31))
    train_predY, test_predY = training_data(train_data,test_data)
    
    orders = price.copy()
    orders.ix[:,:] = 0
    orders[(train_predY > 0.3)] = 1
    orders[(train_predY < -0.3)] = -1
    q_data = pd.concat([train_data, orders], axis = 1)
    q_data = q_data.rename(columns = {'AAPL':'color'})
    fig2, ax2 = plt.subplots()
    colors2 = {1 : 'green', 0:'black',-1:'red'}
    ax2.scatter(q_data['stoch'], q_data['rsi'], c=q_data['color'].apply(lambda x: colors2[x]))
    ax2.set_xlabel('Stochastic Oscillator (stoch)')
    ax2.set_ylabel('Relative Strength Index (rsi)')
    ax2.set_ylim([-1.5,1.5])
    ax2.set_xlim([-1.5,1.5])
    ax2.set(title = 'ML_based Strategy for Stoch and RSI after Training')
    plt.gca().set_aspect('equal',adjustable = 'box')
    plt.savefig('ML_scatter_after_training.png')
    

def test_code():
    visualization_data()
    

if __name__ =="__main__":
    test_code()
    print 'good job'