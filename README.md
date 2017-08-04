# LS-optimize

import numpy as np
import matplotlib as p
import pdb
import scipy.optimize
import pandas as pd
from pandas import *


def loadData(csv_file_name):
    
    dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')
    data = pd.read_csv(csv_file_name, parse_dates=['Date'], index_col='Date',date_parser=dateparse)
    data = data.sort_index(axis=0, level=None, ascending=True)

    return data


# Calculates portfolio mean return
def port_mean(W, R):
    return sum(R * W)


# Calculates portfolio variance of returns
def port_var(W, C):
    return np.dot(np.dot(W, C), W)


# Combination of the two functions above - mean and variance of returns calculation
def port_mean_var(W, R, C):
    return port_mean(W, R), port_var(W, C)


def port_mean_std(W, R, C):
    return port_mean(W, R), (port_var(W, C)**(0.5))


# weights of tangency portfolio with respect to sharpe ratio maximization
def solve_weights(W, R, C, v):
    
    def fitness(W, R, C, v):
        # calculate mean/variance of the portfolio
        mean, var = port_mean_var(W, R, C)
        penalty = 100 * abs(var**(0.5) - v)         # also target specific annual vol
        return 1/mean + penalty                     # maximize mean by minimizing 1/mean
    
    n = len(R)
    # W = np.ones([n])/n                     # start with equal weights
    b_ = [(0.,1.) for i in range(n)]    # weights between 0%..100%. 
                                        # No leverage, no shorting
    #c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })   # Sum of weights = 100%
    optimized = scipy.optimize.minimize(fitness, W, (R, C, v), 
                method='SLSQP', bounds=b_)  
    #            method='SLSQP', constraints=c_, bounds=b_)  
    if not optimized.success: 
        raise BaseException(optimized.message)
    return optimized.x  # Return optimized weights



if __name__ == '__main__':

    data = loadData('optimization_data.csv')
    
    data.index
    data.dtypes

    vol_target = 0.1

    # calculate weights based on equal vol
    data_vol = data.std()*252**(0.5)
    equal_wgts = vol_target/data_vol/len(data_vol)

    # calculate mean and covariance matrix (annualized) for return data
    R = data.mean()*252
    C = data.cov()*252
    
    v = 0.05    # target 5% vol
                
    optimized_wgts = solve_weights(equal_wgts, R, C, v)
    pandas.DataFrame(optimized_wgts, index=list(data))
    
    
    port_mean_std(equal_wgts, R, C)
    port_mean_std(optimized_wgts, R, C)
