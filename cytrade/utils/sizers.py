import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
from numpy.linalg import inv
from numpy.random import dirichlet
import pandas as pd

from sympy import symbols, solve, log, diff
from scipy.optimize import minimize_scalar, newton, minimize
from scipy.integrate import quad
from scipy.stats import norm

import pytz
from datetime import datetime as dt
from histData import binance_bars
from utils.tools import to_utc, multiJoin, multiDF
from utils.flow import Kalman
from utils.analysis import join_dfs

np.random.seed(42)


def norm_integral(f, mean, std):
    val, er = quad(lambda s: np.log(1 + f * s) * norm.pdf(s, mean, std), 
                               mean - 3 * std, 
                               mean + 3 * std)
    return -val


def norm_dev_integral(f, mean, std):
    val, er = quad(lambda s: (s / (1 + f * s)) * norm.pdf(s, mean, std), mean-3*std, mean+3*std)
    return val


def get_kelly_share(data, mean_col='SMA_60', std_col='Std_60_Min'):
    solution = minimize_scalar(norm_integral, 
                        args=(data[mean_col], data[std_col]), 
                        bounds=[0, 2], 
                        method='bounded') 
    return solution.x


def single_kelly(m, s, bounds=[0., 2.]):
    sol = minimize_scalar(norm_integral, args=(m, s), bounds=bounds, method='bounded')
    return round(sol.x, 4)