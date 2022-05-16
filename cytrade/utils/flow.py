from pykalman import KalmanFilter
import numpy as np
import pandas as pd
import calendar
import math
from river import linear_model, compose, preprocessing


class Kalman(object):
    """
    Estimates the moving average of a price process via Kalman Filtering.
    """

    def __init__(
        self,
        tkr,
        interval="1h",
        price="Price_Returns",
        observation_covariance=1.0,
        initial_state_mean=0,
        initial_state_covariance=1.0,
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=0.01,
        initial_window=30,
        maxlen=300,
    ):
        self.tkr = tkr
        self.price = price
        self.freq = interval
        self.initial_window = initial_window

        self.kf = KalmanFilter(
            transition_matrices=transition_matrices,
            observation_matrices=observation_matrices,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
        )
        self.state_mean = pd.Series([self.kf.initial_state_mean], name=self.tkr)
        self.state_cov = pd.Series([self.kf.initial_state_covariance], name=self.tkr)

    def update(self, df):
        for time, obs in df[self.price].bfill().iteritems():
            self._update(time, obs)

    def _update(self, time, obs):
        mu, cov = self.kf.filter_update(
            self.state_mean.iloc[-1], self.state_cov.iloc[-1], obs
        )
        self.state_mean[time] = mu.flatten()[0]
        self.state_cov[time] = cov.flatten()[0]


def get_hour(x):
    x["hour"] = x["t"].hour
    return x


def get_hour(x):
    x["hour"] = x["t"].hour
    return x


def get_ordinal_date(x, t_label="t"):
    return {"ordinal_date": x[t_label].toordinal()}


def get_month(x, t_label="t"):
    return {
        calendar.month_name[month]: month == x[t_label].month for month in range(1, 13)
    }


def get_minute_dist(x, t_label="t"):
    return {
        'M' + str(minute): math.exp(-((x[t_label].minute - minute) ** 2))
        for minute in range(0, 60)
    }

def get_hour_dist(x, t_label="t"):
    return {
        'H' + str(hour): math.exp(-((x[t_label].hour - hour) ** 2))
        for hour in range(0, 24)
    }

def get_weekday_dist(x, t_label="t"):
    return {
        calendar.day_name[day]: math.exp(-((x[t_label].weekday() - day) ** 2))
        for day in range(0, 7)
    }

def get_month_dist(x, t_label="t"):
    return {
        calendar.month_name[month]: math.exp(-((x[t_label].month - month) ** 2))
        for month in range(1, 13)
    }

def get_radial_basis_min(x, t_label="t"):
    '''Apply radial basis functions to discretize a datetime and extract features.
        Apply to '1m', '15m', '30m' candles. '''
    return get_minute_dist(x, t_label) | get_hour_dist(x, t_label) | get_weekday_dist(x, t_label) | get_month_dist(x, t_label)

def get_radial_basis_hr(x, t_label="t"):
    '''Apply radial basis functions to discretize a datetime and extract features.
        Apply to '1h', '4h' candles. '''
    return get_hour_dist(x, t_label) | get_weekday_dist(x, t_label) | get_month_dist(x, t_label)

def get_radial_basis_day(x, t_label="t"):
    '''Apply radial basis functions to discretize a datetime and extract features.
        Apply to '1d', '1w' candles. '''
    return get_weekday_dist(x, t_label) | get_month_dist(x, t_label)


def get_hour_sincos(x, t_label="t"):
    return {
        "hour_sin": math.sin(2 * math.pi * x[t_label].hour / 24.),
        "hour_cos": math.cos(2 * math.pi * x[t_label].hour / 24.),
    }

def get_day_sincos(x, t_label="t"):
    yr = 365.2425
    return {
        "day_sin": math.sin(2 * math.pi * x[t_label].month / yr),
        "day_cos": math.cos(2 * math.pi * x[t_label].timetuple().tm_yday / yr),
    }

def get_month_sincos(x, t_label="t"):
    return {
        "month_sin": math.sin(2 * math.pi * x[t_label].month / 12.),
        "month_cos": math.cos(2 * math.pi * x[t_label].month / 12.),
    }


def discretize_dt_min(x, t_label="t", interval='1m'):
    '''Extract discrete datetime features by applying cosine and sine functions.
            Apply to '1m', '15m', '30m' candles. '''
    return {
        "minute_sin": math.sin(2 * math.pi * x[t_label].minute / 1440.),
        "minute_cos": math.cos(2 * math.pi * x[t_label].minute / 1440.),
        "hour_sin": math.sin(2 * math.pi * x[t_label].minute / 24.),
        "hour_cos": math.cos(2 * math.pi * x[t_label].hour / 24.),
        "day_sin": math.sin(2 * math.pi * x[t_label].month / 365.2425),
        "day_cos": math.cos(2 * math.pi * x[t_label].timetuple().tm_yday / 365.2425),
        "month_sin": math.sin(2 * math.pi * x[t_label].month / 12.),
        "month_cos": math.cos(2 * math.pi * x[t_label].month / 12.),
    }
    
def discretize_dt_hr(x, t_label="t", interval='1m'):
    '''Extract discrete datetime features by applying cosine and sine functions.
                Apply to '1h', '4h' candles. '''
    return {
        "hour_cos": math.cos(2 * math.pi * x[t_label].hour / 24.),
        "day_sin": math.sin(2 * math.pi * x[t_label].month / 365.2425),
        "day_cos": math.cos(2 * math.pi * x[t_label].timetuple().tm_yday / 365.2425),
        "month_sin": math.sin(2 * math.pi * x[t_label].month / 12.),
        "month_cos": math.cos(2 * math.pi * x[t_label].month / 12.),
    }

def discretize_dt_day(x, t_label="t", interval='1m'):
    '''Extract discrete datetime features by applying cosine and sine functions.
            Apply to '1d', '1w' candles. '''
    return {
        "day_sin": math.sin(2 * math.pi * x[t_label].month / 365.2425),
        "day_cos": math.cos(2 * math.pi * x[t_label].timetuple().tm_yday / 365.2425),
        "month_sin": math.sin(2 * math.pi * x[t_label].month / 12.),
        "month_cos": math.cos(2 * math.pi * x[t_label].month / 12.),
    }


class logisticReg(object):
    def __init__(self,
                select=("o", "h", "l", "c", "v", "ret", "kal", "ama", "bbh", "bbm", "bbl"),
                extract=(get_ordinal_date, get_radial_basis_day),
                scale=preprocessing.StandardScaler,
                learn=linear_model.LogisticRegression):

        self.select = compose.Select(*select)
        self.extract_features = compose.TransformerUnion(*extract)
        self.scale = scale()
        self.learn =  learn()

        self.model = self.select 
        self.model += self.extract_features 
        self.model |= self.scale | self.learn

