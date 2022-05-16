import numpy as np
import pandas as pd
import utils.ta as ta
# from cyutils.analysis import Security
from utils.analysis import Security
from utils.tools import sort_index, split_datetime, get_exp_dates, expir_delta, candle_close_dt


def add_hist_features(ticker: str, sec: pd.DataFrame, interval='1h', trading_periods=365, market_hours=24, vol_window=60,
                      test_size=0.2):
    sec = sec.fillna(method="ffill")

    sec.name = ticker
    sec = Security(sec)

    sort_index(sec.df)
    split_datetime(sec.df, date=True, year=True, quarter=True,
                   month=True, day=False, dow=True, time=True, hour=True)

    sec.get_returns()

    sec.get_vol(window=vol_window, trading_periods=trading_periods,
                interval=interval, market_hours=market_hours)
    sec.YangZhang_estimator(window=vol_window, trading_periods=trading_periods,
                            interval=interval, market_hours=market_hours)

    pair_expirations, pair_last_fris_df, sec.df = get_exp_dates(
        sec.df, start=None, end=None, freq='D', time='15:00:00', exp_tz='London', expires='last_fri', name='CME Exp.')
    pair_expirations, pair_last_fris_df, sec.df = get_exp_dates(
        sec.df, start=None, end=None, freq='h', time='15:00:00', exp_tz='London', expires='last_fri', name='CME Exp.')
    expir_delta(sec.df, interval='h', expirations_df=pair_last_fris_df)
    sec.df.drop(columns='DateTime', inplace=True)

    df = sec.df

    ta.ama(df, price='close', window=9, fast_period=6,
           slow_period=12, fillna=False)
    ta.bol_bands(df, window=20, std=2, price='close',
                 fillna=False, lib=False, inputs=None)
    ta.kalman_filter(df, price='close', transition_matrices=[1], observation_matrices=[1], initial_state_mean=0, initial_state_covariance=1,
                     observation_covariance=1, transition_covariance=.01)

    df['target'] = np.where(df['Price_Returns'] > 0, 1,
                            np.where(df['Price_Returns'] < 0, -1, 0))

    if interval == '1m':
        mins = 0
        secs = 59
    elif interval == '1h':
        mins = 59
        secs = 59
    elif interval == '1d':
        mins = 1439
        secs = 59
    candle_close_dt(df, mins=mins, secs=secs)

    keep_columns = ~(df.columns.isin(['Date', 'time', 'Time', 'Year', 'Price', 'Exp_Date', 'Exp_DateTime', 'Next_Exp', 'Last_Exp',
                                     'Candle_Length', 'Close_DateTime', 'timestamp']))
    df = df.loc[:, keep_columns]

    dset = df.copy()
    
    tminus_mask = ~(dset.columns.isin(['target']))
    cols_to_shift = dset.columns[tminus_mask]
    dset[cols_to_shift] = dset.loc[:, tminus_mask].shift(1)
    dset = dset[1:]

    return dset
