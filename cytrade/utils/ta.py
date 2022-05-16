# from datetime import datetime, date, timedelta
# from typing import OrderedDict
import pytz
# from os import error
import sys
import pandas as pd
import numpy as np
import ta
# import talib
# from talib import abstract
from pykalman import KalmanFilter


def kalman_filter(df, price='close', transition_matrices=[1], observation_matrices=[1],
                  initial_state_mean=0, initial_state_covariance=1,
                  observation_covariance=1, transition_covariance=.01):

    kf = KalmanFilter(transition_matrices=transition_matrices,
                      observation_matrices=observation_matrices,
                      initial_state_mean=initial_state_mean,
                      initial_state_covariance=initial_state_covariance,
                      observation_covariance=observation_covariance,
                      transition_covariance=transition_covariance)

    state_means, _ = kf.filter(df[price])
    df['Kalman_Filter'] = state_means


def abstract_ta_lib(df, open="Open", high="High", low="Low", close="Close", volume="Volume"):
    # note that all ndarrays must be the same length.
    inputs = {
        'open': df[open],
        'high': df[high],
        'low': df[low],
        'close': df[close],
        'volume': df[volume]}
    return inputs


def all_ta(df, open="Open", high="High", low="Low", close="Close", volume="Volume"):
    """
    Add all ta standard features from bukosabino ta.
    """
    df = ta.all_ta_features(df, open=open, high=high,
                            low=low, close=close, volume=volume)
    return df


def ama(df, window=9, fast_period=6, slow_period=12, price='Close', fillna=False):
    """
    Kaufman's Adaptive Moving Average (KAMA)
    """
    # === talib ====
    # _df['AMA']=pd.Series(talib.MA(df[column].values,periods),index=df.index)
    # _df['AMA']=pd.Series(talib.KAMA(df[column].values,periods),index=df.index)
    # === /talib ====

    # === ta ====
    df[f"AMA_{fast_period}_{slow_period}"] = ta.momentum.KAMAIndicator(
        df[price], window=window, pow1=6, pow2=12, fillna=fillna).kama()


def bol_bands(df, window=20, std=2, price='Close', fillna=True, lib=False, inputs=None):
    """
    Bolliger Bands
    """
    # === ta ====
    if not lib:
        indicator_bb = ta.volatility.BollingerBands(
            close=df[price], window=window, window_dev=std, fillna=fillna)

        df[f"bb_bbh_{std}_{window}"] = indicator_bb.bollinger_hband()
        df[f"bb_bbm_{std}_{window}"] = indicator_bb.bollinger_mavg()
        df[f"bb_bbl_{std}_{window}"] = indicator_bb.bollinger_lband()

        # Add Bollinger Band high indicator
        df[f"bb_bbhi_{std}_{window}"] = indicator_bb.bollinger_hband_indicator()

        # Add Bollinger Band low indicator
        df[f"bb_bbli_{std}_{window}"] = indicator_bb.bollinger_lband_indicator()

        # Add Width Size Bollinger Bands
        df[f"bb_bbw_{std}_{window}"] = indicator_bb.bollinger_wband()

        # Add Percentage Bollinger Bands
        df[f"bb_bbp_{std}_{window}"] = indicator_bb.bollinger_pband()

    # === talib ====
    # uses close prices (default)
    # elif lib:
    #     if inputs != None:
    #         df[f"bbu_{std}_{window}"], df[f"bbm_{std}_{window}"], df[f"bbl_{std}_{window}"] = abstract.BBANDS(
    #             inputs, timeperiod=window, nbdevup=std, nbdevdn=std)
    #     else:
    #         df[f"bbu_{std}_{window}"], df[f"bbm_{std}_{window}"], df[f"bbl_{std}_{window}"] = talib.BBANDS(
    #             df['Close'].values, timeperiod=10, nbdevup=2, nbdevdn=2)
    # # === /talib ====


def adx(df, high='High', low='Low', close='Close', window=14, fillna=True, lib=False, inputs=None):
    """
    Average Directional Movement Index (ADX) - strength of trend.

    The Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI) are derived from smoothed averages of these differences, and measure trend direction over time. These two indicators are often referred to collectively as the Directional Movement Indicator (DMI).

    The Average Directional Index (ADX) is in turn derived from the smoothed averages of the difference between +DI and -DI, and measures the strength of the trend (regardless of direction) over time.
    """
    if not lib:
        # === ta ====
        indicator_adx = ta.trend.ADXIndicator(
            high=df[high], low=df[low], close=df[close], window=window, fillna=fillna)
        df[f"ADX_{window}"] = indicator_adx.adx()
        df[f"ADX_pos_{window}"] = indicator_adx.adx_pos()
        df[f"ADX_neg_{window}"] = indicator_adx.adx_neg()

    # elif lib:
    #     # === talib ====
    #     if inputs != None:
    #         df[f"ADX_{window}"] = abstract.ADX(inputs, timeperiod=window)
    #     else:
    #         df[f"ADX_{window}"] = talib.ADX(
    #             df[high], df[low], df[close], timeperiod=window)
