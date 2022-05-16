import json
import pytz
import logging
import requests
import pandas as pd
from binance import Client
from datetime import timedelta
from datetime import datetime as dt
from utils.tools import localize_dt_tz, get_local_tz, change_tz

logger = logging.getLogger(__name__)


def binance_bars(symbol='BTCUSDT', start_dt=None, end_dt=None, interval='1h', local_tz="UTC", dtype='dict', limit=1000):
    """
    Parameters
    ----------
    start_dt : Should be UTC datetime object or a string representing a UTC datetime.
    end_dt :   Should be UTC datetime object or a string representing a UTC datetime.
    symbol :     TYPE, optional
                 The default is "BTC".
    your_tz : Your system tz. Binance API does not return klines in UTC. It returns datetimes your local timezone.

    Returns
    -------
    A DataFrame of Binance kline candles with OHLCV prices for a single ticker.  
    """

    if not local_tz:
        local_tz = get_local_tz()

    if end_dt:
        end_dt = localize_dt_tz(end_dt, from_tz=local_tz, to_tz=local_tz)
        end_dt = end_dt.replace(tzinfo=None)
    else:
        end_dt = dt.utcnow().astimezone(pytz.UTC)

    if start_dt:
        start_dt = localize_dt_tz(start_dt, from_tz=local_tz, to_tz=local_tz)
        start_dt = start_dt.replace(tzinfo=None)
    else:
        # start_dt = end_dt - timedelta(days=1)
        # start = int(start_dt.timestamp() * 1000)
        start = None

    if not limit:
        limit = '1000'

    last_datetime = start_dt
    df_list = []

    while True:
        new_df = get_binance_bars(
            symbol, interval, last_datetime, end_dt, limit=limit)
        if new_df is None:
            break
        df_list.append(new_df)
        last_datetime = max(new_df.index) + timedelta(0, 1)

    if not df_list:
        df = f'No data found for {symbol} on Binance.'
    else:
        df = pd.concat(df_list)
        print(df.shape)

        df = change_tz(df, from_tz=local_tz, to_tz="UTC")
        df.index = pd.DatetimeIndex(df.index)
        pd.to_datetime(df.index, yearfirst=True, dayfirst=True)
        df.name = f"{symbol}_{interval}"

        tz_col = 'Dt_'+local_tz.replace('/', '_')
        df = df.drop(columns=[tz_col])
        df['time'] = pd.to_datetime(df.index.to_series())

        # print(df.head())
        if dtype == 'dict':
            df = df.to_dict(orient='records')

        if dtype == 'json':
            df = df.to_json(orient='records')
    return df;


def get_binance_bars(symbol, interval, start=None, end=None, limit='1000'):

    if start:
        start = str(int(start.timestamp() * 1000))
    if end:
        end = str(int(end.timestamp() * 1000))

    url = "https://api.binance.com/api/v3/klines"

    req_params = {"symbol": symbol, 'interval': interval,
                  'startTime': start, 'endTime': end, 'limit': limit}

    d = json.loads(requests.get(url, params=req_params).text)
    if type(d) == dict:
        logger.warning(d)
        d = None
    else:
        d = pd.DataFrame(d)

        if (len(d.index) == 0):
            return None

        d = d.iloc[:, 0:6]
        d.columns = ['time', 'open', 'high', 'low', 'close', 'volume']

        d.open = d.open.astype("float")
        d.high = d.high.astype("float")
        d.low = d.low.astype("float")
        d.close = d.close.astype("float")
        d.volume = d.volume.astype("float")

        d.index = [dt.fromtimestamp(x / 1000.0) for x in d.time]
        d.index.names = ['time']
        d.index = pd.to_datetime(d.index)
        d = d.drop(columns=['time'])

    return d;


def binance_fut_bars(symbol="BTCUSDT", interval='1m', start_dt=None, end_dt=None,
                     local_tz="UTC", dtype='dict', limit=None):

    if not local_tz:
        local_tz = get_local_tz()

    if end_dt:
        end_dt = localize_dt_tz(end_dt, from_tz=local_tz, to_tz=local_tz)
        end_dt = end_dt.replace(tzinfo=None)
    else:
        end_dt = dt.utcnow().astimezone(pytz.UTC)

    if start_dt:
        start_dt = localize_dt_tz(start_dt, from_tz=local_tz, to_tz=local_tz)
        start_dt = start_dt.replace(tzinfo=None)
        start = int(start_dt.timestamp() * 1000)
    else:
        start = None

    end = int(end_dt.timestamp() * 1000)

    if not limit:
        limit = '1000'

    futures_client = Client()

    data = futures_client.get_historical_klines(symbol, interval,
                                                start_str=start, end_str=end, limit=limit)

    df = pd.DataFrame(data)

    if (len(df.index) == 0):
        return f'No data found for {symbol} on Binance (Futures).'

    df = df.iloc[:, 0:6]
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']

    df.open = df.open.astype("float")
    df.high = df.high.astype("float")
    df.low = df.low.astype("float")
    df.close = df.close.astype("float")
    df.volume = df.volume.astype("float")

    df.index = [dt.fromtimestamp(x / 1000.0) for x in df.time]
    df.index.names = ['time']
    df.index = pd.DatetimeIndex(df.index)
    df = change_tz(df, from_tz=local_tz, to_tz="UTC")

    tz_col = 'Dt_'+local_tz.replace('/', '_')
    df = df.drop(columns=[tz_col])
    df['time'] = df.index.to_series().apply(dt.isoformat)

    # print(df.head())
    if dtype == 'json':
        df = df.to_json(orient='records')
    elif dtype == 'dict':
        df = df.to_dict(orient='records')
    return df
