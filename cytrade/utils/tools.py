import os
import sys
import numpy as np
import pandas as pd
from statistics import mode
import math
from datetime import datetime as dt
from datetime import timedelta
import pytz
import glob
from django.utils import timezone


def ends(df, x=5):
    '''return both the head and tail of a dataframe.'''
    return pd.concat([df.head(x), df.tail(x)])


def last_col_first(df):
    """
    Reorders the columns of a pandas DataFrame.
    Brings the last column to the start (left) of the table.
    Leaves the other columns in their original order.
    df  :   a padas DataFrame.
    """
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    # df = df[cols]
    df = df.loc[:, cols]
    return df


def sort_index(df):
    if 'DateTime' in df:
        df.set_index('DateTime')
    else:
        df.index = pd.DatetimeIndex(df.index)
        pd.to_datetime(df.index)
        df.index.names = ['DateTime']

    df.sort_index(ascending=True, inplace=True)

    return df


def format_ohlc(df, caps=False):
    if 'Close' in df:
        pass
    else:
        if caps:
            df.columns = df.columns.str.strip().str.capitalize().str.replace(' ', '_')
        else:
            df.columns = df.columns.str.strip().str.replace(' ', '_')
    return df


def delete_none(_dict):
    """
    Deletes dict keys if their value is None.
    """
    for key, value in list(_dict.items()):
        if isinstance(value, dict):
            delete_none(value)
        elif value is None or key is None:
            del _dict[key]
        elif isinstance(value, list):
            for v_i in value:
                delete_none(v_i)

    return _dict

def change_tz(arg, from_tz, to_tz):
    """
    Change the tz of a date or an array of dates the index of pandas DataFrame.
    Retrun a new Pandas DateTime index and either return the old DateTime index as a series or drop it.
    If a single datetime or timestamp is passed, it will return a datetime or timestamp.
    This function can also infer the timezone from the city, if it is a substring of the timezone. e.g. 'Dubai' will match 'Asia/Dubai.'

    arg   :   list of dates or pandas datetime index or pandas series.
    from_tz :   timezone to change.
    to_tz   :   desired timezone.

    e.g.    from_tz = 'US/Pacific'
            to_tz = 'US/Eastern'

    To see all, enter: pytz.all_timezones
                   or: pytz.common_timezones
    """
    errmsgs = {}
    # pytz.utc
    while True:
        if issubclass(type(from_tz), pytz.BaseTzInfo):
            tz1 = from_tz
        else:
            try:
                tz1 = pytz.timezone(from_tz)

            except pytz.UnknownTimeZoneError:
                print("Searching for TZ...")
                zones = pytz.common_timezones
                for z in list(zones):
                    if (from_tz in z) and (from_tz not in list(zones)):
                        from_tz = z
                        print(f"Timezone: {z}")
                tz1 = pytz.timezone(from_tz)

        if issubclass(type(to_tz), pytz.BaseTzInfo):
            tz2 = to_tz
        else:
            try:
                tz2 = pytz.timezone(to_tz)

            except pytz.UnknownTimeZoneError:
                print("Searching for TZ...")
                zones = pytz.common_timezones
                for z in list(zones):
                    if (to_tz in z) and (to_tz not in list(zones)):
                        to_tz = z
                        print(f"Timezone: {z}")
                tz2 = pytz.timezone(to_tz)

        if type(arg) is pd.DatetimeIndex:
            idx = arg
            try:
                tz1_idx = idx.tz_localize(tz1)
            except TypeError as error:
                already_tz = "Already tz-aware, use tz_convert to convert."
                if already_tz in error.args[0]:
                    print(
                        "Existing datetimes already localized... Proceeding to change TZ.")

                    tz1_idx = idx

            df = pd.DataFrame(index=tz1_idx)

        elif type(arg) is pd.DataFrame:
            df = arg
            try:
                tz1_idx = df.index.tz_localize(tz1)
            except TypeError as error:
                already_tz = "Already tz-aware, use tz_convert to convert."
                if already_tz in error.args[0]:
                    print(
                        "Existing datetimes already localized... Proceeding to change TZ.")

                    tz1_idx = df.index
                    df.index = pd.DatetimeIndex(tz1_idx)

        elif (type(arg) is pd.Timestamp or isinstance(arg, pd._libs.tslibs.timestamps.Timestamp)):
            try:
                arg = arg.tz_localize(tz1)
            except TypeError as error:
                already_tz = "Cannot localize tz-aware Timestamp, use tz_convert for conversions"
                if already_tz in error.args[0]:
                    print("Timestamp already localized... Proceeding to change TZ.")
                    arg.tz_convert(tz2)
            return arg

        elif type(arg[1]) is dt:
            date_lst = list(arg)
            df = pd.DataFrame(index=date_lst)

        elif type(arg[0]) is str:
            try:
                for d in arg:
                    date_lst = dt.strptime(d, '%Y-%m-%d')
                    df = pd.DataFrame(index=date_lst)

            except ValueError as error:
                errmsgs = (
                    "please use the date format '%Y-%m-%d' to avoid ambiguity")
                exception_type, exception_object, exception_traceback = sys.exc_info()
                filename = exception_traceback.tb_frame.f_code.co_filename
                line_number = exception_traceback.tb_lineno
                print("Exception type: ", exception_type)
                print("File name: ", filename)
                print("Line number: ", line_number)
                # print ("Exception object:", exception_object)
                print(error)
                print(errmsgs)

        df.sort_index(ascending=True, inplace=True)

        try:
            tz1_idx = df.index.tz_localize(tz1)
        except TypeError as error:
            already_tz = "Already tz-aware, use tz_convert to convert."
            if already_tz in error.args[0]:
                print("Existing datetimes already localized... Proceeding to change TZ.")
                tz1_idx = df.index

        tz1_name = from_tz.replace('/', '_')
        df[f"Dt_{tz1_name}"] = tz1_idx

        tz2_idx = tz1_idx.tz_convert(tz2)
        df.index = tz2_idx
        df.index.names = ['DateTime']
        if df.columns[0] != f"Dt_{tz1_name}":
            df = last_col_first(df)

        return df


def resample_ohlc(df, interval='1H', o='Open', h='High', l='Low', c='Close', v='Volume',
                  v2=None, msc=None, msc_method='mean'):
    """
    Resample ohlc data.
    interval    :   The desired interval. default ='1H'
    v2          :   An option second volume for some securities expressed in price.
    msc         :   Miscellaneous column. Specify the name.
    msc_method  :   How to resample to msc column. Default = mean.

    Read the below link to get correct market trading hours when resampling:
    https://atekihcan.com/blog/codeortrading/changing-timeframe-of-ohlc-candlestick-data-in-pandas/

    """
    if msc == None:
        msc_method = None
    ohlc = {o: 'first', h: 'max', l: 'min', c: 'last',
            v: 'sum', v2: 'sum', msc: msc_method}
    ohlc_dict = delete_none(ohlc)

    for x in ohlc_dict:
        df = df.resample(interval, offset=0).apply(ohlc_dict)  # origin=0

    return df


def ticks_to_ohlc(df, bid='Bid', ask='Ask', interval='1Min'):
    df_ask = df[ask].resample(interval).ohlc()
    df_bid = df[bid].resample(interval).ohlc()
    df.head()
    df_ab = pd.concat([df_ask, df_bid], axis=1, keys=[ask, bid])
    return df_ab


def ends(df, x=5):
    return pd.concat([df.head(x), df.tail(x)])


def chunk_list(lst, chunks):
    n = math.ceil(len(lst)/chunks)

    for x in range(0, len(lst), n):
        each_chunk = lst[x: n+x]

        if len(each_chunk) < n:
            each_chunk = each_chunk + [None for y in range(n-len(each_chunk))]
        yield each_chunk


def str_to_dt(arg, yearfirst=True, dayfirst=True):
    """
    Convert a string representation of a either a date, datetime object to a string.
    As a last resort, convert to a pandas datetime using dateutil parser.
    """
    if type(arg) is str:
        try:
            arg = dt.strptime(arg, '%Y-%m-%d')
        except ValueError:
            try:
                arg = dt.strptime(arg, '%d-%m-%Y')
                print(
                    f"Assuming {arg} is D/M/Y format. Ensure this is correct. Recommend Y/M/D to avoid ambiguity.")
            except ValueError:
                try:
                    arg = dt.strptime(arg, "%Y-%m-%d %H:%M:%S%z")
                except ValueError:
                    try:
                        arg = dt.strptime(arg, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        try:
                            arg = dt.strptime(
                                arg, "%Y-%m-%d %H:%M:%S.%f%z")
                        except ValueError:
                            try:
                                arg = dt.strptime(
                                    arg, "%Y-%m-%d %H:%M:%S.%f")
                            except ValueError:
                                print(
                                    f"Assuming {arg} is D/M/Y format. Ensure this is correct. Recommend Y/M/D to avoid ambiguity.")
                                arg = pd.to_datetime(
                                    arg, yearfirst=yearfirst, dayfirst=dayfirst)
    return arg


def localize_dt_tz(d_t, from_tz, to_tz):
    """
    Localize the TZ of a single datetime object.
    i.e. convert from one timezone to another in local format.
    """
    errmsgs = {}
    # pytz.utc
    while True:
        if issubclass(type(from_tz), pytz.BaseTzInfo):
            tz1 = from_tz
        else:
            try:
                tz1 = pytz.timezone(from_tz)

            except pytz.UnknownTimeZoneError:
                print("Searching for TZ...")
                zones = pytz.common_timezones
                for z in list(zones):
                    if (from_tz in z) and (from_tz not in list(zones)):
                        from_tz = z
                        print(f"Timezone: {z}")
                tz1 = pytz.timezone(from_tz)

        if issubclass(type(to_tz), pytz.BaseTzInfo):
            tz2 = to_tz
        else:
            try:
                tz2 = pytz.timezone(to_tz)

            except pytz.UnknownTimeZoneError:
                print("Searching for TZ...")
                zones = pytz.common_timezones
                for z in list(zones):
                    if (to_tz in z) and (to_tz not in list(zones)):
                        to_tz = z
                        print(f"Timezone: {z}")
                tz2 = pytz.timezone(to_tz)

        if type(d_t) is str:
            d_t = str_to_dt(d_t, yearfirst=True, dayfirst=True)

        if type(d_t) is dt:
            if d_t.tzinfo == None:
                d_t = tz1.localize(d_t)
                d_t = d_t.astimezone(tz2)
            elif d_t.tzinfo != None:
                d_t = d_t.astimezone(tz2)

        return d_t


def get_local_tz():
    try:
        now = timezone.now()
    except:
        now = dt.now()
    local_now = now.astimezone()
    local_tz = local_now.tzinfo
    local_tzname = local_tz.tzname(local_now)
    print(local_tzname)
    return local_tzname


def dt_to_str(arg):
    """
    Convert a date, datetime to a string. Use isoformat as a last resort.
    """
    if type(arg) is dt:
        try:
            arg = dt.strftime(arg, '%Y-%m-%d')
        except ValueError:
            try:
                arg = dt.strftime(arg, '%d-%m-%Y')
                print(
                    f"Assuming {arg} is D/M/Y format. Ensure this is correct. Recommend Y/M/D to avoid ambiguity.")
            except ValueError:
                try:
                    arg = dt.strftime(arg, "%Y-%m-%d %H:%M:%S%z")
                except ValueError:
                    try:
                        arg = dt.strftime(arg, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        try:
                            arg = dt.strftime(
                                arg, "%Y-%m-%d %H:%M:%S.%f%z")
                        except ValueError:
                            try:
                                arg = dt.strftime(
                                    arg, "%Y-%m-%d %H:%M:%S.%f")
                            except ValueError:
                                print(f"Converting {arg} to isoformat.")
                                arg = arg.isoformat()
    return arg


def check_file(name, dir='assets/images/crypto_icons/svg/color', file_type='svg'):
    '''Chack if a file exists in a directory. If so, return its path.'''
    abs_path = os.path.abspath(dir)
    if not os.path.isdir(abs_path):
        print("No folder called: ", dir)
        return
    else:
        file = glob.glob(f"{abs_path}/{name}.{file_type}")
        return file


def is_utc(date, yearfirst=True, dayfirst=True):
    """
    Assert that datetime object is UTC and format it as a pandas datetime Timestamp.
    If it is not already tz aware, make it timezone aware by assigning it UTC a timezone attribute.
    """
    date = pd.to_datetime(
        date, utc=True, yearfirst=yearfirst, dayfirst=dayfirst)
    return date


def first_day_of_current_year(time=False, utc=False):
    """
    Calculate the first date or datetime of the current year.
    If time=True, it will return the first microsecond of the current year.
    """
    if time:
        fdocy = dt.now().replace(month=1, day=1, hour=0,
                                       minute=0, second=0, microsecond=0)
    else:
        fdocy = dt.now().date().replace(month=1, day=1)
    if utc:
        fdocy = is_utc(fdocy)
    return fdocy


def to_utc(arg, yearfirst=True, dayfirst=True):
    """
    Convert a datetime object, a Timestamp, a pandas DateTimeIndex to UTC.
    Pass a dataframe as an arguement and it will convert the DateTimeIndex.

    Check if either a string, datetime or timestamp is TZ aware or TZ naive.
    Convert to UTC or else assign it to UTC if it is TZ naive.
    Caution: assumes any naive datetime object is already in UTC.
    If the time is not already in UTC and doesn't have any tz metadate, first, assign it a timezone attribute with tz_localize
    yearfirst /dayfirst  : see: https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
    """
    utc_tz = pytz.timezone("UTC")

    if type(arg) is str:
        arg = str_to_dt(arg, yearfirst=True, dayfirst=True)

        if str(arg.tzinfo) == 'UTC':
            print(f"{arg} is already in UTC.")
            return arg
        elif arg.tzinfo == None:
            print(
                f"Assuming naive datetime: {arg} is already in UTC and labelling it as such. Ensure this is correct.")
            arg = utc_tz.localize(arg)
        elif arg.tzinfo != None:
            print(f"Converting {arg.tzinfo} to UTC")
            arg = arg.astimezone(utc_tz)

    elif type(arg) is dt:
        if str(arg.tzinfo) == 'UTC':
            print(f"{arg} already in UTC.")
            return arg
        elif arg.tzinfo == None:
            print(
                f"Assuming naive datetime: {arg} is already in UTC and labelling it as such. Ensure this is correct.")
            arg = utc_tz.localize(arg)
        elif arg.tzinfo != None:
            print(f"Converting {arg.tzinfo} to UTC")
            arg = arg.astimezone(utc_tz)

    elif type(arg) is pd.Timestamp:
        if str(arg.tzinfo) == 'UTC' or str(arg.tzinfo) == 'tzutc()':
            print(f"{arg} already in UTC.")
            return arg
        elif arg.tzinfo == None:
            print(
                f"Assuming naive datetime: {arg} is already in UTC and labelling it as such. Ensure this is correct.")
            arg = pd.to_datetime(
                arg, utc=True, yearfirst=yearfirst, dayfirst=dayfirst)
        elif arg.tzinfo != None:
            print(f"Converting {arg.tzinfo} to UTC")
            arg = arg.tz_convert(utc_tz)

    elif type(arg) is pd.DatetimeIndex:
        if str(arg.tzinfo) == 'UTC':
            print("Index already in UTC.")
            return arg
        elif arg.tzinfo == None:
            print(f"Assuming naive DateTime Index is already in UTC and labelling it as such. Ensure this is correct.")
            arg = arg.tz_localize(utc_tz)
        elif arg.tzinfo != None:
            print(f"Converting {arg.tzinfo} to UTC")
            arg = arg.tz_convert(utc_tz)

    elif type(arg) is pd.DataFrame:
        if str(arg.index.tzinfo) == 'UTC':
            print("Index already in UTC.")
            return arg
        if arg.index.tzinfo == None:
            print(f"Assuming naive DateTime Index is already in UTC and labelling it as such. Ensure this is correct.")
            arg.index = arg.index.tz_localize(utc_tz)
        elif arg.index.tzinfo != None:
            print(f"Converting {str(arg.index.tzinfo)} to UTC")
            arg.index = arg.index.tz_convert(utc_tz)

    return arg


def get_tz(tz):
    """
    Pass a string and return  a pytz timezone object.
    This function can also infer the timezone from the city, if it is a substring of the timezone. e.g. 'Dubai' will return 'Asia/Dubai.'
    """
    try:
        t_z = pytz.timezone(tz)

    except pytz.UnknownTimeZoneError:
        print("Searching for TZ...")
        zones = pytz.common_timezones
        for z in list(zones):
            if (tz in z) and (tz not in list(zones)):
                tz = z
                print(f"Timezone: {z}")
        t_z = pytz.timezone(tz)

    return t_z


def split_datetime(df, date=True, year=False, quarter=False, month=False, day=False, dow=False,
                   time=False, hour=False, minute=False, second=False, microseconds=False):
    if date:
        df['Date'] = df.index.date
    if year:
        df['Year'] = df.index.year
    if quarter:
        df['Quarter'] = df.index.quarter
    if month:
        df['Month'] = df.index.month
    if day:
        df['Day'] = df.index.day
    if dow:
        df['Day_of_Week'] = df.index.dayofweek
    if time:
        df['Time'] = df.index.time
    if hour:
        df['Hour'] = df.index.hour
    if minute:
        df['Minute'] = df.index.minute
    if second:
        df['Second'] = df.index.second
    if microseconds:
        df['Microsecond'] = df.index.microseconds


def get_t_delta(interval='h'):
    if interval == 'h':
        t_delta = 'timedelta64[h]'
    elif interval == 'd':
        t_delta = 'timedelta64[d]'
    elif interval == 'm':
        t_delta = 'timedelta64[m]'
    elif interval == 'ms':
        t_delta = 'timedelta64[ms]'
    elif interval == 'ns':
        t_delta = 'timedelta64[ns]'
    return t_delta


def t_to_exp(df, interval='h'):
    df['Next_Exp'] = df['Exp_DateTime'].bfill()
    t_delta = get_t_delta(interval=interval)
    df['t_to_Exp'] = (df['Next_Exp']-df['DateTime']).astype(t_delta)
    return df


def t_since_exp(df, interval='h'):
    df['Last_Exp'] = df['Exp_DateTime'].ffill()
    t_delta = get_t_delta(interval=interval)
    df['t_since_Exp'] = (df['DateTime']-df['Last_Exp']).astype(t_delta)
    return df


def expir_delta(df, expirations_df, interval='h'):
    """
    Add columns expressing the timedelta since expiration and until the next expiration.

    df          : a pandas DataFrame.
    interval    : the format to express the timedelta. Default is 'h' for hours.
                  d, h, m, ms, ns.

    expirations_df : a datafame of expiration date range. e.g., for bitcoin,last_fris_df
    """
    t_to_exp(df, interval=interval)
    t_since_exp(df, interval=interval)

    if df['t_to_Exp'].isnull().any():
        t_delta = get_t_delta(interval=interval)
        df['t_to_Exp'] = np.where(df['t_to_Exp'].isnull(
        ), (expirations_df.index[-1]-df['DateTime']).astype(t_delta), df['t_to_Exp'])

    if df['t_since_Exp'].isnull().any():
        t_delta = get_t_delta(interval=interval)
        previous_exp = expirations_df.index[0] - \
            pd.offsets.LastWeekOfMonth(n=1, weekday=4)
        df['t_since_Exp'] = np.where(df['t_since_Exp'].isnull(
        ), (df['DateTime']-previous_exp).astype(t_delta), df['t_since_Exp'])



def add_settle_dates(df, hist_exp_df, time=False):
    if time == False:
        if 'Date' not in df:
            df['Date'] = df.index.date
        df['Is_Exp_Date'] = df['Date'].astype(str).isin(hist_exp_df['Date'])
        df.tail()
        df['Is_Exp_Date'].value_counts()
        print("Expiration dates in range:", hist_exp_df['Date'].count())

        df['Exp_Date'] = df[df['Is_Exp_Date'] == True]['Date']
        found_set_dates = pd.DataFrame(df['Exp_Date'], columns=[
                                       'Exp_Date']).drop_duplicates()
        print("Found expiration dates:", found_set_dates['Exp_Date'].count())

        missing_set_date = hist_exp_df.loc[(
            ~hist_exp_df['Date'].isin(found_set_dates['Exp_Date'].astype(str)))]
        print("Missing expiration dates:", len(missing_set_date))

    elif time:
        if 'DateTime' not in df:
            df['DateTime'] = df.index
        df['Is_Exp_DateTime'] = df['DateTime'].astype(
            str).isin(hist_exp_df['DateTime'].astype(str))
        df.tail()
        df['Is_Exp_DateTime'].value_counts()

        valid_exp_dates = pd.DataFrame(
            hist_exp_df.index.date, columns=['Date'])
        valid_exp_dates = valid_exp_dates.drop_duplicates(ignore_index=False)

        print("Expiration dates in range:", (valid_exp_dates['Date'].count()))
        print("Expiration day datetimes in range:",
              (hist_exp_df['DateTime'].count()))

        df['Exp_DateTime'] = df[df['Is_Exp_DateTime'] == True]['DateTime']
        print("Found expiration day datetimes:", df['Exp_DateTime'].count())

        missing_set_date = hist_exp_df.loc[(~hist_exp_df['DateTime'].astype(
            str).isin(df['Exp_DateTime'].astype(str)))]
        print("Missing expiration day datetimes:", len(missing_set_date))

    return df


def last_fridays(df, time=False):
    # df['Day'] = df.index.day
    # df['Month'] = df.index.month
    # df['Year'] = df.index.year
    # #df['Time'] = df.index.time
    # df['Date'] = pd.to_datetime(df[['Year','Month','Day']])
    #df['DateTime'] = pd.to_datetime(df[['Year','Month','Day','Time']])
    if time == False:
        split_datetime(df, date=True, day=False,
                       month=False, year=False, time=False)
        last_fris = pd.DataFrame(df.apply(
            lambda x: x['Date'] + pd.offsets.LastWeekOfMonth(n=1, weekday=4), axis=1), columns=['Date'])
    elif time:
        last_fris = pd.DataFrame(df.apply(
            lambda x: x['DateTime'] + pd.offsets.LastWeekOfMonth(n=1, weekday=4), axis=1), columns=['DateTime'])

    return last_fris


def get_exp_dates(df=None, start=None, end=None, freq='h', time=None, exp_tz='UTC', expires='last_fri', name='CME Exp.'):
    """
    Add a Series of futures expiration dates for the DataFrame of a given security.

    If a DataFrame is passed and no start_date and end_date passed, the settle dates for the entire series is returned.

    If utc=True, the datetimestamp of expiration will be in UTC timezone format.

    start, end  : the start and end dates. If a string, the dates are assumed to be in UTC.

    expires :   A setting to specify the expiration date interval. Currently only supports 'last_fri' but can easily be expanded.

    time    :   The expiration time (either local or UTC. Does not need to match tz to tz of dataset)
                e.g. If the dataset is formatted in UTC and the contract expires at 4pm London time, you can input '16:00:00' and set the tz to 'London.'
                It will account for daylight saving and return Expiration_DateTimes in UTC.
                format: string - '%H:%M:%S'

    freq    :   the interval of the timeseries. Defaultis 'h' for hours. 'm' forminutes, 'd'for days. 'ms' for microseconds etc., as per pandas date_range function.

    exp_tz :   The timezone of the expiration time.

    name    :   The name of the expiration for chart anotation purposes.

    Returns:    a dict of settlemet dates to use as annotations on a chart.
                a pandas DataFrame of the settlemt dates as a series.
    """
    errors = {}

    if exp_tz == 'UTC':
        exp_tz = 'UTC'
    elif exp_tz != 'UTC' and exp_tz != None:
        exp_tz = get_tz(exp_tz)
    elif exp_tz == None:
        exp_tz = None

    if expires == 'last_fri':

        if start != None:
            sd = pd.to_datetime(start, utc=True)
            sd = sd.tz_convert(exp_tz)
        if end != None:
            ed = pd.to_datetime(end, utc=True)
            ed = ed.tz_convert(exp_tz)

        if start == None and isinstance(df, pd.DataFrame):
            start_date = df.index.min()
            start_date = pd.to_datetime(start_date, utc=True)
            start_date = start_date.tz_convert(exp_tz)

        else:
            start_date = sd
            print("Start date:", start_date)

        if end == None and isinstance(df, pd.DataFrame):
            end_date = df.index.max()
            end_date = pd.to_datetime(end_date, utc=True)
            end_date = end_date.tz_convert(exp_tz)

        else:
            end_date = ed
            print("End date:", end_date)

        if (time == None) or (freq == 'D'):
            date_range_df = pd.DataFrame(index=pd.date_range(
                start=start_date, end=end_date, freq='D', tz=exp_tz))
            date_range_df.index = pd.to_datetime(date_range_df.index, utc=True)
            # date_range['Date'] = date_range.index.to_series()
            # date_range_df['DateTime'] = date_range_df.index.to_series()
            date_range_df['Date'] = date_range_df.index.date
            date_range_df['DateTime'] = date_range_df.index
            range_df = date_range_df

        elif freq != 'D' and time != None:
            dt_range_df = pd.DataFrame(index=pd.date_range(
                start=start_date, end=end_date, freq=freq, tz=exp_tz))
            dt_range_df['Time'] = dt_range_df.index.time
            dt_range_df['Exp_DateTime'] = dt_range_df['Time'].astype(
                str) == time
            dt_range_df = dt_range_df.loc[dt_range_df['Exp_DateTime'] == True]
            dt_range_df.index = pd.to_datetime(dt_range_df.index, utc=True)
            dt_range_df['DateTime'] = dt_range_df.index
            dt_range_df = dt_range_df.drop(columns=['Time', 'Exp_DateTime'])
            range_df = dt_range_df

    while True:
        try:
            if expires == 'last_fri':

                if (time == None) or (freq == 'D'):
                    last_fris = last_fridays(range_df)
                    last_fris = last_fris.drop_duplicates(ignore_index=False)
                    last_fris_df = last_fris.set_index(
                        'Date', drop=False, inplace=False)
                    last_fris_df.index = last_fris_df.index.strftime(
                        '%Y-%m-%d')
                    last_fris_df['Date'] = last_fris_df.index.to_series()

                    expirations = pd.Series(
                        last_fris_df['Date'].values, index=last_fris_df['Date']).to_dict()
                    for key in list(expirations.keys()):
                        expirations[key] = name

                    dates_to_remove = last_fris_df.loc[(
                        ~last_fris_df['Date'].astype(str).isin(range_df['Date'].astype(str)))].copy()
                    hist_exp_df = last_fris_df.loc[(~last_fris_df['Date'].astype(
                        str).isin(dates_to_remove['Date'].astype(str)))].copy()
                    removal_lst = dates_to_remove['Date'].unique().tolist()
                    for nul_date in removal_lst:
                        expirations.pop(nul_date, None)
                    print("")
                    print("First expiration date:", next(
                        iter(expirations.items())))
                    print("Last expiration date:", next(
                        reversed(expirations.items())))

                    hist_exp_df['Date'] = hist_exp_df.index.to_series()
                    break

                elif (time != None):
                    last_fris = last_fridays(range_df, time=True)
                    last_fris = last_fris.drop_duplicates(ignore_index=False)
                    last_fris_df = last_fris.set_index(
                        'DateTime', drop=False, inplace=False)
                    last_fris_df['DateTime'] = last_fris_df.index.to_series()

                    str_exp_df = pd.DataFrame(last_fris_df.index.strftime(
                        '%Y-%m-%d %H:%M:%S'), index=last_fris_df.index.strftime('%Y-%m-%d %H:%M:%S'), columns=['DateTime'])

                    expirations = pd.Series(
                        str_exp_df['DateTime'].values, index=str_exp_df['DateTime']).to_dict()
                    for key in list(expirations.keys()):
                        expirations[key] = name
                    print("")
                    print("First Expiration_DateTime:",
                          next(iter(expirations.items())))

                    dates_to_remove = last_fris_df.loc[(~last_fris_df['DateTime'].astype(
                        str).isin(range_df['DateTime'].astype(str)))].copy()
                    hist_exp_df = last_fris_df.loc[(~last_fris_df['DateTime'].astype(
                        str).isin(dates_to_remove['DateTime'].astype(str)))].copy()

                    dates_to_remove['DateTime'] = dates_to_remove['DateTime'].dt.strftime(
                        '%Y-%m-%d %H:%M:%S')
                    removal_lst = dates_to_remove['DateTime'].unique().tolist()
                    for nul_date in removal_lst:
                        expirations.pop(nul_date, None)
                    print("Last expiration date:", next(
                        reversed(expirations.items())))

                    hist_exp_df['DateTime'] = hist_exp_df.index.to_series()
                    break

                elif expires != 'last_fri':
                    errors[0] = ("\nInvalid expiration date type.")
                    errors[1] = (
                        "\nPlease input paramater: expires='last_fri or ask Saran to edit the source code to add more dates.'\n")
                    raise NotImplementedError

            elif expires != 'last_fri':
                errors[0] = ("\nInvalid expiration date type.")
                errors[1] = (
                    "\nPlease input paramater: expires='last_fri or ask Saran to edit the source code to add more dates.'\n")
                raise NotImplementedError

        except Exception as error:
            exception_type, exception_object, exception_traceback = sys.exc_info()
            filename = exception_traceback.tb_frame.f_code.co_filename
            line_number = exception_traceback.tb_lineno
            print("Exception type: ", exception_type)
            print("File name: ", filename)
            print("Line number: ", line_number)
            # print ("Exception object:", exception_object)
            print(error)
            for e in errors.values():
                if e is not None:
                    print(e)
            break

    if (time == None) or (freq == 'D'):
        df = add_settle_dates(df, hist_exp_df)
    elif freq != 'D' and time != None:
        df = add_settle_dates(df, hist_exp_df, time=True)

    return expirations, last_fris_df, df


def candle_close_dt(df, days=0, mins=0, secs=0):
    """
    Get the closing datetime of a candle. Add a timedelta to the index and returns a new dt column.
    """
    df['Candle_Length'] = pd.to_timedelta(
        ((days*60*60*24)+(mins*60)+secs), unit='s')
    df['Close_DateTime'] = df.index + df['Candle_Length']
    df.drop('Candle_Length', axis=1)


def multiJoin(dfs, method, names):
    """ Join a list of dfs.

        Joining dataframes with different indexes will result in
        omitted and / or missing data.

        method = 'outer' : display missing values for mismatching dates.

        method = 'inner' : keep only dates where all dfs have values.
    """

    # Isolate a df to join all other dfs on
    df_left = dfs[0]
    df_left.columns = [names[0]+ '_' + col for col in df_left.columns]
    df_other = dfs[1:]

    # Manage names
    names_other = names[1:]

    # Loop through list of dataframes to join on the first one,
    # and rename columns
    counter = 0
    for df in df_other:
        df.columns = [names_other[counter] + '_' + col for col in df.columns]
        df_left = df_left.join(df, how = method)
        counter = counter + 1

    return df_left


def multiDF(df, sep, level1='Ticker', level2='Series'):
    """ Takes a single column index from a pandas dataframe,
        splits the original titles by a specified separator,
        and replaces the single column index with a 
        multi index. You can also assign names to levels of your new index
    """
    df_temp = df
    sep = '_'

    single = pd.Series(list(df_temp))
    multi= single.str.split(sep, expand = True)

    multiIndex = pd.MultiIndex.from_arrays((multi[0], multi[1]), names = (level1, level2))


    df_new = pd.DataFrame(df_temp.values, index = df_temp.index, columns = multiIndex)

    return(df_new)


def np_shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result
