from datetime import datetime, date, timedelta
from typing import OrderedDict
import pytz
import sys
import simplejson as json
import pandas as pd
import numpy as np
from math import sqrt, ceil
from utils.tools import first_day_of_current_year, to_utc, np_shift
from datatile.summary.df import DataFrameSummary
from scipy.stats import kurtosis, skew, jarque_bera, shapiro, linregress, zscore  # anderson,


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        # if isinstance(obj, float):
        #     return "null"
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, date):
            return obj.isoformat()
        try:
            return super(NpEncoder, self).default(obj)
        except:
            print(obj)


class Security:
    def __init__(self, inp):
        if ".csv" in inp:
            self.df = self.eq1(inp)
        else:
            self.df = self.eq2(inp)

        if hasattr(self.df, 'name'):
            self.name = self.df.name
        else:
            raise Exception(
                "Please name the DataFrame first before creating this object. Use df.name = 'Some_Name.'\nUse the ticker or symbol.")

        if hasattr(self, 'subseries') == False:
            self.ticker = self.name
        else:
            self.subseries.ticker = None

        self.df.name = self.name
        #self.df._metadata += ['name']

    def eq1(self, inp):
        '''If the input for forming a security is a csv file.'''
        x = pd.read_csv(inp, low_memory=False, index_col=['time'], parse_dates=[
                        'time'], infer_datetime_format=True,)
        return x

    def eq2(self, inp):
        '''If the input for forming a security is a dataframe.'''
        y = (inp)
        y.index = pd.to_datetime(y.time)
        return y

    def rolling_cum_returns(self, returns, starting_value=0, out=None):
        '''The rolling cumulative returns for the series.'''

        if len(returns) < 1:
            return returns.copy()

        nanmask = np.isnan(returns)
        if np.any(nanmask):
            returns = returns.copy()
            returns[nanmask] = 0

        allocated_output = out is None
        if allocated_output:
            out = np.empty_like(returns)

        np.add(returns, 1, out=out)
        out.cumprod(axis=0, out=out)

        if starting_value == 0:
            np.subtract(out, 1, out=out)
        else:
            np.multiply(out, starting_value, out=out)

        if allocated_output:
            if returns.ndim == 1 and isinstance(returns, pd.Series):
                out = pd.Series(out, index=returns.index)
            elif isinstance(returns, pd.DataFrame):
                out = pd.DataFrame(
                    out, index=returns.index, columns=returns.columns,
                )
        return out

        # Period returns:
    def get_returns(self, returns='Price_Returns', price='close'):
        """
        Calculate percentage returns.
        """
        df = self.df;

        df['Price'] = df[price].copy();

        df[f'{returns}'] = (df[price] / df[price].shift(1) - 1);

        df['Log_Returns'] = np.log(df[price]/df[price].shift(1)).dropna();

        df['Cumulative Returns'] = self.rolling_cum_returns(df[f'{returns}'])

    def get_geomean(self, returns, axis=None):
        returns = np.asarray(returns)
        valids = returns[~np.isnan(returns)]
        cum_ret = np.nanprod((1 + valids), axis=axis)
        periods = np.count_nonzero(valids)
        return np.power(cum_ret, 1 / periods) - 1

    def get_cum_returns(self, price='close'):
        '''The final cumulative return for the series.'''
        df = self.df
        return (df[price].iloc[-1] / df[price].iloc[0])-1

    def annualized_return(self, returns, ann_factor=252, cum_rets=None):
        if len(returns) < 1:
            return np.nan
        if not cum_rets:
            cum_rets = self.get_cum_returns(price='close')
        num_years = len(returns) / ann_factor
        return (1 + cum_rets) ** (1 / num_years) - 1


    def aggregate_returns(self, returns, convert_to):
        """
        Aggregates returns by week, month, or year.

        Parameters
        ----------
        returns : pd.Series
        Daily returns of the strategy, noncumulative.
            - See full explanation in :func:`~empyrical.stats.cum_returns`.
        convert_to : str
            Can be 'weekly', 'monthly', or 'yearly'.

        Returns
        -------
        aggregated_returns : pd.Series
        """

        def cumulate_returns(x):
            return self.rolling_cum_returns(x).iloc[-1]

        if convert_to == '1w':
            grouping = [lambda x: x.year, lambda x: x.isocalendar()[1]]
        elif convert_to == '1M':
            grouping = [lambda x: x.year, lambda x: x.month]
        elif convert_to == '1Q':
            grouping = [lambda x: x.year, lambda x: int(ceil(x.month/3.))]
        elif convert_to == '1y':
            grouping = [lambda x: x.year]
        else:
            raise ValueError(
                'convert_to must be {}, {} or {}'.format('1w', '1M', '1Q', '1Y')
            )
        return returns.groupby(grouping).apply(cumulate_returns)

    def max_drawdown(self, returns):
        returns_array = np.asanyarray(returns)

        cumulative = np.empty(
            (returns.shape[0] + 1,) + returns.shape[1:],
            dtype='float64',
        )
        cumulative[0] = start = 100
        self.rolling_cum_returns(
            returns_array, starting_value=start, out=cumulative[1:])

        max_return = np.fmax.accumulate(cumulative, axis=0)

        return np.nanmin((cumulative - max_return) / max_return, axis=0)

    def vwap(self, h='high', l='low', c='close', v='volume', window=None):
        """
        Volume-Weighted Average Price.
        VWAP = (Cumulative (Price * Volume)) / (Cumulative Volume)

        """
        df = self.df

        if window == None:
            df['AP'] = (df[[h, l, c]].mean(axis=1))
            # Cumulative price * volume:
            df['CPV'] = (df['AP'] * df[v]).cumsum()
            df['Cum_Volume'] = df[v].cumsum()
            df['VWAP'] = df['CPV']/df['Cum_Volume']
            df.drop(columns=['CPV', 'Cum_Volume'])

        else:
            # Average price:
            df['AP'] = (df[[h, l, c]].mean(axis=1))
            # Cumulative price * volume:
            df['CPV'] = (df['AP'] * df[v]).rolling(window, min_periods=1).sum()
            df['Cum_Volume'] = df[v].rolling(window, min_periods=1).sum()
            df['VWAP'] = df['CPV']/df['Cum_Volume']
            df.drop(columns=['CPV', 'Cum_Volume'])

        return

    def vwp(self, price, volume):
        """
        Support function for the vwap_close fucntion below:
        """
        df = self.df

        return ((df[price]*df[volume]).sum()/df[volume].sum()).round(2)

    def vwap_close(self, window=1, price='close', volume='volume'):
        """
        Returns the Volume-Weighted Average Price for close prices or Adj. close prices.
        """
        df = self.df

        vwap = pd.concat([(pd.Series(self.vwp(df.iloc[i:i+window], price, volume),
                                     index=[df.index[i+window]])) for i in range(len(df)-window)])
        vwap = pd.DataFrame(vwap, columns=['close_VWAP'])
        df = df.join(vwap, how='left')

        return df

    def get_ann_factor(self, interval='1d', trading_periods=252, market_hours=24):

        if interval == '1d':
            ann_factor = trading_periods
            t = 'days'
            p = 'Day'
            #vol_window = vol_window

        elif interval == '1y':
            ann_factor = 1
            t = 'years'
            p = "Yr"
            #vol_window = vol_window

        elif interval == '1h':
            ann_factor = trading_periods*market_hours
            t = 'hours'
            p = "Hr"
            #vol_window = vol_window*market_hours

        elif interval == '4h':
            ann_factor = trading_periods*market_hours/4
            t = '4 hours'
            p = "4Hr"
            #vol_window = vol_window*market_hours

        elif interval == '30m':
            ann_factor = trading_periods*market_hours*2
            t = '30min'
            p = '30min'
            #vol_window = vol_window*market_hours*2
        elif interval == '15m':
            ann_factor = trading_periods*market_hours*4
            t = '15min'
            p = '15min'
            #vol_window = vol_window*market_hours*4
        elif interval == '5m':
            ann_factor = trading_periods*market_hours*12
            t = '5min'
            p = '5min'
            #vol_window = vol_window*market_hours*12
        elif interval == '1m':
            ann_factor = trading_periods*market_hours*60
            t = 'minutes'
            p = 'Min'
            #vol_window = vol_window*market_hours*60

        elif interval == '1s':
            ann_factor = trading_periods*market_hours*(60**2)
            t = 'seconds'
            p = 'Sec'
            #vol_window = vol_window*market_hours*(60**2)

        elif interval == '1w':
            ann_factor = 52
            t = 'weeks'
            p = 'Wk'
            #vol_window = vol_window/7

        elif interval == '1Q':
            ann_factor = 4
            t = 'quarters'
            p = 'Q'

        elif interval == '2Q':
            ann_factor = 2
            t = 'half years'
            p = '6M'

        elif interval == '1M':
            ann_factor = 12
            t = 'months'
            p = 'Month'

        return ann_factor, t, p

    def get_vol(self, window=21, returns='Price_Returns', trading_periods=252, interval='1d',
                market_hours=24):
        """
        1 month window = 21
        3 month window = 63
        window: the lookback period, expressed in terms of the time interval.
        trading_periods : the number of trading days in a year.

        """
        df = self.df

        af, t, p = self.get_ann_factor(
            interval=interval, trading_periods=trading_periods, market_hours=market_hours)

        # Standard deviation:
        df['Std_{}_{}'.format(window, p)] = (
            df[returns][1:].rolling(window).std())
        std = df['Std_{}_{}'.format(window, p)]
        df['Ann_Std_{}_{}'.format(window, p)] = (
            df[returns][1:].rolling(window).std())*np.sqrt(af)
        ann_vol = df['Ann_Std_{}_{}'.format(window, p)]

        # Volatility of log returns:
        df['Vol_{}_{}'.format(window, p)] = (
            df['Log_Returns'][1:].rolling(window).std())
        vol = df['Vol_{}_{}'.format(window, p)]
        df['Ann_Vol_{}_{}'.format(window, p)] = (
            df['Log_Returns'][1:].rolling(window).std())*np.sqrt(af)
        an_vol = df['Ann_Vol_{}_{}'.format(window, p)]

        # Variance Swaps (returns are not demeaned):
        df['Ann_VS_Var_{}_{}'.format(window, p)] = np.square(
            df['Log_Returns'][1:]).rolling(window).sum() * af
        vs_var = df['Ann_VS_Var_{}_{}'.format(window, p)]
        df['Ann_VS_Vol_{}_{}'.format(window, p)] = np.sqrt(vs_var)
        vs_vol = df['Ann_VS_Vol_{}_{}'.format(window, p)]

        # Classic by period (returns are demeaned, dof=1)
        df['Realized_Var_{}_{}'.format(window, p)] = (
            df['Log_Returns'][1:].rolling(window).var()) * af
        #df['Realized_Var_{}_{}'.format(window,p)] = (df['Log_Returns'].rolling(window).var())* af
        r_var = df['Realized_Var_{}_{}'.format(window, p)]
        df['Realized_Vol_{}_{}'.format(window, p)] = np.sqrt(r_var)
        r_vol = df['Realized_Vol_{}_{}'.format(window, p)]

        return std, ann_vol, vol, an_vol, vs_vol, r_vol  # ,vs_var,r_var

    def YangZhang_estimator(self, window=6, trading_periods=252, clean=True, interval='daily', market_hours=24):
        """
        window : The lookback window for rolling calculation. 
                If series is daily, this is days. If houly, this should be hours.
                e.g. To calculate 30 Day volatility from an hourly series, window should =720,
                if the market is open 24 hours, 7 days a week, 
                or 480 if it is open 24 hours, 5 days per week.

        trading_periods : The number of periods in a year. e.g. For a daily series, 252 or 365.
                        For hourly, input the number of trading hours in a year. e.g. 6048 (252*24)

        clean : Whether to drop nan values or not 
                (largely irrelevent for Pandas analysis as nans are dropped automatically)
        """
        df = self.df

        af, t, p = self.get_ann_factor(
            interval=interval, trading_periods=trading_periods, market_hours=market_hours)

        log_ho = (df['high'] / df['open']).apply(np.log)
        log_lo = (df['low'] / df['open']).apply(np.log)
        log_co = (df['close'] / df['open']).apply(np.log)

        log_oc = (df['open'] / df['close'].shift(1)).apply(np.log)
        log_oc_sq = log_oc**2

        log_cc = (df['close'] / df['close'].shift(1)).apply(np.log)
        log_cc_sq = log_cc**2

        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

        close_vol = log_cc_sq.rolling(
            window=window, center=False).sum() * (1.0 / (window - 1.0))
        open_vol = log_oc_sq.rolling(
            window=window, center=False).sum() * (1.0 / (window - 1.0))
        window_rs = rs.rolling(
            window=window, center=False).sum() * (1.0 / (window - 1.0))

        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        df['YangZhang_{}_{}'.format(window, p)] = (
            open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt)
        yz = df['YangZhang_{}_{}'.format(window, p)]
        df['YangZhang{}_{}_Ann'.format(window, p)] = (
            open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * sqrt(af)
        yz_an = df['YangZhang{}_{}_Ann'.format(window, p)]

        if clean:
            return yz.dropna(), yz_an.dropna()
        else:
            return yz, yz_an

    def get_arithmetic_sharpe(self, returns, rfr=0, ann_factor=252):
        adj_returns = returns - rfr
        return np.multiply(
            np.divide(
                np.nanmean(adj_returns, axis=0),
                np.nanstd(adj_returns, ddof=1, axis=0),
            ),
            np.sqrt(ann_factor),
        )

    def get_geometric_sharpe(self, returns, rfr=0, ann_factor=252):
        adj_returns = returns - rfr
        return np.multiply(
            np.divide(
                self.get_geomean(adj_returns, axis=0),
                np.nanstd(adj_returns, ddof=1, axis=0),
            ),
            np.sqrt(ann_factor),
        )

    def adj_returns(self, returns, adjustment_factor):
        if isinstance(adjustment_factor, (float, int)) and adjustment_factor == 0:
            return returns
        return returns - adjustment_factor

    def downside_risk(self, returns, rfr=0, ann_factor=252):

        downside_diff = np.clip(
            self.adj_returns(
                np.asanyarray(returns),
                np.asanyarray(rfr),
            ),
            np.NINF,
            0,
        )
        downside_diff = np.square(downside_diff)
        mean = np.nanmean(downside_diff, axis=0)
        return np.multiply(np.sqrt(mean), np.sqrt(ann_factor))

    def get_sortino(self, returns, required_return=0, rfr=0, ann_factor=252):
        
        adj_rets = np.asanyarray(self.adj_returns(returns, required_return))
        avg_ann_ret = np.nanmean(adj_rets, axis=0) * ann_factor
        ds_risk = self.downside_risk(adj_rets, rfr, ann_factor)
        
        return np.divide(avg_ann_ret, ds_risk)

    def get_calmar(self, returns, ann_factor=252, annlzd_ret=None, max_dd=None, cum_rets=None):
        """
        Calmar ratio / drawdown ratio.
        """
        if not annlzd_ret:
            annlzd_ret = self.annualized_return(
                returns, ann_factor=ann_factor, cum_rets=cum_rets)

        if not max_dd:
            max_dd = self.max_drawdown(returns=returns)
        if max_dd < 0:
            temp = annlzd_ret / abs(max_dd)
        else:
            return np.nan

        if np.isinf(temp):
            return np.nan
        return temp

    def get_omega(self, returns, rfr=0.0, req_return=0.0,
                  ann_factor=252):

        if len(returns) < 2:
            return np.nan

        if ann_factor == 1:
            return_thresh = req_return
        elif req_return <= -1:
            return np.nan
        else:
            return_thresh = (1 + req_return) ** \
                (1. / ann_factor) - 1

        ret_less_thresh = returns - rfr - return_thresh

        numer = sum(ret_less_thresh[ret_less_thresh > 0.0])
        denom = -1.0 * sum(ret_less_thresh[ret_less_thresh < 0.0])

        if denom > 0.0:
            return numer / denom
        else:
            return np.nan

    def stability_of_series(self, returns):
        """Determines R-squared of a linear fit to the cumulative
        log returns. Computes an OLS linear fit,
        and returns R-squared.
        """
        if len(returns) < 2:
            return np.nan

        returns = np.asanyarray(returns)
        returns = returns[~np.isnan(returns)]

        cum_log_returns = np.log1p(returns).cumsum()
        rhat = linregress(np.arange(len(cum_log_returns)),
                          cum_log_returns)[2]

        return rhat ** 2

    def get_tail_ratio(self, returns):
        """Returns the ratio between the right (95%) and left tail (5%).
        e.g. a ratio of 0.33 implies losses are 3x > profits.
        """

        if len(returns) < 1:
            return np.nan

        returns = np.asanyarray(returns)
        # Be tolerant of nan's
        returns = returns[~np.isnan(returns)]
        if len(returns) < 1:
            return np.nan

        return np.abs(np.percentile(returns, 95)) / \
            np.abs(np.percentile(returns, 5))

    def get_VaR(self, returns, period=None, sigma=2.0):
        """
        Value at risk (VaR) - sigma - the standard deviation.
        """
        if period:
            returns_agg = self.aggregate_returns(returns, period)
        else:
            returns_agg = returns.copy()

        value_at_risk = returns_agg.mean() - sigma * returns_agg.std()
        return value_at_risk

    def get_CVaR(self, returns, cutoff=0.05):
        """
        Conditional value at risk (CVaR).

        CVaR measures the expected worst single-period returns, defined as falling below
        a cutoff as a percentile of all daily returns.
        """
        returns = returns.dropna()
        cutoff_index = int((len(returns) - 1) * cutoff)
        return np.mean(np.partition(returns, cutoff_index)[:cutoff_index + 1])

    def add_data_summary(self):
        dfs = DataFrameSummary(self.df)
        smrize = dfs['Price_Returns'].to_dict()
        remove_keys = ['kurtosis', 'skewness', 'sum', 'zeros_num', 'zeros_perc',
        'top_correlations', 'counts', 'unique', 'missing', 'missing_perc', 'types', 'uniques']
        summary = {key:val for key, val in smrize.items() if key not in remove_keys}
        return summary

    def stats(self, returns='Price_Returns', price='close', trading_periods=252, market_hours=24, interval='1d', vol_window=30, geo=True):
        """
        returns : A Pandas series of % returns on which to calculate the below statistics.

        trading_periods : The number of trading days per year for the given self.

        market_hours :  In the case of annualizing intraday returns and volatility, this is a pretty crude calculation.
                    Maybe best to keep market_hours at 24, regardless of whether the market is 24 hours each trading day. 
                    It's fine for daily candles or higher.

        interval: The time interval of the series under examination. 
        Paramaters:(str): 'daily', 'hourly', 'minutes', 'seconds', 'weekly', 'monthly' 'quarterly', 'semiannual', 'annual'

        vol_window: The lookback period for calculating volatility estimators.
                    For daily intervals or higher, this is multiplied by 1.
                    e.g. For a weekly interval, if vol_window=3, this reflects 3 weeks.
                    For intraday series, it should be set to the desired number of days. 
                    e.g. For an hourly series, the vol_window will be multiplied by the number of trading hours.

        """
        df = self.df

        start_date = df.index.min()
        setattr(self, 'start_date', start_date)
        end_date = df.index.max()
        setattr(self, 'end_date', end_date)

        factor, t, p = self.get_ann_factor(
            interval, trading_periods, market_hours)

        periods = df[returns].count()
        setattr(self, 'periods', periods)
        years = periods/factor
        setattr(self, 'years', years)

        if price in df:
            cum_ret = self.get_cum_returns(price=price)
            setattr(self, 'cum_ret', cum_ret)
        else:
            raise ValueError(
                'Include a price index in function paramaterss, e.g. price="close"')

#####################################################################################################################

        # Arithmetic average returns:
        avg_ret = df[returns].mean()
        setattr(self, 'avg_ret', avg_ret)

        # Geometric average returns:
        geomean = ((1 + self.cum_ret)**(1/(periods))) - 1
        setattr(self, 'geomean', geomean)

        # Median returns:
        med_ret = df[returns].median()
        setattr(self, 'med_ret', med_ret)

#####################################################################################################################

        # Annualized Return:
        avg_ann = ((1 + self.avg_ret)**factor)-1
        setattr(self, 'avg_ann', avg_ann)

        if geo:
            # Annualaized Geometric:
            # avg_ann_geo = ((self.cum_ret)**(1/(periods/factor))) - 1
            avg_ann_geo = self.annualized_return(df[returns], ann_factor=factor, cum_rets=self.cum_ret)
            setattr(self, 'avg_ann_geo', avg_ann_geo)

#####################################################################################################################

        # Volatility of returns:
        vol = df['Log_Returns'][1:].std()
        setattr(self, 'vol', vol)

        # Average {vol_window} Day Volatility:
        if f'Vol_{vol_window}_{p}' in df:
            self.vol_roll_mean = df[f'Vol_{vol_window}_{p}'].mean()
        else:
            std, ann_vol, vol_roll, vol_roll_ann, vs_vol, r_vol = self.get_vol(
                window=vol_window, returns=returns, trading_periods=factor)
            vol_roll_mean = vol_roll.mean()
            setattr(self, 'vol_roll_mean', vol_roll_mean)

        if ('high' in df) and ('low' in df) and ('open' in df) and ('close' in df):
            # Average period YangZhang Estimator:
            yz_roll, yz_roll_ann = self.YangZhang_estimator(window=vol_window, trading_periods=trading_periods, clean=True,
                                                            interval=interval, market_hours=market_hours)
            yz_roll_mean = yz_roll.mean()
            setattr(self, 'yz_roll_mean', yz_roll_mean)
        else:
            pass

        # Annualized Volatility:
        ann_vol = self.vol*np.sqrt(factor)
        setattr(self, 'ann_vol', ann_vol)

        # Average Annualized {vol_window} Volatility:
        vol_roll_ann = self.vol_roll_mean*np.sqrt(factor)
        setattr(self, 'vol_roll_ann', vol_roll_ann)

        # Annualized {vol_window} YangZhang:
        if ('high' in df) and ('low' in df) and ('open' in df) and ('close' in df):
            yz_roll_ann = self.yz_roll_mean*np.sqrt(factor)
            setattr(self, 'yz_roll_ann', yz_roll_ann)
        else:
            pass

#####################################################################################################################

        # Compute Simple Sharpe (No RFR)
        sharpe_ar = self.get_arithmetic_sharpe(
            df[returns], rfr=0, ann_factor=factor)
        setattr(self, 'sharpe_ar', sharpe_ar)

        if geo:
            # Compute Geometric Sharpe (No RFR)
            sharpe_geo = self.get_geometric_sharpe(
                df[returns], rfr=0, ann_factor=factor)
            setattr(self, 'sharpe_geo', sharpe_geo)

        # Compute Sortino Ratio (No RFR)
        sortino = self.get_sortino(df[returns], rfr=0, ann_factor=factor)
        setattr(self, 'sortino', sortino)

        # Compute Max Drawdown
        max_dd = self.max_drawdown(df[returns])
        setattr(self, 'max_dd', max_dd)

        # Calmar Ratio
        calmar = self.get_calmar(
            df[returns], ann_factor=factor, annlzd_ret=self.avg_ann_geo, max_dd=self.max_dd, cum_rets=self.cum_ret)
        setattr(self, 'calmar', calmar)

        # Omega Ratio
        omega = self.get_omega(
            df[returns], rfr=0.0, req_return=0.0, ann_factor=factor)
        setattr(self, 'omega', omega)

        stability = self.stability_of_series(df[returns])
        setattr(self, 'stability', stability)

        tail_ratio = self.get_tail_ratio(df[returns])
        setattr(self, 'tail_ratio', tail_ratio)

        VaR = self.get_VaR(df[returns], period=None, sigma=2.0)
        setattr(self, 'VaR', VaR)

        CVaR = self.get_CVaR(df[returns], cutoff=0.05)
        setattr(self, 'CVaR', CVaR)

#####################################################################################################################

        summary = self.add_data_summary()
        setattr(self, 'summary', summary)

        if df[returns].count() >= 5000:
            jarq_bera = jarque_bera(df[returns].dropna())
            shapiro_wilk = 'N/A'
            shapiro_wilk = ['N/A', 'N/A']
        elif df[returns].count() < 5000:
            jarq_bera = 'N/A'
            shapiro_wilk = shapiro(df[returns].dropna())

        setattr(self, 'jarq_bera', jarq_bera)
        setattr(self, 'shapiro_wilk', shapiro_wilk)
        # anderson(df[returns].dropna(), dist='norm')

        kurt_fish = kurtosis(df[returns], nan_policy='omit', fisher=True)
        setattr(self, 'kurt_fish', kurt_fish)
        # kurt_pear = kurtosis(df[returns], nan_policy='omit', fisher=False)
        # setattr(self, 'kurt_pear', kurt_pear)
        skw = skew(df[returns], nan_policy='omit', bias=True)
        setattr(self, 'skw', skw)

        stats = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "periods": self.periods,
            "years": self.years,
            "cum_ret": self.cum_ret,
            "avg_ret": self.avg_ret,
            "geo_mean_ret": self.geomean,
            "med_ret": self.med_ret,
            "avg_ann_ret": self.avg_ann,
            "avg_ann_geo_ret": self.avg_ann_geo,
            "period_vol": self.vol,
            "vol_roll_mean": self.vol_roll_mean,
            "yz_roll_mean": self.yz_roll_mean,
            "ann_vol": self.ann_vol,
            "vol_roll_ann": self.vol_roll_ann,
            "yz_roll_ann": self.yz_roll_ann,
            "sharpe_ar": self.sharpe_ar,
            "sharpe_geo": self.sharpe_geo,
            "sortino": self.sortino,
            "max_dd": self.max_dd,
            "calmar": self.calmar,
            "omega": self.omega,
            "stability": self.stability,
            "tail_ratio": self.tail_ratio,
            "VaR": self.VaR,
            "CVaR": self.CVaR,
            "jarq_bera": self.jarq_bera,
            "shapiro_wilk": {
                "statistic": self.shapiro_wilk[0],
                "p_value": self.shapiro_wilk[1]
            },
            "kurtosis_fish": self.kurt_fish,
            # "kurtosis_pear": float(self.kurt_pear.data),
            "skew": float(self.skw.data),
            "summary": self.summary
        }

        # stats = {k:v if not np.isnan(v) else "null" for k,v in stats.items() }

        return json.loads(json.dumps(stats, cls=NpEncoder));


    def get_sub_series(self, start_date=None, end_date=None, utc=True):
        """
        df: dataframe to split

        start_date: Default DateTime is one year from now in UTC.

        end_date: Default DateTime is now in UTC.
        """
        df = self.df

        if start_date == None:
            utc_now = pytz.utc.localize(datetime.utcnow())
            auto_start = utc_now - timedelta(days=365)
            start_date = auto_start

        if end_date == None:
            end_date = pytz.utc.localize(datetime.utcnow())

        # sd = pd.to_datetime(start_date)
        sd = to_utc(start_date)
        # ed = pd.to_datetime(end_date)
        ed = to_utc(end_date)

        df['DateTime'] = pd.DatetimeIndex(df.index)
        df['DateTime'] = pd.to_datetime(df.index, utc=utc)
        subseries_df = df[(df['DateTime'] >= sd) & (df['DateTime'] <= ed)]
        if ((ed - sd) <= timedelta(days=366)) and (sd.year == ed.year):
            subseries_df.name = f"_{ed.year}"
        else:
            subseries_df.name = f"_{sd.year}_{sd.month}_{sd.day}_{ed.year}_{ed.month}_{ed.day}"

        self.subseries = Security(subseries_df)

        setattr(self, f"{subseries_df.name}", self.subseries)

        #  tkr = self.name.replace('/', '_')
        tkr = self.name
        setattr(self.subseries, 'ticker', tkr)

        print(f"Subseries stored as: {subseries_df.name}")

        return self.subseries

    def get_fibs(self, start=None, end=None, period=None, utc=True):
        """
        Calculate Fibonacci retracement levels, expressed as support and resistance.
        kwargs:
        period :    If None, calculate for the entire DataFrame.
                        If 'ytd', calcualte based on year-to-date high and low.
                        If 'whole', calcualte based on the high and low of entire series.
        start,end : If 'period' not set, alternatively, select a timeframe for highs and lows.
                    The 'period' setting will override start and end.
        If no period, start, or end are provided, the default setting is 'whole.'

        utc :   If True, it will convert all datetimes to UTC.
        """
        df = self.df
        φ = 1.618034
        errors = {}
        while True:
            try:
                if not period:
                    if ((start == None) and (end == None)):
                        period = 'whole'

                if period == 'whole':

                    range_df = df
                    start_dt = df.index.min()
                    end_dt = df.index.max()

                elif (period == 'ytd') and (start == None):
                    start_dt = first_day_of_current_year(
                        time=True, utc=utc)
                    end_dt = pd.to_datetime(datetime.now(), utc=utc)

                elif (period == None) and (start == None):
                    start_dt = df.index.min()
                    start_dt = pd.to_datetime(start_dt, utc=utc)

                elif (period == None) and (start != None):
                    start_dt = pd.to_datetime(start, utc=utc)

                elif (period != None) and (start != None):
                    errors[0] = (
                        "\nEnsure that either 'period' or 'start' and 'end' are set, or else neither")
                    errors[1] = (
                        "(if neither, then the high and low of the past year will be used).")
                    raise NotImplementedError

                if (period == None) and (end == None):
                    end_dt = df.index.max()
                    end_dt = pd.to_datetime(end_dt, utc=utc)

                elif (period == None) and (end != None):
                    end_dt = pd.to_datetime(end, utc=utc)

                elif (period != None) and (end != None):
                    errors[0] = (
                        "\nEnsure that either 'period' or 'start' and 'end' are set, or else neither")
                    errors[1] = (
                        "(if neither, then the high and low of the past year will be used).")
                    raise NotImplementedError

                if (period == None) or (period == 'ytd'):
                    try:
                        range_df = df.loc[(df.index >= start_dt) & (
                            df.index <= end_dt)]
                    except:
                        TypeError
                        print(
                            "\nThere may be a timezone mismatch or other date format issue here.")
                        print(
                            f"Coverting the timezone to UTC. Please verify the times are correct. To check: input: {self.name}.df.head() \n")
                        df.index = pd.to_datetime(df.index, utc=utc)
                        range_df = df.loc[(df.index >= start_dt) & (
                            df.index <= end_dt)]

                elif period != None:
                    if not (period == 'whole') or (period == 'ytd'):
                        errors[0] = (
                            "\nEnsure that the 'period' setting is correct - either 'whole' or 'ytd'.")
                        errors[1] = (
                            "Alternatively,input start and end dates instead.")
                        raise NotImplementedError

                self.high = np.round(range_df['high'].max(), 2)
                self.low = np.round(range_df['low'].min(), 2)
                self.f_236 = np.round(
                    self.high-((self.high-self.low)*0.236), 2)
                self.f_382 = np.round(
                    self.high-((self.high-self.low)*(1-(1/φ))), 2)
                self.f_50 = np.round(self.high-((self.high-self.low)*0.5), 2)
                self.f_618 = np.round(
                    self.high-((self.high-self.low)*(1/φ)), 2)

                # print(
                #     f"high= {self.high}  f_236= {self.f_236}  f_382= {self.f_382}  f_50= {self.f_50}  f_618= {self.f_618}  low= {self.low}")

                return {
                    'high': self.high,
                    'low': self.low,
                    'f_236': self.f_236,
                    'f_382': self.f_382,
                    'f_50': self.f_50,
                    'f_618': self.f_618,
                    'low': self.low,
                }

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
        return


def to_sec(df, name=None):
    """
    Name a DataFrame and initialize it as a Security instance.
    name :  Default is a string of the df variable name.
    """
    if name == None:
        df.name = f'{df=}'.split('=')[0]
    else:
        df.name = name
    df = Security(df)
    return df


def df_dict_to_sec(df_dict):
    """
    Convert a dict of DataFrames into Security objects.
    df_dict : a dictionary of DataFrame names as keys and DataFrames as values.

    To unpack all newly created securities within the dict to globals, use: globals().update(dict_of_dfs)
    Or else call the security from within the dict by its key.
    """
    for key in df_dict.keys():
        df_dict[key].name = f"{key}"
        df_dict[key] = Security(df_dict[key])
    return df_dict


def sec_dict_stats(sec_dict, returns='Price_Returns', price='close', trading_periods=252, market_hours=24, interval='daily', vol_window=30, geo=True):
    """
    Apply the stats function to a dictionary of Securities.
    sec_dict :   a dictionary of Security instances.

    To unpack all securities within the dict to globals, use: globals().update(sec_dict)
    Or else call the security from within the dict by its key. i.e. sec_dict[key]
    """
    for key in sec_dict.keys():
        sec_dict[key].stats(returns=returns, price=price, trading_periods=trading_periods,
                            market_hours=market_hours, interval=interval, vol_window=vol_window, geo=geo)
    return sec_dict


def sec_dict_stat(sec_dict, stat):
    """
    Return a stat for each Security in a dict.
    stat (str): 'cum_ret', 'avg_ret', 'geomean', 'med_ret', 'avg_ann', 'avg_ann_geo', 'vol', 'ann_vol', 
                'vol_roll_mean', 'yz_roll_mean', 'yz_roll_ann', 'sharpe_ar', 'sharpe_geo', 
    """
    stat_dict = OrderedDict()
    stat_dict = {key: getattr(sec_dict[key], stat) for key in sec_dict.keys()}
    return stat_dict


def z_score(df: pd.DataFrame, col: str, ddof: int=0) -> pd.Series:
    df_z = zscore(df[col], ddof=ddof, nan_policy='omit')
    return df_z


def mod_z(df: pd.DataFrame, col: str) -> pd.Series:
    def get_mod_z(col: pd.Series) -> pd.Series:
        ''' modified z-scores'''
        med_col = col.median()
        dev = col - med_col
        MAD = (np.abs(dev)).median()
        mod_z = dev / (1.4826 * MAD)
        # mod_z = 0.6745 * dev / MAD
        # mod_z = mod_z[np.abs(mod_z) < thresh]
        # return np.abs(mod_z)
        return mod_z

    df_mod_z = get_mod_z(df[col])
    return df_mod_z


def join_dfs(data: dict, col='close') -> pd.DataFrame:
    '''Join one column from each df in a dict of OHLCV dfs.'''
    df_dict = dict()
    for tkr, ohlcv in data.items():
        df_dict[tkr] = pd.DataFrame(ohlcv)
        df_dict[tkr].index = pd.to_datetime(df_dict[tkr].time);
        df_dict[tkr] = df_dict[tkr].loc[:, [col]]
        df_dict[tkr] = df_dict[tkr].rename(columns={col: tkr})

    dfs = [df for df in df_dict.values()]
    join_df = dfs[0].join([df for df in dfs[1:]])
    join_df['time'] = join_df.index.to_series();

    return join_df;