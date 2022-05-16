import pytz
import pandas as pd
import pyfolio as pf
import seaborn as sns
from time import time
import matplotlib.pyplot as plt
from datetime import datetime as dt
import pandas_datareader.data as web
import backtrader as bt
import backtrader.analyzers as btanalyzers
from histData import binance_bars
from utils.tools import to_utc
from features import add_hist_features
from mlPipelines import ttsplit, feat_pipe1, lda_pipe
from sklearn.metrics import accuracy_score, classification_report

# from strategies import Classifier

def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return f'{h:>02.0f}:{m:>02.0f}:{s:>02.0f}'


class BinanceCommision(bt.CommInfoBase):
    """
    Simple fixed commission scheme for demo
    """
    params = (
        ('commission', 0.001),
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * self.p.commission


# class to define the columns we will provide
class SignalData(bt.feeds.PandasData):
    """
    Define pandas DataFrame structure
    """
    OHLCV = ['open', 'high', 'low', 'close']
    cols = OHLCV + ['predicted']

    # create lines
    lines = tuple(cols)
    # define parameters
    params = {c: -1 for c in cols}
    params.update({'datetime': None})
    params = tuple(params.items())


# class to define the columns we will provide
class KellySignalData(bt.feeds.PandasData):
    """
    Define pandas DataFrame structure
    """
    OHLCV = ['open', 'high', 'low', 'close']
    cols = OHLCV + ['Current_Open'] + ['predicted'] + ['kelly']

    # create lines
    lines = tuple(cols)
    # define parameters
    params = {c: -1 for c in cols}
    params.update({'datetime': None})
    params = tuple(params.items())


def run_backtest(strategy: bt.Strategy, tickers=['BTCUSDT'], interval='1h', start_dt='2017-01-01', end_dt=None, vol_window=60, test_size=0.2,
                start_nav=100000):

    start_dt = to_utc(start_dt)
    print ("START DATE:", start_dt)

    end_dt = pytz.utc.localize(dt.utcnow()) if not end_dt else to_utc(end_dt)
    print("END DATE:", end_dt)

    print("TICKERS:", tickers)

    sets = {}
    for ticker in tickers:
        sec = binance_bars(symbol=ticker, interval=interval,
                            start_dt=start_dt, end_dt=end_dt, limit=None, dtype='df');

        dset = add_hist_features(ticker, sec, interval=interval, vol_window=vol_window, test_size=test_size)

        X_train, X_test, y_train, y_test = ttsplit(dset, test_size)

        feat_pipe = feat_pipe1(dset)
        X_train = feat_pipe.fit_transform(X_train)
        X_test = feat_pipe.transform(X_test)

        model = lda_pipe().fit(X_train, y_train)
        predictions = model.predict(X_test)

        print (ticker, "Accuracy:","{:.2%}".format(accuracy_score(y_test, predictions)))
        print (classification_report(y_test, predictions, zero_division=0))
        print(pd.DataFrame({'Test Set':y_test, 'Predictions': predictions}))

        train_len = int(len(dset) * (1-test_size))
        train_set, test_set = dset.iloc[0:train_len].copy(),dset.iloc[train_len:len(dset)].copy()

        test_set['predicted'] = predictions

        sets[ticker] = test_set


    cerebro = bt.Cerebro()
    comminfo = BinanceCommision()
    cerebro.broker.addcommissioninfo(comminfo)
    cerebro.broker.setcash(start_nav)
    print ("Opening NAV: ${:,.2f}".format(start_nav))

    for ticker in sets:
        bt_df = sets[ticker]
        bt_df.index.name = 'datetime'
        bt_data = SignalData(dataname=bt_df)
        cerebro.adddata(bt_data, name=ticker)

    cerebro.addstrategy(strategy, n_positions=1, min_positions=0, 
                        verbose=True)

    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    cerebro.addanalyzer(btanalyzers.Transactions, _name = "trans")

    start = time()
    results = cerebro.run()
    ending_value = cerebro.broker.getvalue()
    duration = time() - start
    n_trades = len(results[0].analyzers.trans.get_analysis()) 

    print ("Test Start:", bt_df.index.min())
    print ("Test End:", bt_df.index.max())
    test_dt_len = ((bt_df.index.max() - bt_df.index.min()).total_seconds())/(24*60*60)
    print (f"Days: {test_dt_len:,.2f}")

    print ("Opening NAV: ${:,.2f}".format(start_nav))
    print(f'Closing NAV ${ending_value:,.2f}')
    gross_ret = (ending_value/start_nav)-1
    print ("Gross Return: {:,.2%}".format(gross_ret))
    print(f'Duration: {format_time(duration)}')
    print(f'Trades: {n_trades:,}')
    print('System Quality Number: ', results[0].analyzers.sqn.get_analysis())

    # prepare pyfolio inputs
    pyfolio_analyzer = results[0].analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()

    returns.to_hdf('backtrader.h5', 'returns')
    positions.to_hdf('backtrader.h5', 'positions')
    transactions.to_hdf('backtrader.h5', 'transactions/')
    gross_lev.to_hdf('backtrader.h5', 'gross_lev')

    bench_name = 'CBBTCUSD' # 'SP500'
    benchmark = web.DataReader(bench_name, 'fred', returns.index.min(), returns.index.max()).squeeze()
    benchmark = benchmark.pct_change().tz_localize('UTC')

    daily_tx = transactions.groupby(level=0)
    longs = daily_tx.value.apply(lambda x: x.where(x>0).sum())
    shorts = daily_tx.value.apply(lambda x: x.where(x<0).sum())

    fig, axes = plt.subplots(ncols=2, figsize=(15, 5))

    df = returns.to_frame('Strategy').join(benchmark.to_frame(f'Benchmark ({bench_name})'))
    df.add(1).cumprod().sub(1).plot(ax=axes[0], title='Cumulative Return')

    longs.plot(label='Long',ax=axes[1], title='Positions')
    shorts.plot(ax=axes[1], label='Short')
    positions.cash.plot(ax=axes[1], label='PF Value')
    axes[1].legend()
    sns.set()
    sns.despine()
    fig.tight_layout();

    pf.create_full_tear_sheet(returns,
                            transactions=transactions,
                            positions=positions,
                            benchmark_rets=benchmark.fillna(0))