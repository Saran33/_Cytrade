import datetime as dt
import backtrader as bt
from backtrader_binance import BinanceStore

from setEnv import set_binance_env
from strategies import RSIStrategy

LIVE, FUTS, TEST, BINANCE_KEY, BINANCE_SECRET = set_binance_env(FUTS=False)


def execute(base='BTC', quote='USDT', strategy=RSIStrategy, start_dt=None, interval=1, args=None):
    cerebro = bt.Cerebro(quicknotify=True)

    store = BinanceStore(
        api_key=BINANCE_KEY,
        api_secret=BINANCE_SECRET,
        base=base,
        quote=quote,
        testnet=TEST)
    broker = store.getbroker()
    cerebro.setbroker(broker)

    data = store.getdata(
        timeframe_in_minutes=interval,
        start_date=start_dt)

    cerebro.addstrategy(strategy, **args) if args else cerebro.addstrategy(strategy)
    cerebro.adddata(data, name=base+quote)
    cerebro.run()


if __name__ == '__main__':
    start_dt = dt.datetime.utcnow() - dt.timedelta(minutes=5*16)
    execute(base='BTC', quote='USDT', strategy=RSIStrategy, start_dt=start_dt, interval=1, args=dict(n_positions=1, min_positions=0, verbose=True))
