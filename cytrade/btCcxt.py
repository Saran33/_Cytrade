import time
from datetime import datetime as dt, timedelta

# import backtrader as bt
# from ccxtbt import CCXTStore
import cytrader as bt
from ccxtct import CCXTStore

from setEnv import set_binance_env
from strategies import TestStrategyCCXT, TestStrategy1

LIVE, FUTS, TEST, BINANCE_KEY, BINANCE_SECRET = set_binance_env(FUTS=False)


def get_timeframe(interval: str):
    frames = {
        "1m": {"timeframe": bt.TimeFrame.Minutes, "interv": 1},
        "15m": {"timeframe": bt.TimeFrame.Minutes, "interv": 15},
        "30m": {"timeframe": bt.TimeFrame.Minutes, "interv": 30},
        "1h": {"timeframe": bt.TimeFrame.Minutes, "interv": 60},
        "4h": {"timeframe": bt.TimeFrame.Minutes, "interv": 240},
        "1d": {"timeframe": bt.TimeFrame.Days, "interv": 1},
    }
    f = frames.get(interval)
    timeframe = f.get("timeframe")
    interv = f.get("interv")
    return timeframe, interv


def execute(
    bases=["BTC", "ETH"],
    quote="USDT",
    strategy=TestStrategyCCXT,
    start_dt=None,
    end_dt=None,
    interval="1m",
    debug=True,
    verbose=False,
    stratKwargs=None,
    futures=FUTS,
):

    cerebro = bt.Cerebro(quicknotify=True)

    cerebro.addstrategy(strategy, **stratKwargs)


    config = {
        "apiKey": BINANCE_KEY,
        "secret": BINANCE_SECRET,
        "enableRateLimit": True,
        "nonce": lambda: str(int(time.time() * 1000)),
    }

    if TEST:
        print("DEMO ACCOUNT")
        options = {"fetchCurrencies": False,}
    else:
        options = {}
    if FUTS:
        options = options | {"defaultType": "future"}
        exchange = 'binanceusdm'
    else:
        exchange = 'binance'

    config = config | {"options": options}


    store = CCXTStore(
        exchange=exchange,
        quote=quote,
        config=config,
        retries=5,
        debug=debug,
        verbose=verbose,
        sandbox=TEST,
        futures=FUTS,
    )

    broker_mapping = {
        "order_types": {
            bt.Order.Market: "market",
            bt.Order.Limit: "limit",
            bt.Order.Stop: "stop-loss",
            bt.Order.StopLimit: "stop limit",
        },
        "mappings": {
            "closed_order": {"key": "status", "value": "closed"},
            "canceled_order": {"key": "status", "value": "canceled"},
        },
    }

    broker = store.getbroker(
        broker_mapping=broker_mapping, debug=debug, verbose=verbose
    )
    cerebro.setbroker(broker)

    cash, value = broker.get_wallet_balance(quote)
    print("CASH: ", cash, quote)
    # cerebro.broker.setcash(cash)

    timeframe, interv = get_timeframe(interval)

    for base in bases:
        data = store.getdata(
            dataname=base + "/" + quote,
            name=base + quote,
            timeframe=timeframe,
            fromdate=start_dt,
            todate=end_dt,
            compression=interv,
            ohlcv_limit=50,
            drop_newest=True,
        )  # ,debug=debug, historical=True)
        data.base = base
        cerebro.adddata(data)
        broker.updatePortfolio(data)

    cerebro.run()


if __name__ == "__main__":
    start_dt = dt.utcnow() - timedelta(minutes=1440)
    end_dt = dt.utcnow()
    execute(
        bases=["BTC", "ETH"],
        quote="USDT",
        strategy=TestStrategy1,
        start_dt=start_dt,
        end_dt=end_dt,
        interval="1m",
        debug=True,
        verbose=False,
        # stratKwargs = dict(
        # model=logisticReg,
        # metric=metrics.ROCAUC,
        # metric_window=100,
        # log=True)
    )
