import signal
import numpy as np
import cytrader as bt
from cytrader import Order
# import backtrader as bt
# from backtrader import Order
from datetime import datetime as dt
from cystratbase import CyStrategy
from cyutils.cyzers import ll_single_kelly


class CloseAll(CyStrategy):
    def __init__(self):
        self.count = 0
        signal.signal(signal.SIGINT, self.sigstop)

    params = (('maintain', 1),
              ('verbose', False),
              ('log_file', f'../results/bt_closeAll_{dt.now().strftime("%Y_%m_%d-%H_%M_%S")}.csv'))

    def stop(self):
        print('Positions closed.')

    def notify_timer(self, timer, when, *args, **kwargs):
        self.log(f'notify_timer: when: {when}')
        if self.p.stop_on_eod and self._state == self._ST_LIVE and self._in_terminal_state():
            self.log('Stopping strategy.')
            self.cerebro.runstop()

    def sigstop(self, a, b):
        print('Stopping Backtrader')
        self.env.runstop()

    def next(self):
        free, locked = self.broker.get_asset_balance(self.broker._store.base)
        possize = (free + locked - self.p.maintain) * -1

        if possize > 0:
            self.buy(size=possize)
        elif possize < 0:
            self.sell(size=possize)

        signal.SIGINT
        return


class TestStrategy1(CyStrategy):
    params = (('verbose', False),
              ('log_file', f'../results/bt_test_{dt.now().strftime("%Y_%m_%d-%H_%M_%S")}.csv'))

    def next(self):
        for data in self.datas:
            print('{} - {} | O: {} H: {} L: {} C: {} V:{}'.format(
                data.datetime.datetime(0),
                data._name,
                data.open[0],
                data.high[0],
                data.low[0],
                data.close[0],
                data.volume[0]))


class TestStrategyCCXT(bt.Strategy):
    params = (('verbose', False),
              ('log_file', f'../results/bt_testCCXT_{dt.now().strftime("%Y_%m_%d-%H_%M_%S")}.csv'))

    def __init__(self):

        self.bought = False
        # To keep track of pending orders and buy price/commission
        self.order = None

    def next(self):
        if self.live_data:
            for data in self.datas:
                print('{} - {} | O: {} H: {} L: {} C: {} V:{}'.format(data.datetime.datetime(),
                                                                      data._name, data.open[0], data.high[0], data.low[0],
                                                                      data.close[0], data.volume[0]))
                if not self.bought:
                    # self.order = self.buy(size=1.0, exectype=Order.Limit, price=data.close[0])
                    self.order = self.sell(size=1.0, exectype=Order.Market)
                    # self.cancel(self.order);
                    self.bought = True

    def notify_data(self, data, status, *args, **kwargs):
        dn = data._name
        time = dt.now()
        msg = 'Data Status: {}, Order Status: {}'.format(
            data._getstatusname(status), status)
        print(time, dn, msg)
        if data._getstatusname(status) == 'LIVE':
            self.live_data = True
        else:
            self.live_data = False


class RSIStrategy(CyStrategy):
    def __init__(self):
        self.rsi = bt.indicators.RSI(period=14)  # RSI indicator

    params = (('n_positions', 1),
              ('min_positions', 0),
              ('verbose', False),
              ('log_file', f'../results/bt_rsi_{dt.now().strftime("%Y_%m_%d-%H_%M_%S")}.csv'))

    # def prenext(self):
    #     self.next()

    def next(self):
        print('Time: {}, Open: {}, High: {}, Low: {}, Close: {}'.format(
            self.data.datetime.datetime(0),
            self.data.open[0],
            self.data.high[0],
            self.data.low[0],
            self.data.close[0]))
        print('RSI: {}'.format(self.rsi[0]))

        # possize = self.broker.getvalue(datas=[self.data])*0.01
        possize = 0.0025

        if not self.position:
            if self.rsi < 45:  # Enter long
                self.buy(size=possize)
        else:
            if self.rsi > 55:
                self.sell(size=possize)  # Close long position


class RSIStrategyMulti(CyStrategy):
    def __init__(self):
        for data in self.datas:
            self.data.rsi = bt.indicators.RSI(data, period=14)

    params = (('n_positions', 1),
              ('min_positions', 0),
              ('verbose', False),
              ('log_file', f'../results/bt_rsi_{dt.now().strftime("%Y_%m_%d-%H_%M_%S")}.csv'))

    def next(self):
        positions = [d._name for d, pos in self.getpositions().items() if pos]
        up, down = {}, {}
        for data in self.datas:
            print('Time: {}, Open: {}, High: {}, Low: {}, Close: {}'.format(
                data.datetime.datetime(0),
                data.open[0],
                data.high[0],
                data.low[0],
                data.close[0]))
            print('{} RSI: {}'.format(data._name, data.rsi[0]))

            if data.rsi[0] < 40:
                up[data._name] = data.rsi[0]
            elif data.rsi[0] > 60:
                down[data._name] = data.rsi[0]

        shorts = sorted(down, key=down.get)[:self.p.n_positions]
        longs = sorted(up, key=up.get, reverse=True)[:self.p.n_positions]
        n_shorts, n_longs = len(shorts), len(longs)

        # only take positions if at least min_n longs and shorts
        if n_shorts < self.p.min_positions or n_longs < self.p.min_positions:
            longs, shorts = [], []
        for ticker in positions:
            if ticker not in longs + shorts:
                self.order_target_percent(data=ticker, target=0)
                self.log(f'{ticker}: CLOSING ORDER CREATED')

        short_target = -1 / max(self.p.n_positions, n_shorts)
        long_target = 1 / max(self.p.n_positions, n_longs)
        for ticker in shorts:
            self.order_target_percent(data=ticker, target=short_target)
            self.log(f'{ticker}: SHORT ORDER CREATED')
        for ticker in longs:
            self.order_target_percent(data=ticker, target=long_target)
            self.log(f'{ticker}: LONG ORDER CREATED')

    def notify_order(self, order):
        print(order)


class Classifier(bt.Strategy):
    params = (('n_positions', 1),
              ('min_positions', 0),
              ('verbose', False),
              ('log_file', f'../results/bt_classifier_{dt.now().strftime("%Y_%m_%d-%H_%M_%S")}.csv'))

    # bt calls prenext instead of next unless
    # all datafeeds have current values.
    # Call next to avoid duplicating logic.
    def prenext(self):
        self.next()

    def next(self):
        verbose = self.p.verbose
        today = self.datas[0].datetime.date()
        positions = [d._name for d, pos in self.getpositions().items() if pos]
        if verbose:
            self.log(f"POSITIONS: {positions}")
        up, down = {}, {}
        missing = not_missing = 0
        for data in self.datas:
            if data.datetime.date() == today:
                if verbose:
                    self.log("today")
                if data.predicted[0] > 0:
                    up[data._name] = data.predicted[0]
                elif data.predicted[0] < 0:
                    down[data._name] = data.predicted[0]

        # sort dictionaries ascending/descending by value -> list of tuples
        shorts = sorted(down, key=down.get)[:self.p.n_positions]
        longs = sorted(up, key=up.get, reverse=True)[:self.p.n_positions]
        n_shorts, n_longs = len(shorts), len(longs)
        if verbose:
            self.log(f"n_shorts: {n_shorts}, n_longs: {n_longs}")
            self.log(f"SHORTS: {shorts}")
            self.log(f"LONGS: {longs}")

        # only take positions if at least min_n longs and shorts
        if n_shorts < self.p.min_positions or n_longs < self.p.min_positions:
            longs, shorts = [], []
        for ticker in positions:
            if ticker not in longs + shorts:
                self.order_target_percent(data=ticker, target=0)
                if verbose:
                    self.log(f'{ticker} | CLOSING ORDER CREATED')

        short_target = -1 / max(self.p.n_positions, n_shorts)
        long_target = 1 / max(self.p.n_positions, n_longs)
        if verbose:
            self.log(
                f'short_target: {short_target}, long_target: {long_target}')
        for ticker in shorts:
            self.order_target_percent(data=ticker, target=short_target)
            if verbose:
                self.log(f'{ticker} | SHORT ORDER CREATED')
        for ticker in longs:
            self.order_target_percent(data=ticker, target=long_target)
            if verbose:
                self.log(f'{ticker} | LONG ORDER CREATED')


class btKellyML(CyStrategy):
    params = (('n_positions', 1),
              ('min_positions', 0),
              ('window', 60),
              ('verbose', False),
              ('log_file', f'../results/bt_KellyML_{dt.now().strftime("%Y_%m_%d-%H_%M_%S")}.csv'))

    # def prenext(self):
    #     self.next()

    def next(self):
        verbose = self.p.verbose
        today = self.datas[0].datetime.date()
        positions = [d._name for d, pos in self.getpositions().items() if pos]
        up, down = {}, {}
        for data in self.datas:
            if data.datetime.date() == today:
                if data.predicted[0] > 0:
                    up[data._name] = data.predicted[0]
                elif data.predicted[0] < 0:
                    down[data._name] = data.predicted[0]

        shorts = sorted(down, key=down.get)[:self.p.n_positions]
        longs = sorted(up, key=up.get, reverse=True)[:self.p.n_positions]
        n_shorts, n_longs = len(shorts), len(longs)

        if n_shorts < self.p.min_positions or n_longs < self.p.min_positions:
            longs, shorts = [], []
        for ticker in positions:
            if ticker not in longs + shorts:
                self.order_target_percent(data=ticker, target=0)
                if verbose:
                    self.log(f'{ticker} | CLOSING ORDER CREATED')
                    
        for data in self.datas:
            kelly = self.data.kelly[0]

        short_target = -1 * kelly
        long_target = kelly
        if verbose:
            self.log(
                f'short_target: {short_target}, long_target: {long_target}')
        for ticker in shorts:
            self.order_target_percent(data=ticker, target=short_target)
            if verbose:
                self.log(f'{ticker} | SHORT ORDER CREATED')
        for ticker in longs:
            self.order_target_percent(data=ticker, target=long_target)
            if verbose:
                self.log(f'{ticker} | LONG ORDER CREATED')


class KellyML(CyStrategy):
    def __init__(self):
        self.pctChange = bt.ind.PercentChange(self.data.close, period=1)
        self.mean = bt.indicators.MovingAverageSimple(self.pctChange, period=self.p.kel_window)
        self.std = bt.indicators.StdDev(self.pctChange, period=self.p.kel_window)

    params = (('n_positions', 1),
              ('min_positions', 0),
              ('kel_window', 60),
              ('verbose', False),
              ('kel_bounds', [0., 2.]),
              ('log_file', f'../results/bt_KellyML_{dt.now().strftime("%Y_%m_%d-%H_%M_%S")}.csv'))

    # def prenext(self):
    #     self.next()

    def next(self):
        verbose = self.p.verbose
        today = self.datas[0].datetime.date()
        positions = [d._name for d, pos in self.getpositions().items() if pos]
        up, down = {}, {}
        for data in self.datas:
            if data.datetime.date() == today:
                if data.predicted[0] > 0:
                    up[data._name] = data.predicted[0]
                elif data.predicted[0] < 0:
                    down[data._name] = data.predicted[0]

        shorts = sorted(down, key=down.get)[:self.p.n_positions]
        longs = sorted(up, key=up.get, reverse=True)[:self.p.n_positions]
        n_shorts, n_longs = len(shorts), len(longs)

        if n_shorts < self.p.min_positions or n_longs < self.p.min_positions:
            longs, shorts = [], []
        for ticker in positions:
            if ticker not in longs + shorts:
                self.order_target_percent(data=ticker, target=0)
                if verbose:
                    self.log(f'{ticker} | CLOSING ORDER CREATED')
                    
        try:
            kelly = ll_single_kelly(self.mean[0], self.std[0], bounds=np.array(self.p.kel_bounds))
        except:
            kelly = 0

        short_target = -1 * kelly
        long_target = kelly
        if verbose:
            self.log(
                f'target: {long_target}')
        for ticker in shorts:
            self.order_target_percent(data=ticker, target=short_target)
            if verbose:
                self.log(f'{ticker} | SHORT ORDER CREATED')
        for ticker in longs:
            self.order_target_percent(data=ticker, target=long_target)
            if verbose:
                self.log(f'{ticker} | LONG ORDER CREATED')



class SMAStrategy(CyStrategy):
    def __init__(self):
        self.pctChange = bt.ind.PercentChange(self.data.close, period=1)
        self.mean = bt.indicators.MovingAverageSimple(self.pctChange, period=self.p.kel_window)
        self.std = bt.indicators.StdDev(self.pctChange, period=self.p.kel_window)
        self.sma = bt.indicators.MovingAverageSimple(self.data.close, period=self.p.ma_window)

    params = (('n_positions', 1),
              ('min_positions', 0),
              ('ma_window', 60),
              ('kel_window', 60),
              ('kel_bounds', [0., 1.]),
              ('verbose', False),
              ('log_file', f'../results/bt_sma_{dt.now().strftime("%Y_%m_%d-%H_%M")}.csv'))

    def next(self):
        verbose = self.p.verbose
        today = self.datas[0].datetime.date()
        positions = [d._name for d, pos in self.getpositions().items() if pos]
        up, down = {}, {}
        for data in self.datas:
            if data.datetime.date() == today:
                if self.sma > data.close[0]:
                    up[data._name] = data.predicted[0]
                elif self.sma < data.close[0]:
                    down[data._name] = data.predicted[0]

        shorts = sorted(down, key=down.get)[:self.p.n_positions]
        longs = sorted(up, key=up.get, reverse=True)[:self.p.n_positions]
        n_shorts, n_longs = len(shorts), len(longs)

        if n_shorts < self.p.min_positions or n_longs < self.p.min_positions:
            longs, shorts = [], []
        for ticker in positions:
            if ticker not in longs + shorts:
                self.order_target_percent(data=ticker, target=0)
                if verbose:
                    self.log(f'{ticker} | CLOSING ORDER CREATED')
                    
        if self.mean[0]:
            kelly = ll_single_kelly(self.mean[0], self.std[0], bounds=np.array(self.p.kel_bounds))
        else:
            kelly = 0

        short_target = -1 * kelly
        long_target = kelly
        if verbose:
            self.log(
                f'target: {long_target}')
        for ticker in shorts:
            self.order_target_percent(data=ticker, target=short_target)
            if verbose:
                self.log(f'{ticker} | SHORT ORDER CREATED')
        for ticker in longs:
            self.order_target_percent(data=ticker, target=long_target)
            if verbose:
                self.log(f'{ticker} | LONG ORDER CREATED')


# class CyStrategy(bt.Strategy):

#     def log(self, txt, dt=None):
#         """ Logger for the strategy"""
#         dt = dt or self.datas[0].datetime.datetime(0)
#         with Path(self.p.log_file).open('a') as f:
#             log_writer = csv.writer(f)
#             log_writer.writerow([dt.isoformat()] + txt.split(','))

#         # Check if an order has been completed
#         # broker could reject order if not enough cash
#     def notify_order(self, order):
#         if order.status in [order.Submitted, order.Accepted]:
#             return

#         if self.p.verbose:
#             msg = ''
#             if order.status in [order.Completed]:
#                 dets = f'{order.executed.price:.2f} | SIZE: {order.executed.size} | VALUE: {order.executed.value} | COMM: {order.executed.comm}'

#                 if order.isbuy():
#                     msg = f'{order.data._name}: BUY executed {dets}'

#                 elif order.issell():
#                     msg = f'{order.data._name}: SELL executed {dets}'

#             elif order.status in [order.Canceled, order.Margin, order.Rejected]:
#                 msg = f'{order.data._name}: Order Canceled/Margin/Rejected'
#             else:
#                 print(order.status)

#             # print(''), print(order), print(msg), print('')
#             self.log(msg)

    # def order_target_value(self, data=None, target=0.0, price=None, **kwargs):
    #     '''
    #     See bt.Strategy.order_target_value()
    #     This function has been modified to order fractional position sizes for 
    #     trading cryptocurrency or forex, as opposed to fixed-integer-sized contracts.
    #     '''

    #     if isinstance(data, str):
    #         data = self.getdatabyname(data)
    #     elif data is None:
    #         data = self.data

    #     possize = self.getposition(data, self.broker).size
    #     if not target and possize:  # closing a position
    #         return self.close(data=data, size=possize, price=price, **kwargs)

    #     else:
    #         value = self.broker.getvalue(datas=[data])
    #         comminfo = self.broker.getcommissioninfo(data)

    #         # Make sure a price is there
    #         price = price if price is not None else data.close[0]

    #         if target > value:
    #             size = comminfo.get_fracsize(price, target - value)
    #             return self.buy(data=data, size=size, price=price, **kwargs)

    #         elif target < value:
    #             size = comminfo.get_fracsize(price, value - target)
    #             return self.sell(data=data, size=size, price=price, **kwargs)

    #     return None  # no execution size == possize