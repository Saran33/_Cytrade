import cython
import csv
import cytrader as bt
from pathlib import Path
from cpython.datetime cimport datetime


class CyStrategy(bt.Strategy):

    def log(self, str txt):
        """ Logger for the strategy"""
        cdef datetime dt = self.datas[0].datetime.datetime(0)
        with Path(self.p.log_file).open('a') as f:
            log_writer = csv.writer(f)
            log_writer.writerow([dt.isoformat()] + txt.split(','))


    def notify_order(self, object order):
        cdef int status = order.status
        cdef unicode msg
        if status in [order.Submitted, order.Accepted]:
            return

        if self.p.verbose:
            msg = u''
            if status in [order.Completed]:
                dets = f'{order.executed.price:.2f} | SIZE: {order.executed.size} | VALUE: {order.executed.value} | COMM: {order.executed.comm}'

                if order.isbuy():
                    msg = f'| {order.data._name}: BUY executed {dets}'

                elif order.issell():
                    msg = f'| {order.data._name}: SELL executed {dets}'

            elif status in [order.Canceled, order.Margin, order.Rejected]:
                msg = f'| {order.data._name}: Order Canceled/Margin/Rejected'
            else:
                self.log(f"| STATUS: {status}")

            # print(''), print(order), print(msg), print('')
            self.log(msg)

    def order_target_value(self, data=None, double target=0.0, price=None, **kwargs):
        '''
        See bt.Strategy.order_target_value()
        This function has been modified to order fractional position sizes for 
        trading cryptocurrency or forex, as opposed to fixed-integer-sized contracts.
        '''
        cdef double possize, value, size, trgt
        cdef object comminfo
        if isinstance(data, str):
            data = self.getdatabyname(data)
        elif data is None:
            data = self.data

        possize = self.getposition(data, self.broker).size
        if not target and possize:  # closing a position
            return self.close(data=data, size=possize, price=price, **kwargs)

        else:
            value = self.broker.getvalue(datas=[data])
            comminfo = self.broker.getcommissioninfo(data)

            # Make sure a price is there
            price = price if price is not None else data.close[0]

            if target > value:
                trgt = target - value
                size = comminfo.get_fracsize(price, trgt)
                return self.buy(data=data, size=size, price=price, **kwargs)

            elif target < value:
                trgt = value - target
                size = comminfo.get_fracsize(price, trgt)
                return self.sell(data=data, size=size, price=price, **kwargs)

        return None  # no execution size == possize

    # bt calls prenext instead of next unless
    # all datafeeds have current values.
    # Call next to avoid duplicating logic.
    # def prenext(self):
    #     self.next()
