import cytrader as bt
from datetime import datetime as dt
from cystratbase import CyStrategy
from cyutils.cyzers import ll_single_kelly
from utils.flow import Kalman, logisticReg
from river import metrics


class StreamLowMemory(CyStrategy):
    def __init__(self):
        self.returns = dict()
        self.kalman = dict()
        self.ama = dict()
        self.boll = dict()
        self.x = dict()
        self.x_1 = dict()
        self.y = dict()
        self.y_pred = dict()
        self.model = dict()
        self.metric = dict()
        for i, d in enumerate(self.datas):
            tkr = self.datas[i]._name
            self.returns[tkr] = bt.ind.PercentChange(self.datas[i].close, period=1)
            self.kalman[tkr] = Kalman(tkr, interval=self.p.interval)
            self.ama[tkr] = bt.ind.KAMA(self.datas[i].close, fast=self.p.ama_fast, slow=self.p.ama_slow)
            self.boll[tkr] = bt.indicators.BollingerBands(period=self.p.bol_window, devfactor=2)
            self.model[tkr] = self.p.model().model
            self.metric[tkr] = metrics.Rolling(self.p.metric(), self.p.metric_window)

    params = (
        ("verbose", False),
        ("log_file", f'../results/bt_test_{dt.now().strftime("%Y_%m_%d-%H_%M_%S")}.csv'),
        ("interval", "1m"),
        ("bol_window", 20),
        ("ama_fast", 6),
        ("ama_slow", 12),
        ("model", logisticReg),
        ("metric", metrics.ROCAUC),
        ("metric_window", 12),
        )

    def next(self):
        for data in self.datas:
            t = data.datetime.datetime(0)
            tkr = data._name
            o = data.open[0]
            h = data.high[0]
            l = data.low[0]
            c = data.close[0]
            v = data.volume[0]
            returns = self.returns[tkr]
            boll = self.boll[tkr]
            ama = self.ama[tkr]
            model = self.model[tkr]

            # print("{} - {} | O: {} H: {} L: {} C: {} V:{}".format(t, tkr, o, h, l, c, v))

            kal = self.kalman[tkr]
            kal._update(t, returns)

            if tkr in self.x:
                self.x_1[tkr] = self.x[tkr]

            self.x[tkr] = {
                "t": t,
                "o": o,
                "h": h,
                "l": l,
                "c": c,
                "v": v,
                "ret": returns[0],
                "kal": kal.state_mean.iloc[-1],
                "ama": ama[0],
                "bbh": boll.top[0],
                "bbm": boll.mid[0],
                "bbl": boll.bot[0],
            }

            if self.p.verbose:
                print(self.x[tkr])

            self.y_pred[tkr] = model.predict_proba_one(self.x[tkr])

            if returns > 0:
                self.y[tkr] = 1
            elif returns < 0:
                self.y[tkr] = -1
            else:
                self.y[tkr] = 0
            
            if tkr in self.x_1:
                model.learn_one(self.x_1[tkr], self.y[tkr])
                metric = self.metric[tkr]
                metric.update(self.y[tkr], self.y_pred[tkr])
                print(f'{self.p.metric.__name__}: {metric}')

            print(self.y[tkr])
