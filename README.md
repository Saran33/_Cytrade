# CyTrade

## CyTrade Algorithmic Trading
CyTrade is a repo with some examples for stream-learning-based Machine Learning strategies for trading cryptocurrency. These examples are built with [Cytrader](https://github.com/Saran33/cytrader) (a cythonized fork of Backtrader), [Ct-CCXT-Store](https://github.com/Saran33/ct-ccxt-store) (a fork of Backtrader-CCXT-Store) and [River](https://riverml.xyz/latest/).

This repository is accessible at:
[CyTrade](https://github.com/Saran33/_CyTrade)

#### To install from git:
`pip install git+https://github.com/Saran33/_CyTrade.git`
- To build from cython files:
```zsh
cd cytrade
python setup.py build_ext --inplace
```

## USAGE:
Cytrade is set up to use environment variables for API keys, as well as to indicate whether you are live trading or usng a demo account, and to indicate whether you are trading spot or futures. Set a `.env` file in the `cytrade` directory, or alternatively set environemnt varibales with your cloud provider. e.g.:
```zsh
LIVE=False
FUTS=False

BINANCE_KEY=YOUR_LIVE_KEY
BINANCE_SECRET=YOUR_LIVE_SECRET

BINANCE_TEST_KEY=YOUR_TEST_KEY
BINANCE_TEST_SECRET=YOUR_TEST_SECRET

BINANCE_FUTS_TEST_KEY=YOUR_FUTURES_DEMO_KEY
BINANCE_FUTS_TEST_SECRET=YOUR_FUTURES_DEMO_SECRET
```

#### Stream Learning strategy
- The `backtests/btRiver.ipynb` notebook file contains an example logistic regression strategy, integrating River with Backtrader.
- It is using the Kelly Critereon to size positions (solving for the maximum growth rate is done in cython, which is 200 times faster than if done in scipy and numpy `cytrade/cyutils/cyzers.pyx`).