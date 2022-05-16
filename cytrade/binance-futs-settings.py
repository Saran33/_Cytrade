import os
import sys
from pprint import pprint

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root + '/python')

import ccxt  # noqa: E402

print('CCXT Version:', ccxt.__version__)

from setEnv import set_binance_env
LIVE, FUTS, TEST, BINANCE_KEY, BINANCE_SECRET = set_binance_env(FUTS=True)

# Must read before your start:
#
#     - https://github.com/ccxt/ccxt/wiki/Manual
#     - https://github.com/ccxt/ccxt/wiki/Manual#implicit-api-methods
#     - https://github.com/ccxt/ccxt/wiki/Manual#unified-api
#
# In short, Binance's API is structured as follows and you should understand
# the meaning and the difference between ISOLATED vs CROSSED margin mode and
# the difference between Hedged positions vs One-way positions.
#
#     - wapi: funding for withdrawals and deposits (wapi)
#     - api: spot (api)
#     - sapi: spot margin
#     - CROSSED margin mode
#         - Hedged positions
#         - One-way positions
#     - ISOLATED margin mode
#         - Hedged positions
#         - One-way positions
#     - fapi: swap/perpetual futures margin
#     - CROSSED margin mode
#         - Hedged positions
#         - One-way positions
#     - ISOLATED margin mode
#         - Hedged positions
#         - One-way positions
#     - dapi: classic delivery futures margin
#     - CROSSED margin mode
#         - Hedged positions
#         - One-way positions
#     - ISOLATED margin mode
#         - Hedged positions
#         - One-way positions
#
# You should pick the following:
#
#     1. which API you want to trade (fapi, i believe)
#     2. which specific margin mode you want (CROSSED or ISOLATED)
#     3. which specific position mode you want (Hedged or One-way)
#
# Differences in margin modes:
#
#     - CROSSED margin mode = you have one futures-margin account for all your positions,
#       if some position requires too much margin, your entire account is affected,
#       leaving less margin for the other positions,
#       thus you share the same margin _"across"_ all your positions
#
#     - ISOLATED margin mode = you have separate futures-margin for each of your positions,
#       if some position runs out of margin the other positions are not affected,
#       thus your positions are _"isolated"_ from one another
#
# Difference in position modes:
#
#     - One-way position mode - when you're in this mode
#       there's no such things as LONG or SHORT positions.
#       You just buy or sell a number of contracts, and
#       if the price goes down, your PnL goes negative,
#       if the price goes up, your PnL is positive.
#       Thus, the position operates `BOTH` ways, both long and short at the same time,
#       the notion of "long" and "short" is abstracted away from you,
#       so there's only one way the position can go and that way is called "BOTH".
#
#     - Hedge mode - you either enter a `LONG` position or a `SHORT` position and
#       your PnL calculation rules depend on that
#       so there's a number of ways a position can go
#
# Which specific mode of trading (margin mode + position mode) do you want?


def table(values):
    first = values[0]
    keys = list(first.keys()) if isinstance(first, dict) else range(0, len(first))
    widths = [max([len(str(v[k])) for v in values]) for k in keys]
    string = ' | '.join(['{:<' + str(w) + '}' for w in widths])
    return "\n".join([string.format(*[str(v[k]) for k in keys]) for v in values])


exchange = ccxt.binance({
    'apiKey': BINANCE_KEY,
    'secret': BINANCE_SECRET,
    'options': {
        'defaultType': 'future',
    },
})

if TEST:
    print("DEMO ACCOUNT")
    exchange.set_sandbox_mode(True)

markets = exchange.load_markets();


def get_futures_balance(symbol='BTC/USDT'):
    market = exchange.market(symbol)

    exchange.verbose = True  # UNCOMMENT THIS AFTER LOADING THE MARKETS FOR DEBUGGING

    print('----------------------------------------------------------------------')

    print('Fetching your balance:')
    response = exchange.fetch_balance();
    pprint(response['total'])  # make sure you have enough futures margin...
    # pprint(response['info'])  # more details

    print('----------------------------------------------------------------------')
    return response['total'];


def get_futures_positions(symbol='BTC/USDT'):
    '''https://binance-docs.github.io/apidocs/futures/en/#position-information-v2-user_data'''

    print('Getting your positions:')
    response = exchange.fapiPrivateV2_get_positionrisk();
    print(table(response))

    print('----------------------------------------------------------------------')
    symb = symbol.replace('/', '')
    res = [r for r in response if r.get('symbol') == symb]
    return res


def get_oneway_or_hedge():
    # https://binance-docs.github.io/apidocs/futures/en/#change-position-mode-trade

    print('Getting your current position mode (One-way or Hedge Mode):')
    response = exchange.fapiPrivate_get_positionside_dual();
    if response['dualSidePosition']:
        print('You are in Hedge Mode')
    else:
        print('You are in One-way Mode')
    print('----------------------------------------------------------------------')
    return response['dualSidePosition']


## Change SETTINGS:
def set_hedge_mode(hedge=False):
    if not hedge:
        print('Setting your position mode to One-way:')
    else:
        print('Setting your positions to Hedge mode:')

    response = exchange.fapiPrivate_post_positionside_dual({
        'dualSidePosition': hedge,
    })
    print(response)
    print('----------------------------------------------------------------------')


def set_cross_mode(symbol='BTC/USDT', cross=False):
    '''https://binance-docs.github.io/apidocs/futures/en/#change-margin-type-trade'''

    market = exchange.market(symbol)
    mtype = 'CROSSED' if cross else 'ISOLATED'

    print(f'Changing your', symbol, 'position margin mode to {mtype}:')
    response = exchange.fapiPrivate_post_margintype({
        'symbol': market['id'],
        'marginType': mtype,
    })
    print(response)
    print('----------------------------------------------------------------------')
    return response


def binance_transfer(coin='USDT', amount=0., typ=1):
    ''' typ:
        # 1: transfer from spot account to USDⓈ-M futures account.
        # 2: transfer from USDⓈ-M futures account to spot account.
        # 3: transfer from spot account to COIN-Ⓜ futures account.
        # 4: transfer from COIN-Ⓜ futures account to spot account.

    https://binance-docs.github.io/apidocs/spot/en/#new-future-account-transfer-futures
    # OR:
    # https://github.com/ccxt/ccxt/blob/master/examples/js/binance-universal-transfer.js
    # https://docs.ccxt.com/en/latest/manual.html#transfers
    '''

    currency = exchange.currency(coin)
    msg = {
        1: 'funds from your spot account to your USDⓈ-M futures account:',
        2: 'funds from your USDⓈ-M account to your spot account:',
        3: 'funds from your spot account to your COIN-Ⓜ futures account:',
        4: 'funds from your COIN-Ⓜ futures account to your spot account:'
        }
    print('Moving', coin, msg.get(typ))

    response = exchange.sapi_post_futures_transfer({
        'asset': currency['id'],
        'amount': exchange.currency_to_precision(coin, amount),
        'type': typ,
    })
    print('----------------------------------------------------------------------')
    return response

def modify_iso_margin(symbol='BTC/USDT', amount=0., typ=1, side='BOTH'):
    '''for ISOLATED positions only
    typ:    1 = add position margin
            2 = reduce position margin
    side:   BOTH for One-way positions
            LONG or SHORT for Hedge Mode
    '''

    market = exchange.market(symbol)
    print('Modifying your ISOLATED', symbol, 'position margin:')
    response = exchange.fapiPrivate_post_positionmargin({
        'symbol': market['id'],
        'amount': amount,
        'positionSide': 'BOTH',
        'type': typ,
    })
    print('----------------------------------------------------------------------')
    return response


if __name__ == "__main__":
    get_futures_balance(symbol='BTC/USDT')
    get_futures_positions(symbol='BTC/USDT')
    get_oneway_or_hedge()



