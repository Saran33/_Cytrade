from decouple import config
from distutils.util import strtobool

def set_binance_env(FUTS=None):

    LIVE=strtobool(config('LIVE'))
    print("LIVE:", LIVE)

    if not FUTS:
        FUTS=strtobool(config('FUTS'))
    print("FUTURES:", FUTS)

    if FUTS==False:
        if LIVE == True:
            TEST=False
            BINANCE_KEY = config('BINANCE_KEY')
            BINANCE_SECRET = config('BINANCE_SECRET')
        else:
            TEST=True
            BINANCE_KEY = config('BINANCE_TEST_KEY')
            BINANCE_SECRET = config('BINANCE_TEST_SECRET')

    elif FUTS==True:
        if LIVE == True:
            TEST=False
            BINANCE_KEY = config('BINANCE_KEY')
            BINANCE_SECRET = config('BINANCE_SECRET')
        else:
            TEST=True
            BINANCE_KEY = config('BINANCE_FUTS_TEST_KEY')
            BINANCE_SECRET = config('BINANCE_FUTS_TEST_SECRET')

    return LIVE, FUTS, TEST, BINANCE_KEY, BINANCE_SECRET