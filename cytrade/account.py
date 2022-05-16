from btBinance import execute
from strategies import CloseAll


if __name__ == '__main__':
    '''Example file for closing all open BTC/USDT spot positions, but keeping 1 BTC open. (maintain=1) '''
    execute(base='BTC', quote='USDT', strategy=CloseAll, args=dict(maintain=1,verbose=True))