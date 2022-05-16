import json
import asyncio
from pprint import pprint as pp
from binance.client import AsyncClient
from binance.streams import BinanceSocketManager

from setEnv import set_binance_env

LIVE, FUTS, TEST, BINANCE_KEY, BINANCE_SECRET = set_binance_env(FUTS=False)


async def get_ac_info(client: AsyncClient, FUTS: bool):
    ac_info =  await client.futures_account() if FUTS else await client.get_account()
    print('A/C INFO:'), pp(ac_info), print('')
    return json.dumps(ac_info)


async def get_ac_snapshot(client: AsyncClient, FUTS: bool):
    '''Not working for futures testnet'''
    ac_snapshot = await client.get_account_snapshot(type='FUTURES') if FUTS else await client.get_account_snapshot(type='SPOT')
    print('A/C SNAPSHOT:'), pp(ac_snapshot)
    return json.dumps(ac_snapshot)


async def get_ac_balance(client: AsyncClient, FUTS: bool):
    ac_balance = await client.futures_account_balance() if FUTS else None
    print('A/C BALANCE:'), pp(ac_balance)
    return json.dumps(ac_balance)


async def main():

    client = await AsyncClient.create(BINANCE_KEY, BINANCE_SECRET, testnet=TEST)

    ac_info =  await get_ac_info(client, FUTS)
    # ac_snapshot = await get_ac_snapshot(client, FUTS)
    ac_balance =  await get_ac_balance(client, FUTS)

    await client.close_connection()

if __name__ == "__main__":
    '''Run this file to check Binance account balances.'''

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())