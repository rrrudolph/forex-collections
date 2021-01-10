import pandas as pd
import MetaTrader5 as mt5
import sqlite3
import time
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from create_db import path
import numpy as np
import datetime as dt
from create_db import setup_conn, path  # get the connection to the database
from ohlc_symbols import mt5_symbols, fin_symbols

conn, c = setup_conn(path)

# This module gets data and updates the database

# Notes: 
# lfd = low frequency data

# ~~~~~~~~~~~~~~~~~  TO DO:  ~~~~~~~~~~~~~~~~~~
''' REMEMBER TO REMOVE THE 'YEAR - 1" PART FROM FFCAL

I need to work with the ff cal data locally throughout the week and
only update the db on the weekend once the 'actuals' are locked in. 

Add some time randomization to the TE request so I
don't get flagged as being a bot '''




    def save_ohlc_to_db(df):

        df = pd.concat([df1, df2])
        for i in df.index:
            params = (  
                        df.loc[i, 'datetime'], 
                        df.loc[i, 'symbol'], 
                        df.loc[i, 'timeframe'], 
                        df.loc[i, 'open'], 
                        df.loc[i, 'high'], 
                        df.loc[i, 'low'], 
                        df.loc[i, 'close'], 
                        df.loc[i, 'volume']
                    )

            UpdateDB.c.execute("INSERT INTO ohlc_raw VALUES (?,?,?,?,?,?,?,?)", params)
        UpdateDB.conn.commit()

    def _get_latest_ohlc_datetime(symbol, timeframe):
        ''' Get the last ohlc timestamp for each symbol '''

        params = (symbol, timeframe)
        # If the db is blank this will error with 'NoneType'
        try:
            UpdateDB.c.execute('''SELECT datetime 
                                    FROM ohlc
                                    WHERE symbol == ?
                                    AND timeframe == ?
                                    ORDER BY datetime DESC
                                    LIMIT 1''', (params))
            return int(UpdateDB.c.fetchone()[0])
        except: TypeError
        pass

        # dealing with single param explanation:
        # https://stackoverflow.com/questions/16856647/sqlite3-programmingerror-incorrect-number-of-bindings-supplied-the-current-sta


def _set_start_time(symbol, timeframe, num_candles=999):
    ''' Start request 'start' time based on the last row in the db, 
        otherwise if there's no data just request a bunch of candles '''

    timeframe = str(timeframe)
    
    if UpdateDB._get_latest_ohlc_datetime(symbol, timeframe):
        start = UpdateDB._get_latest_ohlc_datetime(symbol, timeframe)
    else: 
        start = round(time.time()) - (num_candles * ohlc_symbols.seconds_per_candle[timeframe])

    return start

def finnhub_ohlc_request(symbol, timeframe, num_candles=999):
    ''' Import note, this doesn't account for weekend gaps. So requesting small 
        amounts of M1 candles can cause problems on weekends.  Max returned candles
        seems to be about 650.  Finnhub only returns closed candles, not the current candle.  '''

    # Ensure timeframe is a string
    timeframe = str(timeframe)
    symbol = str(symbol)
    
    start = _set_start_time(symbol, timeframe)
    end = time.time()

    if 'OANDA' in symbol:
        r = requests.get(f'https://finnhub.io/api/v1/forex/candle?symbol={symbol}&resolution={timeframe}&from={start}&to={end}&token=budmt1v48v6ped914310')
    else:
        r = requests.get(f'https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution={timeframe}&from={start}&to={end}&token=budmt1v48v6ped914310')
    
    # Check for a bad request
    r = r.json()
    if r['s'] == 'no_data':
        print('Finnhub OHLC Request error. Check parameters:')
        print(' - symbol:', symbol)
        print(' - timeframe:', timeframe)
        print(' - start:', start)
        print(' - end:  ', end)
        print('current time:', time.time())
    else:
        finnhub_df = pd.DataFrame(data=r)
        finnhub_df = finnhub_df.rename(columns={'t':'datetime', 'o':'open', 'h':'high', 'l':'low', 'c':'close', 'v':'volume'})

        # Ensure all data is SQLite friendly
        finnhub_df['symbol'] = str(symbol)
        finnhub_df['timeframe'] = str(timeframe)
        finnhub_df['datetime'] = finnhub_df['datetime'].astype(str)
        finnhub_df['open'] = finnhub_df['open'].astype(float)
        finnhub_df['high'] = finnhub_df['high'].astype(float)
        finnhub_df['low'] = finnhub_df['low'].astype(float)
        finnhub_df['close'] = finnhub_df['close'].astype(float)
        finnhub_df['volume'] = finnhub_df['volume'].astype(float)
        finnhub_df = finnhub_df[['datetime', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume']]
    
        return finnhub_df

def mt5_ohlc_request(symbol, timeframe):
    ''' Similar to Finnhub, MT5 will not return the current open candle. '''

    if not mt5.initialize(login=50341259, server="ICMarkets-Demo",password="ZhPcw6MG"):
        print("\n ~~~ ERROR: MT5 initialize() failed, error code =", mt5.last_error())
        quit()

    # Ensure proper data formatting
    timeframe = str(timeframe)

    start = _set_start_time(symbol, timeframe)
    end = time.time()

    # Make request
    timeframe = ohlc_symbols.mt5_timeframes[timeframe]
    data = mt5.copy_rates_range(symbol, timeframe, UpdateDB.start, UpdateDB.end)

    mt5_df = pd.DataFrame(data)
    mt5_df =mt5_df.rename(columns={'time': 'datetime', 'tick_volume': 'volume'})
        
        # Ensure all data is SQLite friendly
    mt5_df['symbol'] = str(symbol)
    mt5_df['timeframe'] = str(timeframe)
    mt5_df['datetime'] =mt5_df['datetime'].astype(str)
    mt5_df['open'] =mt5_df['open'].astype(float)
    mt5_df['high'] =mt5_df['high'].astype(float)
    mt5_df['low'] =mt5_df['low'].astype(float)
    mt5_df['close'] =mt5_df['close'].astype(float)
    mt5_df['volume'] =mt5_df['volume'].astype(float)

    return mt5_df


def main():        
    ''' Open the db connection, update the data '''
    
    UpdateDB.connect_to_db()

    mt5_dfs = [mt5_ohlc_request(symbol, 5) for symbol in ohlc_symbols.mt5_symbols]
    fin_dfs = [finnhub_ohlc_request(symbol, 5) for symbol in ohlc_symbols.fin_symbols]

    ohlc_dfs = pd.concat([mt5_dfs, fin_dfs])

    UpdateDB.save_ohlc_to_db(ohlc_dfs)

    # Update MT5 data (confirmed working 1/1/2021)
    # for symbol in ohlc_symbols.mt5_symbols:
    #     UpdateDB.save_ohlc_to_db(symbol, 5, _finnhub_ohlc_request())

    # Update Trading Economics data 
    # for country in ohlc_symbols.te_countries:
    #     UpdateDB.save_te_data_to_db(country)

    UpdateDB.conn.close()

main()
