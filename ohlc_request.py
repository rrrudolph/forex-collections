import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import sqlite3
import time
# import fxcmpy
import socketio
import requests
import json
from datetime import datetime
import concurrent.futures
from create_db import ohlc_db
from symbols_lists import fin_symbols, seconds_per_candle
from tokens import fin_token, mt5_login, mt5_pass, mt5_server, bonds_sheet

ohlc_con = sqlite3.connect(ohlc_db)

''' I have to subtract 6 hours from Finnhub data  to get it into CST.
so a couple conversions take place from saving and requesting '''


def _market_open():
    ''' Check if the FX market is currently open. '''

    # monday == 0, sunday == 6
    minute = datetime.now().minute
    hour = datetime.now().hour
    day = datetime.now().weekday()

    # Monday thru Thursday
    if 0 <= day <= 4:
        return True

    # Friday before 430pm CST
    if day == 5 and hour < 4:
        return True
    
    elif day == 5 and hour == 4 and minute < 30:
        return True

    # Sunday after 430pm CST
    if day == 6 and hour > 4:
        return True
    
    elif day == 6 and hour == 4 and minute > 30:
        return True

    # If nothings been returned at this point market is closed
    return False

def _read_last_datetime(tablename, conn):
    ''' Get the last datetime for a given table '''

    # the tables in CORR_CON list the index as "index" but 
    # still need to be accessed by being called "datetime" ??

    df = pd.read_sql(f"""SELECT * FROM {tablename}""", conn)
    df.datetime = pd.to_datetime(df.datetime)

    return df.datetime.tail(1).values[0]

def _set_request_start_timestamp(timeframe, tablename, conn):
    ''' Get the last datetime entry in the db, add one period of whatever
    timeframe is passed, and return a timestamp of that value '''

    last = _read_last_datetime(tablename, conn)

    if last:
    
        # Add 6 hours to get it back in line with the current GMT and make into timestamp
        request_start = last + pd.Timedelta('6 hours')
        request_start = datetime.timestamp(request_start)

        # Now move the request 1 bar into the future to not duplicate the last row
        request_start += pd.Timedelta(timeframe)

        return request_start


def finnhub_ohlc_request(symbol, timeframe):
    ''' This doesn't account for weekend gaps. So requesting small 
        amounts of M1 candles can cause problems on weekends.  Max returned candles
        seems to be about 650-800.  Finnhub only returns closed candles, not the current candle. '''

    
    start = _set_request_start_timestamp(symbol, timeframe)
    end = round(time.time())

    if 'OANDA' in symbol:
        r = requests.get(f'https://finnhub.io/api/v1/forex/candle?symbol={symbol}&resolution={timeframe}&from={start}&to={end}&token={fin_token}')
    else:
        r = requests.get(f'https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution={timeframe}&from={start}&to={end}&token={fin_token}')
    
    # Check for a bad request
    # print(len(r))
    if str(r) == '<Response [422]>':
        print(r, '~~~ Finnhub unable to process ~~~ \n')
    if str(r) == '<Response [429]>':
        print(r, '~~~ Finnhub max requests exceeded ~~~ \n')

    r = r.json()
    if r['s'] == 'no_data':
        print('Finnhub OHLC Request error. Check parameters:')
        print(' - symbol:', symbol)
        print(' - timeframe:', timeframe)
        print(' - start:', start)
        print(' - end:  ', end)
        print('current time:', time.time())

    else:
        df = pd.DataFrame(data=r)

        df = df.rename(columns={'t':'datetime', 'o':'open', 'h':'high', 'l':'low', 'c':'close', 'v':'volume'})
        df = df.set_index(df.datetime, drop=True)
        df.index = pd.to_datetime(df.index, unit='s')
        df.index -= pd.Timedelta('6 hours')
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        # Save
        df.to_sql(f'{symbol}', ohlc_con, if_exists='append', index=True)


def timeframes_to_request():
    ''' The keys in this dict are checked by parsing the current time '''

    times = {
        0: [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, 
            mt5.TIMEFRAME_M30, mt5.TIMEFRAME_H1],
        30: [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M30],
        # '20': [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M10, mt5.TIMEFRAME_M20],
        15: [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15],
        # '10': [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M10],
        5: [mt5.TIMEFRAME_M5],
        1: [mt5.TIMEFRAME_M1],
        # 'test': [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M2, mt5.TIMEFRAME_M3, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M10, mt5.TIMEFRAME_M15, 
        #     mt5.TIMEFRAME_M30, mt5.TIMEFRAME_H1],
    }

    # Reset the logic gate
    timeframes = None

    while True:

        # Parse the time to get the hour and minute
        hour = int(datetime.today().hour)
        minute = int(datetime.today().minute)
        second = int(datetime.today().second)

        # Get the list of timeframes to request depending on current time
        # starting with the highest timeframes to the lowest
        if hour == 0 and minute == 0:
            timeframes = times[0]
            return timeframes
        
        # '15' should happen 4 times per hour (for example) so need a modulo function
        for t in [15, 5]:

            if minute % t == 0:
                timeframes = times[t]
                return timeframes

        # if second == 0:
        #     timeframes = times['test']
        #     return timeframes

def _format_mt5_data(df):
    
    try:
        df = df.rename(columns={'time': 'datetime', 'tick_volume': 'volume'})
        df.datetime = pd.to_datetime(df.datetime, unit='s')
        df.datetime = df.datetime - pd.Timedelta('8 hours')
        df.index = df.datetime
        df = df[['open', 'high', 'low', 'close', 'volume']]
    except:
        print('Failed to format the dataframe:')
        print(df)

    return df

def mt5_ohlc_request(symbol, timeframe, num_candles=70):
    ''' Get a formatted df from MT5 '''

    # A request to MT5 can occasionally fail. Retry a few times to connect 
    # and a few more times to receive data
    for _ in range(2):
        if mt5.initialize(login=mt5_login, server=mt5_server,password=mt5_pass):
            
            for _ in range(5):
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)  
                if rates is not None:
                    if len(rates) > 0:
                        df = pd.DataFrame(rates)
                        df = _format_mt5_data(df)
                        return df

            print(f'\n ~~~ Request to MT5 failed. [{symbol} {timeframe}] ~~~')
            return
        
        # If init failed pause before retry
        time.sleep(0.1)

    print("MT5 initialize() failed, error code =", mt5.last_error())

def _delete_duplicate_rows(symbol):

    df = pd.read_sql(f'SELECT * FROM {symbol}', ohlc_con)
    df = df.drop_duplicates()
    df.to_sql(f'{symbol}', ohlc_con, if_exists='replace', index=True)


def read_10y_bonds_and_clean(sheet=bonds_sheet) -> pd.DataFrame:
    ''' get the bond data and normalize for plotting with ohlc index data '''
    
    df = pd.DataFrame(sheet.get_all_values())

    # Scrub n Clean
    df = df.set_index(df.iloc[:, 0])
    df.columns = df.iloc[0]
    df = df.iloc[1:, 1:]
    df.index = pd.to_datetime(df.index)
    df = df.replace('', np.nan)
    df = df.fillna(method='ffill')
    df = df.astype(float)

    # Only interested in 10s
    df = df.iloc[:, 1::2]

    return df


def infinite_request():
    ''' weekends will cause an error that I don't catch '''

    while True:
        
        hour = datetime.today().hour - 6
        day = datetime.weekday(datetime.today())

        #     tues-thurs           monday after 6am          friday before 3pm
        if (1 <= day <= 3) or (day == 0 and hour >= 6) or (day == 4 and hour <= 3):

            next_candle_time = _read_last_datetime(fin_symbols[0], '1')
            
            try:
                # First time running needs this
                if next_candle_time is None:
                    for symbol in fin_symbols:
                        finnhub_ohlc_request(symbol, '1')
                
                else:
                    if next_candle_time >= time.time():
                        for symbol in fin_symbols:
                            finnhub_ohlc_request(symbol, '1')

                # once the markets are open and I know things will go smoothly, 
                # delete any duplicates that may have been created
                if day == 0 and hour == 9:
                    for symbol in fin_symbols:
                        _delete_duplicate_rows(symbol)

            except:
                time.sleep(5*60)
                infinite_request()

if __name__ == '__main__':

    print(mt5_ohlc_request('PKG.NYSE', mt5.TIMEFRAME_M5, num_candles=840))

    # infinite_request()
