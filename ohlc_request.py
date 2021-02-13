import pandas as pd
import sqlite3
import time
import fxcmpy
import socketio
import requests
import json
from datetime import datetime
import concurrent.futures
from create_db import setup_conn, ohlc_db
from symbols_lists import fx_symbols, fin_symbols, fx_timeframes, seconds_per_candle
from tokens import fin_token, fxcm_con

conn, c = setup_conn(ohlc_db)



def _get_latest_ohlc_datetime(symbol, timeframe, c=c):
    ''' Get the last ohlc timestamp for each symbol '''

    # If the db is blank this will error with 'NoneType'
    try:
        c.execute(f'''SELECT datetime 
                                FROM ohlc
                                WHERE symbol = {symbol}
                                AND timeframe = {timeframe}
                                ORDER BY datetime DESC
                                LIMIT 1''')
        return c.fetchone()[0]
    
    except TypeError:
        pass


def _set_start_time(symbol, timeframe, num_candles=999):
    ''' Set request 'start' time based on the last row in the db, 
        otherwise if there's no data just request a bunch of candles. '''

    timeframe = str(timeframe)
    
    if _get_latest_ohlc_datetime(symbol, timeframe):
        start = _get_latest_ohlc_datetime(symbol, timeframe)
    else: 
        start = round(time.time()) - (num_candles * seconds_per_candle[timeframe])

    return start


def finnhub_ohlc_request(symbol, timeframe, token=fin_token):
    ''' This doesn't account for weekend gaps. So requesting small 
        amounts of M1 candles can cause problems on weekends.  Max returned candles
        seems to be about 650-800.  Finnhub only returns closed candles, not the current candle.  
        Rate limit is 60 calls per minute.'''

    
    start = _set_start_time(symbol, timeframe)
    end = round(time.time())

    if 'OANDA' in symbol:
        r = requests.get(f'https://finnhub.io/api/v1/forex/candle?symbol={symbol}&resolution={timeframe}&from={start}&to={end}&token={token}')
    else:
        r = requests.get(f'https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution={timeframe}&from={start}&to={end}&token={token}')
    
    # Check for a bad request
    # print(len(r))
    if str(r) == '<Response [422]>':
        print(r, 'Finnhub unable to process')
    if str(r) == '<Response [429]>':
        print(r, 'Finnhub max requests exceeded')

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

        # Ensure all data is SQLite friendly
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        df['datetime'] = pd.to_datetime(df.datetime, unit='s').astype(str)
        # df['datetime'] = df['datetime'].astype(str)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df = df[['datetime', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume']]
    
        return df


def fxcm_ohlc_request(symbol, timeframe):
    ''' Request candles from fxcm '''

    # FXCM requires these in datetime format
    start = datetime.fromtimestamp(_set_start_time(symbol, timeframe, num_candles=5000))
    end = datetime.fromtimestamp(time.time())

    # start = datetime(2021, 1, 27)
    # end = datetime(2021, 1, 28)
    # Make request
    try:
        df = fxcm_con.get_candles(symbol, 
                            period=fx_timeframes[timeframe],
                            start=start, 
                            stop=end,
        )
    except:
        'fxcm fail'
        fxcm_con.close()
    
    df = df.rename(columns={
            'bidopen': 'open', 
            'bidhigh': 'high',
            'bidlow': 'low',
            'bidclose': 'close',
            'tickqty': 'volume',
            }
    )
        
    # I dont think sqlite can save datetime formats so convert to str
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df['datetime'] = df.index.astype(str)
    df['timeframe'] = timeframe
    df['symbol'] = symbol
    df = df.reset_index(drop=True)

    return df


def _timeframes_to_request():


    times = {
        'D': ['1', '5', '15', '60', 'D'],
        0: ['1', '5', '15', '60'],
        15: ['1', '5', '15'],
        5: ['1', '5'],
        1: ['1']  
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
            timeframes = times['D']
            return timeframes
        
        # '15' should happen 4 times per hour (for example) so need a modulo function
        for t in [15, 5]:

            if t in times:
                timeframes = times[t]
                return timeframes

            elif minute % t == 0:
                timeframes = times[t]
                return timeframes

        if second == 0:
            timeframes = times[1]
            return timeframes


def _fxcm(timeframes):
    ''' Run fxcm requests '''

    fxcm_df = [fxcm_ohlc_request(s, t) for s in fx_symbols for t in timeframes]
    fxcm_df = pd.concat(fxcm_df)

    return fxcm_df


def _finnhub(timeframes):
    ''' Run the finnhub requests, but keep in mind the 60 calls per minute max '''

    # Track elapsed time and only make 1 request per second if necessary
    num_calls = len(timeframes) * len(fin_symbols)
    if num_calls >= 60:

        fin_df = pd.DataFrame()

        for symbol in fin_symbols:
            for timeframe in timeframes:
                
                start_time = time.time()

                fin_df.append(finnhub_ohlc_request(symbol, timeframe))

                elapsed_time = time.time() - start_time
                if elapsed_time < 1:
                    time.sleep(1 - elapsed_time)
    
    else:
        fin_df = [finnhub_ohlc_request(s, t) for s in fin_symbols for t in timeframes]
        fin_df = pd.concat(fin_df)

   
    return fin_df


def ohlc_request_handler():
    ''' Continually check the time and if a candle has recently closed request the new data. '''

    while True:
        # print('ohlc true')  ok that at least works
        timeframes = _timeframes_to_request()
        if timeframes:
            # print('minute: ', minute)
            print('timeframes inside request handler:')
            print(timeframes)

            # Create separate threads for each server
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     t1 = executor.submit(_fxcm(timeframes))
            #     t2 = executor.submit(_finnhub(timeframes))
            
            # fxcm_df = t1.result()
            # fin_df = t2.result()

            fxcm_df = _fxcm(timeframes)
            fin_df = _finnhub(timeframes)

            df = pd.concat([fxcm_df,fin_df])

            # Save the data to the database
            for symbol in df.symbol.unique():
                unique_df = df[df.symbol == symbol]
                unique_df.to_sql(f'{symbol}', conn, if_exists='append', index=False)

ohlc_request_handler()

# def save_raw_ohlc(fx_symbols=fx_symbols, fin_symbols=fin_symbols, conn=conn):
#     ''' Update the database with ohlc data. Note, there is no con.close() 
#     present within this function. '''

#     while True:

#         df = pd.DataFrame()
#         df = ohlc_request_handler()

#         if len(df) > 1:

#             # Put each symbol's data into its own table
#             for symbol in df.symbol.unique():
#                 unique_df = df[df.symbol == symbol]
#                 unique_df.to_sql(f'{symbol}', conn, if_exists='append', index=False)
            