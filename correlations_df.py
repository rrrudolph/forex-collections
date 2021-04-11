import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import concurrent.futures
from tqdm import tqdm
import pathlib
import sqlite3
from itertools import chain
from symbols_lists import mt5_symbols, fin_symbols, spreads, trading_symbols
from ohlc_request import mt5_ohlc_request, _read_last_datetime
from create_db import ohlc_db, correlation_db
from tokens import mt5_login, mt5_pass

'''
I want to have multi TF correlation values but I can't use HTFs like D1
when Im filling a historical db. If I did I'd end up with choppy value
lines since I'd only get a single D1 value per day.  So to emulate real historical 
data Im going to use M5 candles but multiply all applicable values (correlation period,
look backs).  

The symbols I scan for correlation have various market open hours.  In order to normalize 
the correlation values between markets that have different hours than the FX pairs, I first
scan for corr using only the rows where each market is open.  However, ultimately I want to
use all the corr that's found in order to plot a single "value" line overlayed on a FX major
ohlc chart.  To have data consistently disappear in the overnight session leads to a lot of
choppiness in that line.  So after the correlation values are found, I reindex the corr data
to add in the overnight session and I just do a 'ffill' on the nans. 
'''

# How much historical data you want 
HIST_CANDLES = 51840  # 160 days
historical_fill = None

# Settings for each time horizon 
settings = {
    # 'LTF': {
    #     'CORR_PERIOD': 1440,  # 5 days
    #     'MIN_CORR': 0.65,
    # },
    'MTF': {
        'CORR_PERIOD': 5760,  # 20 days
        'MIN_CORR': 0.65,
    },
    'HTF': {
        'CORR_PERIOD': 17280,  # 60 days
        'MIN_CORR': 0.60,
    },
    'SHIFT_PERIODS': [0, 10, 50, 150, 300, 500, 700],
    'trading_symbols': trading_symbols,
    'corr_symbols': chain(mt5_symbols['others'], spreads),
}


# This needs to be changed to the ubuntu laptop path
OHLC_CON = sqlite3.connect(ohlc_db)
CORR_CON = sqlite3.connect(correlation_db)

# Path where final parquet files will get saved
p = r'C:\Users\ru\forex\db\correlation'



def _normalize(s):
    ''' Normalize the series '''
    
    s = (s - min(s)) / (max(s) - min(s))

    return s

def _get_data(symbol, hist_fill=True, spread=None):

    # If this is a historical overwrite
    if hist_fill == True:
        candle_count = HIST_CANDLES

    else:
        # First figure out how much data is needed by getting the last correlation timestamp
        start = _read_last_datetime(f'{symbol}_MTF', CORR_CON)
        if start:

            minutes_diff = (pd.Timestamp.now() - start).total_seconds() / 60.0
            
            # How many 5 minute periods exist in that range
            candle_count = minutes_diff // 5
            candle_count = vars['period'] + int(candle_count)

        # If that table didn't exists in the db do a hisorical fill
        # (not ideal cuz I'll end up with oversized tables in a lot of cases)
        else:
            candle_count = HIST_CANDLES
            
    if not mt5.initialize(login=mt5_login, server="ICMarkets-Demo",password=mt5_pass):
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    # if this is being called by the _make_spread function, overwrite symbol
    if spread is not None:
        symbol = spread

    # mt5 request
    if symbol in mt5_symbols['others'] or symbol in mt5_symbols['majors']:
        df = mt5_ohlc_request(symbol, mt5.TIMEFRAME_M5, num_candles=candle_count)

    elif symbol in fin_symbols:
        df = pd.read_sql(f'SELECT * FROM {symbol}', OHLC_CON, parse_dates=False)
        df = df.set_index(df.datetime, drop=True)
        df.index = pd.to_datetime(df.index)

    # Close is all thats needed
    return df.close

def _make_db():
    ''' I'll be making so many repeated disk reads that it makes sense to
    read it all into memory once '''
    
    # Get unique symbol values from spreads
    split_spreads = []
    for s in spreads:

        split_spreads.append(s.split('_')[0])
        split_spreads.append(s.split('_')[1])

    all_symbols = []
    all_symbols.extend(mt5_symbols['majors'])
    all_symbols.extend(mt5_symbols['others'])
    all_symbols.extend(split_spreads)
    all_symbols = set(all_symbols)

    # Create a future for each symbol and save the data to a dict
    # where the key is the symbol name and the value is a df
    with concurrent.futures.ThreadPoolExecutor() as executor:
        
        futures = [] 
        for symbol in all_symbols:
            futures.append(executor.submit(_get_data, symbol=symbol))
        db = {} 
        for symbol, data in zip(all_symbols, concurrent.futures.as_completed(futures)):
            db[symbol] = data.result()

        # Verify all data was retrieved
        if len(db) < len(all_symbols):
            print(' ~~~ you got 99 problems + data request issues')
            print(f' ~~~ missing {set(db.keys()) ^ all_symbols}') # (a | b) - (a & b), the union of both sets minus the intersection
    
    return db

def _make_spread(spread, db):
    ''' Normalize, combine.'''
    
    # First parse spread
    symbol_1 = spread.split('_')[0]
    symbol_2 = spread.split('_')[1]
    
    # Normalize both 
    symbol_1 = _normalize(db[symbol_1])
    symbol_2 = _normalize(db[symbol_2])

    # Align lengths to match the df with least data
    symbol_1 = symbol_1[symbol_1.index.isin(symbol_2.index)]
    symbol_2 = symbol_2[symbol_2.index.isin(symbol_1.index)]
    
    spread = symbol_1 - symbol_2  

    return spread

def _save_result_to_df(cor_rows, overnight, cor_symbol, shift):
    ''' Save close prices and corr value for any correlated pair found before continuing to the next symbol. '''
    
    corr_data = pd.DataFrame(index=cor_rows)
    corr_data.loc[cor_rows, f'{cor_symbol}'] = overnight.loc[cor_rows, 'cor_close']
    corr_data.loc[cor_rows, f'{cor_symbol}_corr'] = round(overnight.loc[cor_rows, 'cor'], 3)
    corr_data.loc[cor_rows, f'{cor_symbol}_shift'] = shift

    return corr_data

def _find_correlation(trading_symbol, cor_symbol, db, tf):
    ''' Create a df with the close prices of the key symbol and cor
    symbol.  Look for correlation above a certain threshold over a
    few different shift() amounts '''

    if cor_symbol in spreads:
        cor = _make_spread(cor_symbol, db)
    else:
        cor = db[cor_symbol]

    # Ensure matching df lengths (drop the overnight candles if there are any)
    key = db[trading_symbol][db[trading_symbol].index.isin(cor.index)].copy() # to avoid warnings
    cor = cor[cor.index.isin(key.index)]

    key = _normalize(key)
    cor = _normalize(cor)

    # Iter various shifts to find highest cor and save the data to this dict
    shift_info = {
        'data': pd.DataFrame(dtype=np.float32),
        'best_sum': 0,
        'shift': 0
    }

    for shift in settings['SHIFT_PERIODS']: 
        cor_values = key.rolling(settings[tf]['CORR_PERIOD']).corr(cor.shift(shift))
        cor_values = cor_values.dropna()

        # Update the shift dict
        if abs(cor_values.sum()) > shift_info['best_sum']:
            shift_info['data'] = round(cor_values, 3)
            shift_info['best_sum'] = abs(cor_values.sum())
            shift_info['shift'] = shift
    
    # If nothing is found jump to the next symbol
    if len(shift_info['data']) == 0:
        # print("len(shift_info['data']) == 0")
        return

    # To eliminate choppiness of the value line I plot later, I want to bring
    # any overnight gaps back in and ffill the nans. I’ll use the original db[trading_symbol] index for this
    overnight = pd.DataFrame(db[trading_symbol])
    overnight['cor'] = shift_info['data']
    overnight['cor_close'] = cor
    overnight = overnight.fillna(method='ffill')
    overnight = overnight.dropna()

    # I'll keep the normalized close values so that 
    # later on I can scan the symbols which have the highest cor with the value line
    overnight.cor_close = _normalize(overnight.cor_close)

    # Get list of rows where correlation is above threshold
    cor_rows = overnight.cor[abs(overnight.cor) > settings[tf]['MIN_CORR'] * 0.7].index
    
    if not cor_rows.empty:
        corr_data = _save_result_to_df(cor_rows, overnight, cor_symbol, shift_info['shift'])
        return corr_data

def _symbol_scanning_manager(trading_symbol, cor_symbols, db, tf):
    ''' This will spin up threads to handle all of the
    different correlation comparisons for a given symbol. Once all threads
    return, make a single df of their values and save to disk. '''

    # Loop thru the comparison symbols
    with concurrent.futures.ThreadPoolExecutor() as executor:
        
        futures = [] 
        for cor_symbol in cor_symbols:
            futures.append(executor.submit(_find_correlation, trading_symbol=trading_symbol, cor_symbol=cor_symbol, db=db, tf=tf))
        
        cor_db = {} 
        for dict in concurrent.futures.as_completed(futures):
            # Unpack the key and value
            result = dict.result()
            if result is not None:
                try:
                    name = list(result.keys())[0]
                except Exception:
                    print(result)
                    print(f'A thread inside {trading_symbol} failed.')
                    
                cor_db[name] = result[name]
                # print(name)
                # print(result[name])

    # Compile a dataframe of the dicts, adding the trading symbol's
    # normalized close values into it for easy reference later on
    df = pd.DataFrame()
    df = df.join(cor_db.values(), how='outer')
    df[trading_symbol] = _normalize(db[trading_symbol])
    print(f'{trading_symbol} done. Found {len(df)} periods of correlation.')

    return df

def scan(trading_symbols, corr_symbols, hist_fill=historical_fill):
    ''' Find correlation between trading_symbols and corr_symbols
    over a rolling window.  Threading is used to load all ohlc data into memory.
    Then each trading_symbol is given its own process, with threading inside each. '''
    
    # Overwrite global variable
    if hist_fill == True:
        historical_fill = True

    # Build out a df of all needed data
    print('Assembling in-memory database..!')
    db = _make_db()
    print('DB assembled.')
    # The outer loop will iter over the various correlation periods to emulate MTF value
    # Each corr period will get saved to its own table in the db
    for tf in settings:

        # Spin up new processes for each symbol in fx majors
        with concurrent.futures.ProcessPoolExecutor() as executor:
        
            futures = [] 
            for symbol in trading_symbols:
                futures.append(executor.submit(_symbol_scanning_manager, trading_symbol=symbol, cor_symbols=corr_symbols, db=db, tf=tf))
            
            corr = {} 
            for dict in concurrent.futures.as_completed(futures):
                result = dict.result()
                if result is not None:
                    # Unpack the key and value
                    name = list(result.keys())[0]
                    corr[name] = result[name]
            
        # result[name] is a df containing all of the correlation for a given trading_symbol.
        # Save those dfs to disk.
        for name, df in corr.items():
            df.to_parquet(pathlib.Path(p, f'{name}.parquet'), index=True)

        # Verify all data was retrieved
        if len(corr) < len(mt5_symbols['majors']):
            print(' ~~~ you got 99 problems + data request issues')
            print(f" ~~~ missing {set(corr.keys()) ^ set(mt5_symbols['majors'])}") # (a | b) - (a & b), the union of both sets minus the intersection
            

    OHLC_CON.close()
    CORR_CON.close()

if __name__ == '__main__':


    print('\ntime started:', datetime.now())
    s = time.time()
    # scan(historical_fill=False)
    scan(trading_symbols=settings['trading_symbols'], corr_symbols=settings['corr_symbols'])
    print('minutes elapsed:', (time.time() - s)/60)
