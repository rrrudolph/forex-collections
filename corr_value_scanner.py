import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import concurrent.futures
from tqdm import tqdm
import pathlib
import sqlite3
from symbols_lists import mt5_symbols, fin_symbols, spreads, trading_symbols, mt5_timeframes
# from ohlc_request import mt5_ohlc_request, finnhub_ohlc_request, _read_last_datetime
from create_db import ohlc_db, correlation_db
from tokens import mt5_login, mt5_pass, bot, mt5_server


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

# this stuff cant be imported because ultimately the gspread api gets called and gives me a timeout error
# from all the processes/threads
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

    # If a period tag (like '_MTF') is passed, drop that
    if any(period in symbol for period in ['_LTF', '_MTF', '_HTF']):
        print(f'mt5_ohlc_request(): {symbol} -> {symbol[:6]}')
        symbol = symbol[:6]
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



# Settings for each time horizon 
windows = {
    'HTF': {
        'COR_PERIOD': 17280,  # 60 days
        'MIN_COR': 0.60,
    },
    # 'LTF': {
    #     'COR_PERIOD': 1440,  # 5 days
    #     'MIN_COR': 0.65,
    # }, 
    # 'MTF': {
    #     'COR_PERIOD': 5760,  # 20 days
    #     'MIN_COR': 0.65,
    # },
}

SHIFT_PERIODS = [0, 10, 50, 150, 300, 500, 700]

# This needs to be changed to the ubuntu laptop path
OHLC_CON = sqlite3.connect(ohlc_db)
COR_CON = sqlite3.connect(correlation_db)

# Path where final parquet files will get saved
p = r'C:\Users\ru\forex\db\correlation'

def _normalize(s: pd.Series) -> pd.Series:
    ''' Normalize a series '''
    
    assert isinstance(s, pd.Series), '_normalize() needs a Series'
    return (s - min(s)) / (max(s) - min(s))

def _get_data(symbol, hist_candles, timeframe=mt5_timeframes['M5'], spread=None) -> pd.Series:
    ''' Request the data. Right now its only coming from MT5. '''

    if not mt5.initialize(login=mt5_login, server="ICMarkets-Demo",password=mt5_pass):
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    # if this is being called by the _make_spread function, overwrite symbol
    if spread is not None:
        symbol = spread

    # mt5 request
    if symbol in mt5_symbols['others'] or symbol in mt5_symbols['majors']:
        df = mt5_ohlc_request(symbol, timeframe, num_candles=hist_candles)

    # Close is all thats needed
    return (symbol, df.close)

def _make_db(hist_candles):
    ''' I'll be making so many repeated disk reads that it makes sense to
    read it all into memory once '''
    
    # Get unique symbol values from spreads
    split_spreads = []
    for s in spreads:
        split_spreads.append(s.split('_')[0])
        split_spreads.append(s.split('_')[1])

    all_symbols = mt5_symbols['majors'] + mt5_symbols['others'] + split_spreads
    all_symbols = set(all_symbols)

    # Create a future for each symbol and save the data to a dict
    # where the key is the symbol name and the value is a Series of close prices
    with concurrent.futures.ThreadPoolExecutor() as executor:
        
        futures = [] 
        for symbol in all_symbols:
            futures.append(executor.submit(_get_data, symbol=symbol, hist_candles=hist_candles))
        db = {} 
        for future in concurrent.futures.as_completed(futures):
            some_tuple = future.result()
            symbol = some_tuple[0]
            data = some_tuple[1]
            db[symbol] = data

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
    
    cor_data = pd.DataFrame(index=cor_rows)
    cor_data.loc[cor_rows, f'{cor_symbol}'] = overnight.loc[cor_rows, 'cor_close']
    cor_data.loc[cor_rows, f'{cor_symbol}_cor'] = round(overnight.loc[cor_rows, 'cor'], 3)
    cor_data.loc[cor_rows, f'{cor_symbol}_shift'] = shift

    return cor_data

def _find_correlation(trading_symbol: str, cor_symbol: str, db: dict, cor_period: int, min_cor: int) -> pd.DataFrame:
    ''' Create a df with the close prices of the key symbol and cor
    symbol.  Look for correlation above a certain threshold over a
    few different shift() amounts '''

    if cor_symbol in spreads:
        cor_df = _make_spread(cor_symbol, db)
    else:
        cor_df = db[cor_symbol]
        
    # Ensure matching df lengths (drop the overnight candles if there are any)
    key_df = db[trading_symbol][db[trading_symbol].index.isin(cor_df.index)].copy() # to avoid warnings
    cor_df = cor_df[cor_df.index.isin(key_df.index)]
    key_df = _normalize(key_df)
    cor_df = _normalize(cor_df)
    # looks like its good to here   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # print(len(cor_df.notna()))
    # Iter various shifts to find highest cor and save the data to this dict
    shift_info = {
        'data': pd.DataFrame(dtype=np.float32),
        'best_sum': 0,
        'shift': 0
    }

    for shift_period in SHIFT_PERIODS: 
        cor_values = key_df.rolling(cor_period).corr(cor_df.shift(shift_period))
        cor_values = cor_values.dropna()

        # Update the shift dict
        if abs(cor_values).sum() > shift_info['best_sum']:
            shift_info['data'] = round(cor_values, 3)
            shift_info['best_sum'] = abs(cor_values).sum()
            shift_info['shift'] = shift_period

    # If nothing is found jump to the next symbol
    if len(shift_info['data']) == 0:
        return

    # To eliminate choppiness of the value line I plot later, I want to bring
    # any overnight gaps back in and ffill the nans. I’ll use the original db[trading_symbol] index for this
    overnight = pd.DataFrame(db[trading_symbol])
    overnight['cor'] = shift_info['data']
    overnight['cor_close'] = cor_df
    overnight = overnight.fillna(method='ffill')
    overnight = overnight.dropna()

    # I'll keep the normalized close values so that later on I 
    # can scan the symbols which have the highest cor with the value line
    overnight.cor_close = _normalize(overnight.cor_close)

    # Get list of rows where correlation is above threshold
    cor_rows = overnight.cor[abs(overnight.cor) > min_cor].index
    if not cor_rows.empty:
        cor_df = _save_result_to_df(cor_rows, overnight, cor_symbol, shift_info['shift'])
        # cor_df is a df of normalized close prices of the key_symbol and cor_symbol,
        # as well as the correlation value at any given time and the shift value used.
        return cor_df

def _cor_scanning_thread_executor(trading_symbol: str, cor_symbols: list, db: dict, cor_period: int, min_cor: int) -> pd.DataFrame:
    ''' This gets called within the ProcessPoolExecutor. Each trading_symbol has
    it's own process, and each cor_symbol for that trading_symbol gets its own 
    thread to calculate the correlation. Once all threads finish, the dataframes
    are combined into a single df and gets returned as the future within ProcessPoolExecutor. '''

    # Loop thru the comparison symbols
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [] 
        for cor_symbol in cor_symbols:
            futures.append(executor.submit(_find_correlation, 
                                           trading_symbol=trading_symbol, 
                                           cor_symbol=cor_symbol, 
                                           db=db, 
                                           cor_period=cor_period, 
                                           min_cor=min_cor,
                                           ))

        cor_db = [] 
        for f in concurrent.futures.as_completed(futures):
            try:
                df = f.result()
                if df is not None:
                    cor_db.append(df)
            except Exception:
                print(f'A _cor_scanning_thread_executor() thread inside {trading_symbol} failed. OR came back empty!\n')
                    
    # Compile a single dataframe from the dfs
    df = pd.DataFrame()
    df = df.join(cor_db, how='outer')
    df[trading_symbol] = _normalize(db[trading_symbol])

    # Move trading_symbol to front 
    df = df[df.columns[::-1]]

    print(f'{trading_symbol} done. Found {len(df)} periods of correlation.')

    return [trading_symbol, df]  # return the symbol name with the data

def scan_correlations(trading_symbols: list, cor_symbols: list, hist_candles:int = 51840):
    ''' Find correlation between trading_symbols and cor_symbols
    over a rolling window.  Threading is used to load all ohlc data into memory.
    Then each trading_symbol is given its own process, with threading inside each
    one to handle the requests to the in-mem database. '''

    # Build out a df of all needed data (threaded)
    print('Assembling in-memory database..!')
    db = _make_db(hist_candles)
    print('DB assembled.')
    # print(db)

    # The outer loop will iter over the various outlook periods to emulate MTF value
    # Each corr period will get saved to its own table in the db
    for window in windows:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [] 
            for symbol in trading_symbols:
                futures.append(executor.submit(_cor_scanning_thread_executor, 
                                               trading_symbol=symbol, 
                                               cor_symbols=cor_symbols, 
                                               db=db, 
                                               cor_period=windows[window]['COR_PERIOD'],
                                               min_cor=windows[window]['MIN_COR'],
                                               ))

            # Each future is a list, with the symbol name in position 0 and the data in 1
            cor_of_all_cor_symbols = {} 
            for future in concurrent.futures.as_completed(futures):
                try:
                    data = future.result()
                    if data[1] is not None:
                        cor_of_all_cor_symbols[data[0]] = data[1]  # name and df

                except Exception:
                    print('A process failed... thats all I know')
        # Save
        for name, df in cor_of_all_cor_symbols.items():
            df.to_parquet(pathlib.Path(p, f'{name}_{window}.parquet'), index=True)

        # Verify all data was retrieved
        if len(cor_of_all_cor_symbols) < len(trading_symbols):
            print(' ~~~ data request issues inside the scan_correlations function ~~~ ')
            # print(f" ~~~ missing {set(cor_db.keys()) ^ set(trading_symbols)}") # (a | b) - (a & b), the union of both sets minus the intersection

if __name__ == '__main__':

    s = time.time()
    # _find_correlation('EURUSD', 'GBPUSD', _make_db(), 555, .5)
    
    # 93,600 candles is about 1 year of trading data for M5 candles
    scan_correlations(mt5_symbols['majors'], mt5_symbols['others'] + spreads, hist_candles=51840)
    print('minutes elapsed:', (time.time() - s)/60)





