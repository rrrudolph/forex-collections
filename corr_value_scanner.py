import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import pathlib
import sqlite3
import mplfinance as mpf
from symbols_lists import mt5_symbols, fin_symbols, spreads
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
# 51840  # 160 days
# HIST_CANDLES = 51840  # 160 days
HIST_CANDLES = 1840  

# Settings for each time horizon 
settings = {
    # 'LTF': {
    #     'CORR_PERIOD': 1440,  # 5 days
    #     'MIN_CORR': 0.7,
    # },
    'MTF': {
        'CORR_PERIOD': 5760,  # 20 days
        'MIN_CORR': 0.65,
    },
    'HTF': {
        'CORR_PERIOD': 17280,  # 60 days
        'MIN_CORR': 0.60,
    },
}

shift_periods = [0, 10, 50, 150, 300, 500, 750]

# This needs to be changed to the ubuntu laptop path
OHLC_CON = sqlite3.connect(ohlc_db)
CORR_CON = sqlite3.connect(r'C:\Users\ru\forex\db\correlation.db')
 
# Make a single list of all the symbols used for leading correlation
cor_symbols = mt5_symbols['others'] + spreads   
# cor_symbols.extend(fin_symbols)

def _get_data(key_symbol, symbol, tf, corr_period, hist_fill=True, spread=None):

    # If this is a historical overwrite
    if hist_fill == True:
        candle_count = HIST_CANDLES

    else:
        # First figure out how much data is needed by getting the last correlation timestamp
        try:
            start = _read_last_datetime(f'{key_symbol}_{tf}', CORR_CON)
            minutes_diff = (pd.Timestamp.now() - start).total_seconds() / 60.0
            
            # How many 5 minute periods exist in that range
            candle_count = minutes_diff // 5
            candle_count = corr_period + int(candle_count)
        except Exception:
            # If that table didn't exists in the db do a hisorical fill
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

    return df

def _normalize(df:pd.DataFrame, *columns:str) -> pd.DataFrame:
    ''' normalize the specified columns within a dataframe '''
    
    for col in columns:
        df[col] = (df[col] - min(df[col])) / (max(df[col]) - min(df[col]))

    return df

def _make_spread(symbol, tf, corr_period, hist_fill, spread=None):
    ''' These are all currently coming from MT5 so I will wrap a few
    steps into one function here: get data, normalize, combine.'''

    # first parse spread
    symbol_1 = spread.split('_')[0]
    symbol_2 = spread.split('_')[1]

    try: 
        symbol_1 = _get_data(symbol, tf, corr_period, hist_fill, spread=symbol_1)
        symbol_2 = _get_data(symbol, tf, corr_period, hist_fill, spread=symbol_2)
    except:
        print(f'failed to get data for {symbol_1} or {symbol_2}')
        quit()
    # normalize both close columns
    symbol_1.close = (symbol_1.close - min(symbol_1.close)) / (max(symbol_1.close) - min(symbol_1.close))
    symbol_2.close = (symbol_2.close - min(symbol_2.close)) / (max(symbol_2.close) - min(symbol_2.close))
    
    spread_df = pd.DataFrame()
    spread_df['close'] = symbol_1.close - symbol_2.close    

    return spread_df

def _append_result_to_final_df(cor_rows, overnight, key_symbol, cor_symbol, shift, final_df):
    ''' Save close prices and corr value for any correlated pair found before continuing to the next symbol. '''

    final_df.loc[cor_rows, f'*{key_symbol}*'] = overnight.loc[cor_rows, 'close']
    final_df.loc[cor_rows, f'{cor_symbol}'] = overnight.loc[cor_rows, 'cor_close']
    final_df.loc[cor_rows, f'{cor_symbol}_corr'] = round(overnight.loc[cor_rows, 'cor'], 3)
    final_df.loc[cor_rows, f'{cor_symbol}_shift'] = shift

    return final_df

def _save_data(key_symbol, tf, final_df):
    ''' Read db table into memory, append new data to it and save. Otherwise 
    an existing db table wouldn't accept any new correlation symbol columns '''

    # for stocks
    try:
        existing = pd.read_sql(f'SELECT * FROM {key_symbol}_{tf}', CORR_CON)
        final = existing.append(final_df).drop_duplicates(subset='datetime')
        final.to_sql(f'{key_symbol}_{tf}', CORR_CON, if_exists='replace', index=True)
    except Exception:
        final_df.to_sql(f'{key_symbol}_{tf}', CORR_CON, if_exists='replace', index=True)

def find_correlations(historical_fill=False):
    ''' Find correlation between FX majors and other symbols/spreads
    over a rolling window. Setting historical_fill as True will overwrite 
    existing data, False will append '''


    # The outer loop will iter over the various correlation periods to emulate MTF value
    # Each corr period will get saved to its own table in the db
    for tf in settings:
        CORR_PERIOD = settings[tf]['CORR_PERIOD']
        MIN_CORR = settings[tf]['MIN_CORR']

        # Iter the fx majors
        length = len(mt5_symbols['majors']) + 1
        for key_symbol in mt5_symbols['majors']:
            length -= 1
            print(f'{tf}: {length} symbols left')

            # Get the data
            key_df = _get_data(key_symbol, key_symbol, tf, CORR_PERIOD, hist_fill=historical_fill)

            # Now iter thru the comparison symbols, saving corr  ata to final_df
            final_df = pd.DataFrame(index=key_df.index)
            for cor_symbol in cor_symbols:

                # if key_symbol == cor_symbol:
                #     continue

                if cor_symbol in spreads:
                    cor_df = _make_spread(key_symbol, tf, CORR_PERIOD, hist_fill=historical_fill, spread=cor_symbol)

                elif cor_symbol in mt5_symbols['others']:
                    cor_df = _get_data(key_symbol, cor_symbol, tf, CORR_PERIOD, hist_fill=historical_fill)

                # Ensure matching df lengths (drop the overnight periods if there are any)
                temp_key_df = key_df[key_df.index.isin(cor_df.index)].copy() # to avoid warnings
                cor_df = cor_df[cor_df.index.isin(temp_key_df.index)].copy()

                temp_key_df = _normalize(temp_key_df, 'close')
                cor_df = _normalize(cor_df, 'close')

                # Iter various shifts to find highest cor and save the data to this dict
                shift_val = {
                    'data': pd.DataFrame(dtype=np.float32),
                    'best_sum': 0,
                    'shift': 0
                }
                for shift in shift_periods:
                    cor_values = temp_key_df.close.rolling(CORR_PERIOD).corr(cor_df.close.shift(shift))
                    cor_values = cor_values.dropna()

                    # Update the shift dict
                    if abs(cor_values).sum() > shift_val['best_sum']:
                        shift_val['best_sum'] = abs(cor_values).sum()
                        shift_val['data'] = round(cor_values, 3)
                        shift_val['shift'] = shift
                
                # If nothing is found jump to the next symbol (this would be a good place to add a min length filter)
                if len(shift_val['data']) == 0:
                    continue

                # To eliminate choppiness of the value line I plot later, I want to bring
                # the overnight gaps back in and ffill the nans. I’ll use the original key_df index for this
                overnight = key_df.copy()
                overnight['cor'] = shift_val['data']
                overnight['cor_close'] = cor_df.close
                overnight = overnight.fillna(method='ffill')
                overnight = overnight.dropna() # nans if shifted

                # I'll keep the normalized close values of the key and cor symbols so that 
                # later on I can scan the symbols which have the highest cor with the value line
                overnight = _normalize(overnight, 'close', 'cor_close')

                # Get list of rows where correlation is above threshold
                cor_rows = overnight[(abs(overnight.cor) > MIN_CORR)
                                     &
                                     (abs(overnight.cor) < .99) # in case it tracks itself
                                    ].index
                if len(cor_rows) > 0:
                    # print(f'cor count for {cor_symbol}:',len(cor_rows),'... shift:', shift_val['shift'])
                    final_df = _append_result_to_final_df(cor_rows, overnight, key_symbol, cor_symbol, shift_val['shift'], final_df)


            # Once all the symbols and spreads have been analyzed, save the data
            # before continuing on to the next FX major
            if final_df.empty:
                continue  
            
            final_df = final_df.dropna(subset=[f'*{key_symbol}*'])
            # print(f'{key_symbol} final df length:', len(final_df))
            _save_data(key_symbol, tf, final_df)

    # OHLC_CON.close()
    # CORR_CON.close()

if __name__ == '__main__':
    find_correlations(historical_fill=False)
    
    # while True:
    #     find_correlations(historical_fill=False)
    #     time.sleep(60 * 30) 
