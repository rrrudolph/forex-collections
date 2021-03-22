import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import pathlib
import sqlite3
from symbols_lists import mt5_symbols, fin_symbols, spreads
from ohlc_request import mt5_ohlc_request
import mplfinance as mpf
from create_db import ohlc_db, correlation_db
from tokens import mt5_login, mt5_pass

'''
I want to have multi TF correlation values but I can't use HTFs like D1
when Im filling a historical db. If I did I'd end up with choppy value
lines since I'd only get a single D1 value per day.  So to emulate real historical 
data Im going to use M5 candles but multiply all applicable values (correlation period,
look backs).  I'll also need to drop the correlation threshold for the longer periods.
'''

# How much historical data you want 
HIST_CANDLES = 11520  # 40 days

# Settings for each time horizon 
settings = {
    # 'LTF': {
    #     'NUM_CANDLES': HIST_CANDLES,  # once the historical data is filled out this should be edited for efficiency
    #     'CORR_PERIOD': 864,  # 3 days
    #     'MIN_CORR': abs(0.6),
    # },
    'MTF': {
        'NUM_CANDLES': HIST_CANDLES,
        'CORR_PERIOD': 1440,  # 5 days
        'MIN_CORR': 0.60,
        'SHIFT': 30,
    },
    'HTF': {
        'NUM_CANDLES': HIST_CANDLES,
        'CORR_PERIOD': 5760,  # 20 days
        'MIN_CORR': 0.60 ,
        'SHIFT': 300,
    },
# '4': {
#     CORR_PERIOD: 4608,  # 16 days
#     MIN_CORR: abs(0.5),
#     },
}
shift_periods = [0, 10, 50, 150, 300, 500, 700]

# This needs to be changed to the ubuntu laptop path
OHLC_CON = sqlite3.connect(ohlc_db)
CORR_CON = sqlite3.connect(correlation_db)

final_df = pd.DataFrame()

def _get_data(symbol):

    if not mt5.initialize(login=mt5_login, server="ICMarkets-Demo",password=mt5_pass):
        print("initialize() failed, error code =", mt5.last_error())
        quit()
            
    # mt5 request
    if symbol in mt5_symbols['others'] or symbol in mt5_symbols['majors']:
        df = mt5_ohlc_request(symbol, mt5.TIMEFRAME_M5, num_candles=HIST_CANDLES)

    elif symbol in fin_symbols:
        df = pd.read_sql(f'SELECT * FROM {symbol}', OHLC_CON, parse_dates=False)
        df = df.set_index(df.datetime, drop=True)
        df.index = pd.to_datetime(df.index)

    return df

def _normalize(df):

    df.close = (df.close - min(df.close)) / (max(df.close) - min(df.close))

    return df


def _make_spread(spread):
    ''' These are all currently coming from MT5 so I will wrap a few
    steps into one function here '''

    spread_df = pd.DataFrame()

    # first parse spread
    symbol_1 = spread.split('_')[0]
    symbol_2 = spread.split('_')[1]
    
    symbol_1 = mt5_ohlc_request(symbol_1, mt5.TIMEFRAME_M1, num_candles=HIST_CANDLES)
    symbol_2 = mt5_ohlc_request(symbol_2, mt5.TIMEFRAME_M1, num_candles=HIST_CANDLES)

    # normalize both close columns
    symbol_1.close = (symbol_1.close - min(symbol_1.close)) / (max(symbol_1.close) - min(symbol_1.close))
    symbol_2.close = (symbol_2.close - min(symbol_2.close)) / (max(symbol_2.close) - min(symbol_2.close))
    
    spread_df['close'] = symbol_1.close - symbol_2.close    

    return spread_df

def _append_result_to_final_df(cor_rows, temp_key_df, cor_df, cor_values, key_symbol, cor_symbol, shift, final_df):
    ''' Save close prices and corr value for any correlated pair found before continuing to the next symbol. '''
    for i in cor_rows:
        if not i in temp_key_df.index:
            print(i, 'not in key_df index')
            continue
        final_df.loc[i, f'*{key_symbol}*'] = temp_key_df.loc[i, 'close']
        final_df.loc[i, f'{cor_symbol}'] = cor_df.loc[i, 'close']
        final_df.loc[i, f'{cor_symbol}_corr'] = round(cor_values.loc[i], 3)
        final_df.loc[i, f'{cor_symbol}_shift'] = shift

def find_correlations():
    ''' Find correlation between FX majors and other symbols and spreads
    over a rolling window.  Where corr breaches the threshold, compare the 
    normalized price differences between key and corr symbols/spreads over a few
    lookback periods of various lengths. Those values get saved ato   '''
 
    # Make a single list of all the symbols used for leading correlation
    cor_symbols = []   
    cor_symbols.extend(mt5_symbols['others'])
    cor_symbols.extend(spreads)
    # cor_symbols.extend(fin_symbols)

    # The outer loop will iter over the various correlation periods to emulate MTF value
    # Each corr period will get saved to its own table in the db
    for tf in settings:
        CORR_PERIOD = settings[tf]['CORR_PERIOD']
        MIN_CORR = settings[tf]['MIN_CORR']
        SHIFT = settings[tf]['SHIFT']

        length = len(mt5_symbols['majors']) + 1
        # Iter the fx majors
        for key_symbol in mt5_symbols['majors']:
            final_df = pd.DataFrame()
            length -= 1
            print(f'{tf}: {length} symbols left')

            # Get the data
            key_df = _get_data(key_symbol)

            # Now iter thru the comparison symbols
            for cor_symbol in cor_symbols:

                if cor_symbol in spreads:
                    cor_df = _make_spread(cor_symbol)

                elif cor_symbol in mt5_symbols['others']:
                    cor_df = _get_data(cor_symbol)

                # Ensure matching df lengths
                temp_key_df = key_df[key_df.index.isin(cor_df.index)].copy() # to avoid warnings
                cor_df = cor_df[cor_df.index.isin(temp_key_df.index)]
                if len(temp_key_df) != len(cor_df):
                    print('mismatched lengths:')
                    print(key_symbol, cor_symbol)
                temp_key_df = _normalize(temp_key_df)
                cor_df = _normalize(cor_df)

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
                    if abs(cor_values).mean() > MIN_CORR:
                        if abs(cor_values).sum() > shift_val['best_sum']:
                            shift_val['data'] = round(cor_values, 3)
                            shift_val['best_sum'] = abs(cor_values).sum()
                            shift_val['shift'] = shift

                # Get list of rows where correlation is above threshold
                cor_rows = shift_val['data'][shift_val['data'] > abs(MIN_CORR)].index.tolist()
                if len(cor_rows) > 0:
                    print(f'best val for {cor_symbol} is shift', {shift_val['shift']})
                    print(f'cor count for {cor_symbol}:',len(cor_rows))
                
                _append_result_to_final_df(cor_rows, temp_key_df, cor_df, shift_val['data'], key_symbol, cor_symbol, shift_val['shift'], final_df)
                
            # Once all the symbols and spreads have been analyzed, save the data
            # before continuing on to the next FX major
            print(f'{key_symbol} final df length:', len(final_df))
            print('shift:', shift_val['shift'] )
            if len (final_df) > 2:
                final_df.to_sql(f'{key_symbol}_{tf}', CORR_CON, if_exists='replace', index=True)
            
    OHLC_CON.close()
    CORR_CON.close()

print('\ntime started:', datetime.now())
s = time.time()
find_correlations()
print('minutes elapsed:', (time.time() - s)/60)
