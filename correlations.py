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
I will break the correlations up into 4 groups to capture a period of 15 mins, 2 hours,
1 day, and 1 week.  Roughly at least.
'''

# How much historical data you want (days * mins/5)
HIST_CANDLES = 4660

# Settings for each time horizon 
settings = {
'LTF': {
    'NUM_CANDLES': HIST_CANDLES,  # once the historical data is filled out this should be edited for efficiency
    'CORR_PERIOD': 576,  # 2 days
    'MIN_CORR': abs(0.7),
    'LOOKBACK_1': 1,  # 5 min
    'LOOKBACK_2': 3,  # 15 min
    'LOOKBACK_3': 6,  # 30 min
    },
'MTF': {
    'NUM_CANDLES': HIST_CANDLES,
    'CORR_PERIOD': 1152,  # 4 days
    'MIN_CORR': abs(0.65),
    'LOOKBACK_1': 12,  # 1 hour
    'LOOKBACK_2': 72,  # 6 hours
    'LOOKBACK_3': 144, # 12 hours
    },
'HTF': {
    'NUM_CANDLES': HIST_CANDLES,
    'CORR_PERIOD': 2304,  # 8 days
    'MIN_CORR': abs(0.6),
    'LOOKBACK_1': 288,   # 1 day
    'LOOKBACK_2': 720,   # 2.5 days
    'LOOKBACK_3': 1440,  # 5 days
        },
# '4': {
#     CORR_PERIOD: 4608,  # 16 days
#     MIN_CORR: abs(0.5),
#     LOOKBACK_1: 864,   # 3 days 
#     LOOKBACK_2: 1152,  # 4 days
#     LOOKBACK_3: 1440,  # 5 days
#     },
}

# This needs to be changed to the ubuntu laptop path
OHLC_CON = sqlite3.connect(ohlc_db)
CORR_CON = sqlite3.connect(correlation_db)

def find_correlations():
    ''' Find correlation between FX majors and other symbols and spreads
    over a rolling window.  Where corr breaches the threshold, compare the 
    normalized price differences between key and corr symbols/spreads over a few
    lookback periods of various lengths. Those values get saved ato   '''

 
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

    def _diffs_over_lookbacks(df, row):
        ''' Return a Series of diff values '''

        s = pd.Series(dtype=int)

        s.loc[0] = df.loc[row, 'close'] - df.loc[row - LOOKBACK_1, 'close']
        s.loc[1] = df.loc[row, 'close'] - df.loc[row - LOOKBACK_2, 'close']
        s.loc[2] = df.loc[row, 'close'] - df.loc[row - LOOKBACK_3, 'close']

        return s

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

    
    # Make a single list of all the symbols used for leading correlation
    cor_symbols = []   
    cor_symbols.extend(mt5_symbols['others'])
    # cor_symbols.extend(fin_symbols)

    # The outer loop will iter over the various correlation periods to emulate MTF value
    # Each corr period will get saved to its own table in the db
    for tf in settings:
        print(settings[tf])
        CORR_PERIOD = settings[tf]['CORR_PERIOD']
        MIN_CORR = settings[tf]['MIN_CORR']
        LOOKBACK_1 = settings[tf]['LOOKBACK_1']
        LOOKBACK_2 = settings[tf]['LOOKBACK_2']
        LOOKBACK_3 = settings[tf]['LOOKBACK_3']

        for key_symbol in mt5_symbols['majors']:
            final_df = pd.DataFrame()

            # Get the data
            key_df = _get_data(key_symbol)

            # Now iter thru the comparison symbols
            for cor_symbol in cor_symbols:
                # print(key_symbol, cor_symbol)
            
                cor_df = _get_data(cor_symbol)

                # Ensure matching dfs
                temp_key_df = key_df[key_df.index.isin(cor_df.index)].copy() # to avoid annnoooyyying warnings
                cor_df = cor_df[cor_df.index.isin(temp_key_df.index)]

                temp_key_df = _normalize(temp_key_df)
                cor_df = _normalize(cor_df)

                # Now that I've aligned via datetimes I want to switch out the 
                # datetime index for a regular numeric range
                temp_key_df = temp_key_df.reset_index(drop=False)
                cor_df = cor_df.reset_index(drop=False)

                # I only want to find symbols that lead price so I will use a shift
                # (the the key_symbol into the future so its lagging)
                cor_values = temp_key_df.close.shift(-7).rolling(CORR_PERIOD).corr(cor_df.close)

                # Check for randomness by shifting the other way and comparing the 2 shifts.
                # If the shift back is not greater then this symbol should be skipped
                cor_values_lagging = temp_key_df.close.shift(7).rolling(CORR_PERIOD).corr(cor_df.close)
                # print('\n')
                # print('key',key_symbol)
                # print('cor',cor_symbol)
                # print('abs diff cor - cor_lagging:', abs(cor_values.sum()) - abs(cor_values_lagging).sum())

                # Get list of rows where correlation is above threshold
                cor_rows = cor_values[cor_values > abs(MIN_CORR)].index.tolist()
                print('cor count:',len(cor_rows))


                # # Compare the normalized price differences between the key
                # # and cor symbols.  Im saving the initial correlation
                # # value to use later as a coefficient in deriving a buy/sell rating.
                for row in cor_rows:

                    if row < LOOKBACK_3:
                        continue

                    key_diffs = _diffs_over_lookbacks(temp_key_df, row)
                    cor_diffs = _diffs_over_lookbacks(cor_df, row)

                    # Get the new correlation values
                    shortterm_cor = key_diffs.corr(cor_diffs)
                    longterm_cor = cor_values.loc[row]
                    
                    # Save the sum of the diffs and correlation values
                    final_df.loc[row, 'datetime'] = temp_key_df.loc[row, 'datetime']
                    final_df.loc[row, f'*{key_symbol}*'] = round(key_diffs.sum(), 3)
                    final_df.loc[row, f'{cor_symbol}'] = round(cor_diffs.sum(), 3)
                    final_df.loc[row, f'{cor_symbol}_corr'] = round(longterm_cor, 3)

            # now do all the same things but with the spreads
            for spread in spreads:

                cor_df = _make_spread(spread)
                temp_key_df = key_df[key_df.index.isin(cor_df.index)].copy()
                cor_df = cor_df[cor_df.index.isin(temp_key_df.index)]
                temp_key_df = temp_key_df.reset_index(drop=False)
                cor_df = cor_df.reset_index(drop=False)

                cor_values = temp_key_df.close.shift(-7).rolling(CORR_PERIOD).corr(cor_df.close)
                cor_values_lagging = temp_key_df.close.shift(7).rolling(CORR_PERIOD).corr(cor_df.close)
                # print('\n cor_values:', cor_values)
                # print('cor_values_lagging:', cor_values_lagging)
                # print('abs diff:', abs(cor_values) - abs(cor_values_lagging))
                cor_rows = cor_values[cor_values > abs(MIN_CORR)].index.tolist()
                # print('\n cor count:',len(cor_rows))

                for row in cor_rows:

                    if row < LOOKBACK_3:
                        continue

                    key_diffs = _diffs_over_lookbacks(temp_key_df, row)
                    cor_diffs = _diffs_over_lookbacks(cor_df, row)
                    longterm_cor = cor_values.loc[row]

                    # Save the sum of the diffs and correlation values
                    final_df.loc[row, 'datetime'] = temp_key_df.loc[row, 'datetime']
                    final_df.loc[row, f'*{key_symbol}*'] = round(key_diffs.sum(), 3)
                    final_df.loc[row, f'{spread}'] = round(cor_diffs.sum(), 3)
                    final_df.loc[row, f'{spread}_corr'] = round(longterm_cor, 3)
                    

            # Once all the symbols and spreads have been analyzed, save the data
            # before continuing on to the next FX major symbol
            final_df.to_sql(f'{key_symbol}_{tf}', CORR_CON, if_exists='replace', index=False)

s = time.time()
find_correlations()
print('minutes elapsed:', time.time() - s)
