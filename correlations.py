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

# This is done with 1 min bars so I multiply by 1440 to get a count in days
HIST_CANDLES = 5 * 1440

# The rolling window to find correlation
CORR_PERIOD = 4 * 1440
MIN_CORR = 0.0025

# these are bar counts. if more look back periods are added
# _diffs_over_lookbacks needs to be adjusted
LOOKBACK_1 = 30
LOOKBACK_2 = 240
LOOKBACK_3 = 1440

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
            df = mt5_ohlc_request(symbol, mt5.TIMEFRAME_M1, num_candles=HIST_CANDLES)

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

    # first open all symbols and make a merged df of close prices
    cor_symbols = []   
    cor_symbols.extend(mt5_symbols['others'])
    # cor_symbols.extend(fin_symbols)

    for key_symbol in mt5_symbols['majors']:
        final_df = pd.DataFrame()

        # Get the data
        key_df = _get_data(key_symbol)

        # Now iter thru the comparison symbols
        # for cor_symbol in cor_symbols:
            # # print(key_symbol, cor_symbol)
        
            # cor_df = _get_data(cor_symbol)

            # # Ensure matching dfs
            # temp_key_df = key_df[key_df.index.isin(cor_df.index)].copy() # to avoid annnoooyyying warnings
            # cor_df = cor_df[cor_df.index.isin(temp_key_df.index)]

            # temp_key_df = _normalize(temp_key_df)
            # cor_df = _normalize(cor_df)

            # # Now that I've aligned via datetimes I want to switch out the 
            # # datetime index for a regular numeric range
            # temp_key_df = temp_key_df.reset_index(drop=False)
            # cor_df = cor_df.reset_index(drop=False)

            # # Get the list of correlation values
            # cor_values = temp_key_df.close.rolling(CORR_PERIOD).corr(cor_df.close)
            # # print('\n cor_values:',cor_values)

            # # Get list of rows where correlation is above threshold
            # cor_rows = cor_values[cor_values > abs(MIN_CORR)].index.tolist()
            # print('\n cor count:',len(cor_rows))


            # # Compare the normalized price differences between the key
            # # and cor symbols.  Im saving the initial long term correlation
            # # value to use later as a coefficient in deriving a buy/sell rating.
            # # The new short term corr value I don't have plans for but saving anyway.
            # for row in cor_rows:

            #     if row < LOOKBACK_3:
            #         continue

            #     key_diffs = _diffs_over_lookbacks(temp_key_df, row)
            #     cor_diffs = _diffs_over_lookbacks(cor_df, row)

            #     # Get the new correlation values
            #     shortterm_cor = key_diffs.corr(cor_diffs)
            #     longterm_cor = cor_values.loc[row]
                
            #     # Save the sum of the diffs and correlation values
                # final_df.loc[row, 'datetime'] = temp_key_df.loc[row, 'datetime']
                # final_df.loc[row, f'*{key_symbol}*'] = key_diffs.sum()
                # final_df.loc[row, f'{cor_symbol}'] = cor_diffs.sum()
                # final_df.loc[row, f'{cor_symbol}_corr'] = longterm_cor

        # now do all the same things but with the spreads
        for spread in spreads:

            cor_df = _make_spread(spread)
            temp_key_df = key_df[key_df.index.isin(cor_df.index)].copy()
            cor_df = cor_df[cor_df.index.isin(temp_key_df.index)]
            temp_key_df = temp_key_df.reset_index(drop=False)
            cor_df = cor_df.reset_index(drop=False)
            cor_values = temp_key_df.close.rolling(CORR_PERIOD).corr(cor_df.close)
            cor_rows = cor_values[cor_values > abs(MIN_CORR)].index.tolist()
            # print(len(cor_rows))
            for row in cor_rows:

                if row < LOOKBACK_3:
                    continue

                key_diffs = _diffs_over_lookbacks(temp_key_df, row)
                cor_diffs = _diffs_over_lookbacks(cor_df, row)

                shortterm_cor = key_diffs.corr(cor_diffs)
                longterm_cor = cor_values.loc[row]

                # Save the sum of the diffs and correlation values
                final_df.loc[row, 'datetime'] = temp_key_df.loc[row, 'datetime']
                final_df.loc[row, f'*{key_symbol}*'] = key_diffs.sum()
                final_df.loc[row, f'{spread}'] = cor_diffs.sum()
                final_df.loc[row, f'{spread}_corr'] = longterm_cor
                

        # Once all the symbols and spreads have been analyzed, save the data
        # before continuing on to the next FX major symbol
        final_df.to_sql(f'{key_symbol}', CORR_CON, if_exists='replace', index=False)

s = time.time()
find_correlations()
print('minutes elapsed:', time.time() - s)