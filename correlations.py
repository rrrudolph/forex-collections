import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import pathlib
import sqlite3
import mplfinance as mpf
from symbols_lists import mt5_symbols, fin_symbols, spreads
from ohlc_request import mt5_ohlc_request
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
HIST_CANDLES = 18040  # ___ days

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
    },
    # 'HTF': {
    #     'NUM_CANDLES': HIST_CANDLES,
    #     'CORR_PERIOD': 5760,  # 20 days
    #     'MIN_CORR': 0.60 ,
    # },
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

def _normalize(df, col):
    ''' normalize the specified column within a dataframe '''

    df[f'{col}'] = (df[f'{col}'] - min(df[f'{col}'])) / (max(df[f'{col}']) - min(df[f'{col}']))

    return df


def _make_spread(spread):
    ''' These are all currently coming from MT5 so I will wrap a few
    steps into one function here '''

    spread_df = pd.DataFrame()

    # first parse spread
    symbol_1 = spread.split('_')[0]
    symbol_2 = spread.split('_')[1]

    try: 
        symbol_1 = mt5_ohlc_request(symbol_1, mt5.TIMEFRAME_M5, num_candles=HIST_CANDLES)
        symbol_2 = mt5_ohlc_request(symbol_2, mt5.TIMEFRAME_M5, num_candles=HIST_CANDLES)
    except:
        print(f'failed to get data for {symbol_1} or {symbol_2}')
        quit()
    # normalize both close columns
    symbol_1.close = (symbol_1.close - min(symbol_1.close)) / (max(symbol_1.close) - min(symbol_1.close))
    symbol_2.close = (symbol_2.close - min(symbol_2.close)) / (max(symbol_2.close) - min(symbol_2.close))
    
    spread_df['close'] = symbol_1.close - symbol_2.close    

    return spread_df

def _append_result_to_final_df(cor_rows, overnight, key_symbol, cor_symbol, shift, final_df):
    ''' Save close prices and corr value for any correlated pair found before continuing to the next symbol. '''
    
    for i in cor_rows:
        final_df.loc[i, f'*{key_symbol}*'] = overnight.loc[i, 'close']
        final_df.loc[i, f'{cor_symbol}'] = overnight.loc[i, 'cor_close']
        final_df.loc[i, f'{cor_symbol}_corr'] = round(overnight.loc[i, 'cor'], 3)
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

                # Ensure matching df lengths (drop the overnight overnight if there are any)
                temp_key_df = key_df[key_df.index.isin(cor_df.index)].copy() # to avoid warnings
                cor_df = cor_df[cor_df.index.isin(temp_key_df.index)]

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
                    if abs(cor_values).mean() > MIN_CORR:
                        if abs(cor_values).sum() > shift_val['best_sum']:
                            shift_val['data'] = round(cor_values, 3)
                            shift_val['best_sum'] = abs(cor_values).sum()
                            shift_val['shift'] = shift
                
                # If nothing is found jump to the next symbol
                if len(shift_val['data']) == 0:
                    continue

                # To eliminate choppiness of the value line I plot later, I want to bring
                # the overnight gaps back in and ffill the nans. I’ll use the original key_df index for this
                overnight = key_df.copy()
                overnight['cor'] = shift_val['data']
                overnight['cor_close'] = cor_df.close.shift(shift_val['shift'])
                overnight = overnight.fillna(method='ffill')
                overnight = overnight.dropna() # if it was shifted, it will the shift value worth of nans

                # I'll keep the normalized close values of the key and cor symbols so that 
                # later on I can scan the symbols which have the highest cor with the value line
                overnight = _normalize(overnight, 'close')
                overnight = _normalize(overnight, 'cor_close')

                # Get list of rows where correlation is above threshold
                cor_rows = overnight.cor[abs(overnight.cor) > MIN_CORR * 0.75].index.tolist()
                if len(cor_rows) > 0:
                    print(f'best val for {cor_symbol} is shift', shift_val['shift'])
                    print(f'cor count for {cor_symbol}:',len(cor_rows))
                
                    _append_result_to_final_df(cor_rows, overnight, key_symbol, cor_symbol, shift_val['shift'], final_df)
                
                # print(final_df)
                # quit()
            # Once all the symbols and spreads have been analyzed, save the data
            # before continuing on to the next FX major
            print(f'{key_symbol} final df length:', len(final_df))
            if len (final_df) > 2:
                final_df.to_sql(f'{key_symbol}_{tf}', CORR_CON, if_exists='replace', index=True)
            
    OHLC_CON.close()
    CORR_CON.close()

if __name__ == '__main__':

    print('\ntime started:', datetime.now())
    s = time.time()
    find_correlations()
    print('minutes elapsed:', (time.time() - s)/60)
