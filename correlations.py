import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import pathlib
import sqlite3
from symbols_lists import mt5_symbols, fin_symbols, spread_symbols
from ohlc_request import mt5_ohlc_request
import mplfinance as mpf
from create_db import ohlc_db, correlation_db

CORR_PERIOD = 20
MIN_CORR = 0.7
RESAMPLE_PERIOD = '10 min'

# these are bar counts. if more look back periods are added
# the coefficient in _set_buy_sell_value needs to be adjusted
LOOKBACK_1 = 3
LOOKBACK_2 = 12
LOOKBACK_3 = 48

# This needs to be changed to the ubuntu laptop path
OHLC_CON = sqlite3.connect(ohlc_db)
CORR_CON = sqlite3.connect(correlation_db)

def _resample_normalize_combine(symbol, df=df, new_df=new_df):

    # clean up any weird symbols
    # 'OANDA:XCU_USD' >>> 'XCUUSD'
    if ':' in symbol:
        symbol = symbol.replace('_', '').split(':')[1]

    # resample to account for gaps in finnhub data
    new_df = new_df.close.resample(RESAMPLE_PERIOD).last()

    # normalize and rename close column
    new_df[f'{symbol}'] = (new_df.close - min(new_df.close)) / (max(new_df.close) - min(new_df.close))
    new_df = new_df[f'{symbol}']

    # Add it to the combined df
    df = new_df if len(df) == 0 else df = df.merge(new_df, left_index=True, right_index=True)

    return df

def _make_spreads(df):
    ''' this will have to be quite manual I think '''

    df['XTIUSD_XAUUSD'] = df.XTIUSD - df.XAUUSD
    df['XTIUSD_XAGUSD'] = df.XTIUSD - df.XAGUSD
    df['XTIUSD_XBRUSD'] = df.XTIUSD - df.XBRUSD
    df['XAGUSD_XAUUSD'] = df.XAGUSD - df.XAUUSD
    df['XPDUSD_XAUUSD'] = df.XPDUSD - df.XAUUSD
    df['XPDUSD_XAGUSD'] = df.XPDUSD - df.XAGUSD
    df['XNGUSD_XAUUSD'] = df.XNGUSD - df.XAUUSD
    df['XCUUSD_XAUUSD'] = df.XCUUSD - df.XAUUSD
    df['XNGUSD_XTIUSD'] = df.XNGUSD - df.XTIUSD
    df['UK100_DE30_US500'] = df.UK100 + df.DE30 - df.US500 * 2
    df['UK100_US500_DE30'] = df.UK100 + df.US500 - df.DE30 * 2
    df['US500_DE30_UK100'] = df.DE30 + df.US500 - df.UK100 * 2
    df['UK100_US30'] = df.UK100 - df.US30
    df['UK100_US500'] = df.UK100 - df.US500
    df['UK100_USTEC'] = df.UK100 - df.USTEC
    df['DE30_US30'] = df.DE30 - df.US30
    df['DE30_US500'] = df.DE30 - df.US500
    df['DE30_USTEC'] = df.DE30 - df.USTEC
    df['US30_US500'] = df.US30 - df.US500
    df['US30_USTEC'] = df.US30 - df.USTEC
    df['US500_USTEC'] = df.US500 - df.USTEC
    df['XLF_XLU'] = df.XLF - df.XLU
    df['VIXM_VIXY'] = df.VIXM - df.VIXY
    df['BAC_REET'] = df.BAC - df.REET
    df['IWF_IWD'] = df.IWF - df.IWD

    # need to add:
    # 10 spreads

    return df

def _make_correlation_dict(df):
    ''' This will only save data for the majors. 
    {'USDCAD': ['OIL', 'CA10Y'] ... etc }'''

    cor = {}
    cols = df.columns.tolist()
    for col in cols:
        if col in mt5_symbols['majors']:
            cor[col] = df[
                (df[f'{col}'] > MIN_CORR) 
                & 
                (df[f'{col}'] < 1)
                ].index.tolist()

    return cor
        
def _get_diff_over_lookbacks(symbol, df=df):
    
        last_row = df.tail(1).index[0]
        x = (3 * df.loc[last_row, f'{symbol}'] - df.loc[last_row - LOOKBACK_1, f'{symbol}']
                                               - df.loc[last_row - LOOKBACK_2, f'{symbol}']
                                               - df.loc[last_row - LOOKBACK_3, f'{symbol}']
        )
        
        return x

def _save_symbol_range(range_dict, symbol=symbol, new_df=new_df):
    ''' save the range from high to low for each symbol so I can
    convert the final normalized buy/sell value back into pips. '''

    high_low = new_df.high.max() - new_df.low.min()
    range_dict[symbol] = high_low

def _set_buy_sell_value(cor_dict, range_dict, df):
    ''' Create a db table for each symbol in the mt5 majors list and save the current 
    correlation buy/sell score to it. For each correlated symbol, the diff over a few 
    look back periods will be calculated.  Those diffs will be added together and the 
    diff of the key symbol will be subtracted from it.  The resulting number will be a 
    normalized value which will be converted back into pips. '''

    # Loop thru the cor dict to calculate the divergence between key symbol and 
    # correlated symbols. At this point only the majors will exist as key_symbol
    for key_symbol, correlated_symbols in cor_dict.items():
        final_df = pd.DataFrame()
        
        key_symbol_diff = _get_diff_over_lookbacks(key_symbol)

        # get the diffs of the correlated symbols
        cor_total_diff = 0
        for cor_symbol in correlated_symbols:

            cor_symbol_diff = _get_diff_over_lookbacks(cor_symbol)
            cor_total_diff += cor_symbol_diff
        

        # put the data into a df to be saved
        final_df['datetime'] = df.datetime.tail(1)

        # get the normalized buy/sell value
        final_df['normd_value'] = cor_total_diff - key_symbol_diff

        # convert that value into pips based on the initial price range that got normalized
        final_df['pip_value'] = (cor_total_diff - key_symbol_diff) * range_dict[key_symbol]

        final_df['correlated_symbols'] = correlated_symbols

        # save
        final_df.to_sql(f'{key_symbol}', CORR_CON, if_exists='append', index=False)


def get_buy_sell_value():
    ''' Based on divergences of correlated markets, find the buy/sell value
    for each of the FX majors (in pips). '''

    # first open all symbols and make a combined df of close prices
    all_symbols = []   
    all_symbols.extend(mt5_symbols['majors'])
    all_symbols.extend(mt5_symbols['others'])
    all_symbols.extend(fin_symbols)

    df = pd.DataFrame()
    range_dict = {}
    for symbol in all_symbols:

        if symbol in fin_symbols:
            new_df = pd.read_sql(f'SELECT * FROM {symbol}', OHLC_CON, parse_dates=False)
            new_df = new_df.set_index(df.datetime, drop=True)
            new_df.index = pd.to_datetime(df.index)

            # Limit the length of the correlation period (1440 mins per day)
            if len(new_df) > CORR_PERIOD * 1440:
                new_df = new_df[:CORR_PERIOD * 1440]
                
        # mt5 request
        if symbol in mt5_symbols['majors'] or symbol in mt5_symbols['others']:
            new_df = mt5_ohlc_request(symbol, '1 min', num_candles=CORR_PERIOD * 1440)
            

        if symbol in mt5_symbols['majors']:
            _save_symbol_range(range_dict)

        df = _resample_normalize_combine(symbol)


    # add spreads to df
    df = _make_spreads(df)

    # run corr
    df = df.drop_duplicates()
    df = df.corr(method ='pearson') 

    # get a dict containing list of correlated symbols {USDCAD: [OIL, CA10Y] ... }
    cor_dict = _make_correlation_dict(df)

    # now reset the index so I can use a look back by bar count
    df = df.reset_index(drop=False)

    # calculate the divergence and save the data
    _set_buy_sell_value(cor_dict, range_dict, df)