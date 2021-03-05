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
from create_db import ohlc_db


ohlc_con = sqlite3.connect(ohlc_db)

def _add_new_df_to_df(symbol, df=df, new_df=new_df):
    
    # normalize and rename column
    new_df[f'{symbol}'] = (new_df.close - min(new_df.close)) / (max(new_df.close) - min(new_df.close))
    new_df = new_df[f'{symbol}']

    # Add it to the combined df
    df = new_df if len(df) == 0 else df = df.merge(new_df, left_index=True, right_index=True)

def _make_spreads(df):
    ''' this will have to be quite manual I think '''

    df['XTIUSD_XAUUSD'] = df.XTIUSD - df.XAUUSD
    df['XTIUSD_XAGUSD'] = df.XTIUSD - df.XAGUSD
    df['XTIUSD_XBRUSD'] = df.XTIUSD - df.XBRUSD
    df['XAGUSD_XAUUSD'] = df.XAGUSD - df.XAUUSD
    df['XPDUSD_XAUUSD'] = df.XPDUSD - df.XAUUSD
    df['XPDUSD_XAGUSD'] = df.XPDUSD - df.XAGUSD
    df['XNGUSD_XAUUSD'] = df.XNGUSD - df.XAUUSD
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
    df['XLF_XLU'] = df.XLF - df.XLU,
    df['VIXM_VIXY'] = df.VIXM - df.VIXY,
    df['BAC_REET'] = df.BAC - df.REET,
    df['IWF_IWD'] = df.IWF - df.IWD,


def find_correlation(corr_period=20, min_corr=0.7):
    ''' Return a list of pd.Series containing close prices '''

    # first open all symbols used for spreads and make a combined df of close prices
    all_symbols =  []   
    all_symbols.extend(mt5_symbols['majors'])
    all_symbols.extend(mt5_symbols['others'])
    all_symbols.extend(fin_symbols)

    df = pd.DataFrame()
    for symbol in all_symbols:

        if symbol in fin_symbols:
            new_df = pd.read_sql(f'SELECT * FROM {symbol}', ohlc_con, parse_dates=True)

            # Limit the length of the correlation period (1440 mins per day)
            if len(new_df) > corr_period*1440:
                new_df = new_df[:corr_period*1440]
                
            _add_new_df_to_df(symbol)


        # mt5 request
        if symbol in mt5_symbols['majors'] or symbol in mt5_symbols['others']:
            new_df = mt5_ohlc_request(pair, '1 min', num_candles=corr_period*1440)
            
            _add_new_df_to_df(symbol)

    # run corr
    df = df.drop_duplicates()
    df = df.corr(method ='pearson') 

    # save ie {'USDCAD': ['OIL', 'CA10Y']}
    cor = {}
    cols = df.columns.tolist()
    for col in cols:
        cor[col] = df[(df[f'{col}'] > min_corr) & (df[f'{col}'] < 1)].index.tolist()
        

spreads = [
]