import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import sqlite3
from datetime import datetime
from ohlc_request import mt5_ohlc_request
from symbols_lists import mt5_symbols
from indexes import make_ccy_indexes
from entry_signals import atr
from tokens import forecast_sheet, ohlc_sheet, bonds_sheet
from ff_calendar_request import rate_weekly_forecasts, rate_monthly_outlook
from create_db import ohlc_db


OHLC_CON = sqlite3.connect(ohlc_db)


def _read_bonds(timeframe, sheet=bonds_sheet):
    ''' get the bond data and normalize for plotting with ohlc index data '''
    
    df = pd.DataFrame(sheet.get_all_values())

    # Scrub n Clean
    df = df.set_index(df.iloc[:, 0])
    df.columns = df.iloc[0]
    df = df.iloc[1:, 1:]
    df.index = pd.to_datetime(df.index)
    df = df.replace('', np.nan)
    df = df.fillna(method='ffill')
    df = df.astype(float)

    # Only interested in 10s
    df = df.iloc[:, 1::2]

    # Convert to diffs
    df = df.diff()
    df = df.dropna()

    # cumsum
    df = df.apply(lambda x: x.cumsum())

    # norm
    df = df.apply(lambda x: (x - min(x)) / (max(x) - min(x)))

    # create the "counter pair" for each symbol. ie USD10Y - DE - GB - JP....
    # so iter thru list of columns and whichever one im on, get sum of every
    # column except the current one. once that's done, I will re-normalize
    # and then create the actual spread. the spread will just be the current
    # iterable minus that aggregate sum that was normalized.
    agg_sum = df.apply(lambda x: x.sum(), axis=1)

    symbols = {}
    for name in df.columns.tolist():
        x = agg_sum - df[f'{name}']
        x = (x - min(x)) / (max(x) - min(x))  # no need to average
        x = df[f'{name}'] - x
        symbols[name] = x

    # damn Im getting good!

    final = pd.DataFrame(symbols)
    final.index = df.index

    # resample 
    final = final.apply(lambda x: x.resample(timeframe).last())

    return final


def _read_indexes(timeframe, days, sheet=ohlc_sheet):
    ''' Get the index data from the data base to plot on ghseets '''

    ccys = ['']    
    # Read from the index database and then resample
    df = pd.read_sql(f'''SELECT * from {ccy}
                    ORDER BY datetime DESC
                    LIMIT 1''', OHLC_CON)
    
    ohlc = pd.DataFrame()
    for ccy, df in indexes.items():
        ohlc[f'{ccy}'] = df.close

    # now merge with bond data
    bonds = _upload_bonds(timeframe)
    ohlc = ohlc.merge(bonds, left_index=True, right_index=True)

    # Norm the values for just whatevers gonna be charted
    ohlc = ohlc.dropna()
    ohlc = ohlc.apply(lambda x: (x - min(x)) / (max(x) - min(x)))


    # Make a dt column for charting and move it to front
    ohlc['datetime'] = ohlc.index
    ohlc = ohlc[ ['datetime'] + [ col for col in ohlc.columns if col != 'datetime' ] ]
    ohlc.index = ohlc.index.astype(str)
    ohlc.datetime = ohlc.datetime.astype(str)
    ohlc = ohlc.fillna(method='ffill')
    ohlc = ohlc.drop_duplicates(subset=['USD', 'EUR', 'GBP'])

    sheet.clear()
    sheet.update([ohlc.columns.values.tolist()] + ohlc.values.tolist())


def _upload_adr(sheet=forecast_sheet):

    symbols = []
    symbols.extend(mt5_symbols['majors'])
    symbols.extend(mt5_symbols['others'])

    combined = pd.DataFrame()
    for symbol in symbols:

        # Get candles and make an atr column
        df = mt5_ohlc_request(symbol, mt5.TIMEFRAME_D1, num_candles=15)
        df['symbol'] = symbol
        atr(df)

        # get the range traveled as a % of adr
        df['range'] = (df.close - df.open) / df.atr
        
        # abs otherwise moves will cancel each other out
        df.range = abs(df.range) 

        # only save the last value
        df = df.tail(1)

        # Add df to group
        combined = pd.concat([combined, df])

    # Sort by highest range
    combined = combined.sort_values(by=['range', 'symbol'])

    # Now get the average for each ccy:                                                                                                                                                                     
    ccys = {
        'USD': '',
        'EUR': '',
        'GBP': '',
        'JPY': '',
        'AUD': '',
        'NZD': '',
        'CAD': '',
        'CHF': '',
    }

    for ccy in ccys:
        df = combined[combined.symbol.str.contains(ccy)].copy()
        ccys[ccy] = df['range'].mean()
    
    s = pd.Series(ccys)
    s = s.sort_values(ascending=False)

    # upload to gsheet
    sheet.update_cell(11, 7, r'% of ADR traveled')
    for num, val in enumerate(s):
        sheet.update_cell(num+12, 7, s.index[num])
        sheet.update_cell(num+12, 8, round(val, 2))
        



def upload_forecasts(horizon='48 hour', sheet=forecast_sheet):

    ''' If I wanted to actually do a rolling 48 hour normalization from previous events 
    I'd need to set up a for loop within 'rate_w_forecasts' to read the database one week at a time 
    and then append the data to a dataframe as I went.  I'll save that for later and just do a simple
    mean() function for now. '''

    # putting this here since it goes to the same sheet
    forecast_sheet.clear()
    _upload_adr()

    # First get a df of however much data you want to normalize against
    # the horizon doesn't change anything if its friday cuz the cal URL is still same week
    df = rate_weekly_forecasts(datetime.now(), horizon=horizon, save_to_db=True)
    df.index = df.event_datetime


    # Group events falling within the same hour (for plotting)
    df.index = df.index.floor('H')
    
    # groupby confirmed working https://prnt.sc/10857pl
    temp = df.groupby([df.index, 'ccy'])['forecast'].sum()


    # ghseets can't deserialize a datetime 
    resampled = pd.DataFrame(temp)
    resampled = resampled.reset_index()
    resampled.event_datetime = resampled.event_datetime.astype(str)

    # Get the mean forecast for each currency in the upcoming data
    mean_forecast = {}
    for ccy in df.ccy.unique():
        value = df.forecast[df.ccy == ccy].mean()
        mean_forecast[ccy] = value

    # Now grab the monthly data as well to plot those values
    month = rate_monthly_outlook()
    month = month.replace(np.nan, '')
    month = month.sort_values(by=['monthly'], ascending=False).reset_index(drop=True)

    sheet.update([resampled.columns.values.tolist()] + resampled.values.tolist())

    # Now I have to upload te rest manually
    # mean forecasts
    sheet.update_cell(1, 5,'ccy')
    sheet.update_cell(1, 6, 'mean_forecast')
    for num, ccy in enumerate(mean_forecast):
        sheet.update_cell(num+2, 5,ccy)
        sheet.update_cell(num+2, 6, round(mean_forecast[ccy], 2))

    # monthly values
    sheet.update_cell(1, 7,'ccy')
    sheet.update_cell(1, 8, 'monthly')
    for i in month.index:
        sheet.update_cell(i+2, 7, month.loc[i, 'ccy'])
        sheet.update_cell(i+2, 8, month.loc[i, 'monthly'])



while True:


    upload_ohlc('15min', days=5)
    time.sleep(60)
    upload_forecasts(horizon='48 hour')
    time.sleep(30*60)
