import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime
from ohlc_request import mt5_ohlc_request
from symbols_lists import mt5_symbols
from indexes import make_ccy_indexes
from entry_signals import atr
from tokens import forecast_sheet, ohlc_sheet, adr_sheet
from ff_calendar_request import rate_weekly_forecasts, rate_monthly_outlook


def upload_ohlc(timeframe, days, sheet=ohlc_sheet):

    indexes = make_ccy_indexes(mt5_symbols['majors'], timeframe, initial_period='5s', days=days)

    ohlc = pd.DataFrame()
    for ccy, df in indexes.items():

        # datetime index needs to be made into a text column
        # df['datetime'] = df.index
        # df.datetime = df.datetime.astype(str)

        # norm so things plot better on gsheets
        df.close = (df.close - min(df.close)) / (max(df.close) - min(df.close))

        # and Im only plotting close prices
        ohlc[f'{ccy}'] = df.close

    sheet.clear()
    sheet.update([ohlc.columns.values.tolist()] + ohlc.values.tolist())


def upload_adr(sheet=adr_sheet):

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
    sheet.clear()
    for num, val in enumerate(s):
        
        sheet.update_cell(num+1, 1, s.index[num])
        sheet.update_cell(num+1, 2, round(val, 2))
        

def upload_forecasts(horizon='48 hour', sheet=forecast_sheet):

    ''' If I wanted to actually do a rolling 48 hour normalization from previous events 
    I'd need to set up a for loop within 'rate_w_forecasts' to read the database one week at a time 
    and then append the data to a dataframe as I went.  I'll save that for later and just do a simple
    mean() function for now. '''

    # First get a df of however much data you want to normalize against
    # the horizon doesn't change anything if its friday cuz the cal URL is still same week
    df = rate_weekly_forecasts(datetime.now(), horizon=horizon)
    df.index = df.event_datetime


    # Group events falling within the same hour (for plotting)
    df.index = df.index.floor('H')
    # temp = df.groupby(['ccy'])['forecast'].sum()
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

    sheet.clear()
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
    upload_ohlc('15min', days=2)
    upload_adr()
    upload_forecasts(horizon='120 hour')
    time.sleep(20*60)
