import pandas as pd
import numpy as np
import time
from datetime import datetime
import MetaTrader5 as mt5
import mplfinance as mpf
import sqlite3
from create_db import econ_db
from ohlc_request import mt5_ohlc_request
from ff_calendar_request import upload_to_gsheets
from tokens import adr_sheet
from symbols_lists import mt5_symbols
from entry_signals import atr


'''
missing data on these

EUR Main Refinancing Rate
USD Federal Funds Rate 
USD Employment Cost Index q/q
'''

econ_con = sqlite3.connect(econ_db)

def test(ccy, resample_timeframe):
    
    ccy = ccy.upper()
    cc1 = ccy[:3]
    cc2 = ccy[-3:]

    # df = pd.read_csv(rf'Desktop\{ccy}x.csv',index_col=0,parse_dates=True)    index files
    df = mt5_ohlc_request(ccy, mt5.TIMEFRAME_M1, 90000)
    df = df.set_index(df.dt, drop=True)


    def add_f_to_ohlc(df, ccy):

        ccy = ccy.upper()

        early_date = df.head(1).index.values[0] - pd.Timedelta('90 days')
        start_date = df.head(1).index.values[0]
        end_date = df.tail(1).index.values[0]
        start_date = pd.to_datetime(start_date)


        ff = rate_weekly_forecasts(ccy, early_date,end_date)
        ff.event_datetime = pd.to_datetime(ff.event_datetime)
        ff = ff.set_index(ff.event_datetime)
        ff.index = pd.to_datetime(ff.index)
        
        # limit f to the length of df
        ff = ff[(ff.event_datetime >= start_date) & (ff.event_datetime <= end_date)]
        
        # make cumsum col and normalize for plotting then drop duplicates
        # ff.forecast = (ff.forecast - min(ff.forecast)) / (max(ff.forecast) - min(ff.forecast))

        # ff.forecast = ff.forecast.fillna(method='ffill')
        # f = ff.drop_duplicates(subset=['event_datetime'], keep='last')
        # f = ff.replace(np.nan, 0)
        if len(ff[ff.forecast.isnull()]) > 0:
            print('\nno data for', )
            print(ff[ff.forecast.isnull()])

        
        print('forecast nans:', ff.forecast.isnull().sum())

        ff = ff.dropna()

        if len(ff[ff.forecast == '']) > 0:
            print('\nblank row..?', )
            print(ff[ff.forecast == ''])

        # print('\nblank row..?', )
        # print(ff.loc['2020-11-24 01:00:00'])

        # Add cumsum and forecast col
        for i in df.index:

            if i in ff.index:
                
                forecast = ff.loc[i, 'forecast'].sum()
                df.loc[i, f'{ccy}_forecast'] = forecast


        df[f'{ccy}_forecast'] = df[f'{ccy}_forecast'].fillna(0)
        df[f'{ccy}_csum'] = df[f'{ccy}_forecast'].cumsum()

        # df[f'{ccy}_csum'] = (df[f'{ccy}_csum'] - min(df[f'{ccy}_csum'])) / (max(df[f'{ccy}_csum']) - min(df[f'{ccy}_csum']))
        return df

    def resample(df, cc1, cc2, resample_timeframe):
        # assert df.forecast == ff.forecast.sum()
    
        # Resample 
        open = df.open.resample(resample_timeframe).first()
        high = df.high.resample(resample_timeframe).max()
        low = df.low.resample(resample_timeframe).min()
        close = df.close.resample(resample_timeframe).last()
        volume = df.volume.resample(resample_timeframe).sum()
        forecast1 = df[f'{cc1}_forecast'].resample(resample_timeframe).sum()
        forecast2 = df[f'{cc2}_forecast'].resample(resample_timeframe).sum()
        csum1 = df[f'{cc1}_csum'].resample(resample_timeframe).last()
        csum2 = df[f'{cc2}_csum'].resample(resample_timeframe).last()

        combined = pd.DataFrame({'open': open,
                                'high': high,
                                'low': low,
                                'close': close,
                                'volume': volume,
                                f'{cc1}_forecast': forecast1,
                                f'{cc2}_forecast': forecast2,
                                f'{cc1}_csum': csum1,
                                f'{cc2}_csum': csum2
                                    })
        
        combined = combined.fillna(method='ffill')

        return combined

    df = add_f_to_ohlc(df, cc1)  
    df = add_f_to_ohlc(df, cc2) 

    print(df)
    df = resample(df, cc1, cc2, resample_timeframe) 
    print(df)


    ap1 = [
    mpf.make_addplot(df[f'{cc1}_forecast'],color='r',panel=2,secondary_y=True),
    mpf.make_addplot(df[f'{cc2}_forecast'],color='b',panel=2,),
    mpf.make_addplot(df[f'{cc1}_csum'],color='r',panel=1,secondary_y=True),
    mpf.make_addplot(df[f'{cc2}_csum'],color='b',panel=1),
    mpf.make_addplot(df[f'{cc1}_forecast']-df[f'{cc2}_forecast'],color='g',panel=0, secondary_y=True),
    ]
    mpf.plot(df, type='candle', volume=False, show_nontrading=False, addplot=ap1,title=f'\n{ccy}',)

# test('usdcad', '15 min')


def get_position_of_price_in_adr():

    symbols = []
    symbols.extend(mt5_symbols['majors'])
    symbols.extend(mt5_symbols['others'])


    combined = pd.DataFrame()
    for symbol in symbols:

        # Get candles and make an atr column
        df = mt5_ohlc_request(symbol, mt5.TIMEFRAME_D1, num_candles=20)
        df['symbol'] = symbol
        atr(df)

        # get the range traveled as a % of adr
        df['diff'] = (df.close - df.open) / df.atr
        # abs otherwise moves will cancel each other out
        df.diff = abs(df.diff) 

        # only save the last value
        df = df.tail(1)

        # Add df to group
        combined = pd.concat([combined, df])


    # Sort by highest range
    combined = combined.sort_values(by=['diff', 'symbol'])

    # I will get RESOURCE EXHAUSTED error if I try to write all the values.
    shortened = pd.concat([combined.head(5), combined.tail(5)])
    shortened = shortened.reset_index(drop=True)
    
    # upload this data to gsheets (first clear the sheet)
    adr_sheet.clear()
    for i in shortened.index:
        adr_sheet.update_cell(i+1, 1, shortened.loc[i, 'symbol'])
        adr_sheet.update_cell(i+1, 2, round(shortened.loc[i, 'diff'], 2))
    # adr_sheet.update_cell(1, 1, shortened['symbol'].values.tolist())
    # adr_sheet.update_cell(1, 2, shortened['diff'].values.tolist())

    # Now get the average for each ccy:                                                                                                                                                                     
    ccys = {'USD': '',
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
        ccys[ccy] = df['diff'].mean()
    
    s = pd.Series(ccys)
    s = s.sort_values(ascending=False)

    # upload to gsheet
    for num, val in enumerate(s):
        ccy = s.index[num]

        adr_sheet.update_cell(num+1, 4,ccy)
        adr_sheet.update_cell(num+1, 5, round(val, 2))


    time.sleep(10*60)
    
while True:
    get_position_of_price_in_adr()