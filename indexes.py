import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import pathlib
import sqlite3
from ohlc_request import _market_open
from symbols_lists import mt5_symbols
import mplfinance as mpf
from create_db import ohlc_db

OHLC_CON = sqlite3.connect(ohlc_db)
vol_tick = pd.DataFrame()

# Silence the SettingWithCopyWarning in _final_ohlc_cleaning
pd.options.mode.chained_assignment = None


''' 
For some reason the time adjustment is different for a tick request
from mt5 than it is for a candle request.
'''

def _make_10y_bond_indexes(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    ''' Follow the same process as making the currency indexes.
    Eg conver to diffs them cumsum, norm, aggregate diff, resample.
    The function to provide this bond data is within ohlc_request module. '''
   
    # Convert to diffs
    df = df.diff()
    df = df.dropna()
    df = df.apply(lambda x: x.cumsum())
    df = df.apply(lambda x: (x - min(x)) / (max(x) - min(x))) # normalize

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
    final = final.apply(lambda x: x.resample(timeframe).last())

    return final

def _request_ticks_and_resample(pair, period, _from, _to):
    ''' Ticks will be resampled into 1 second bars '''


    ticks = mt5.copy_ticks_range(pair, _from, _to, mt5.COPY_TICKS_ALL)
    df = pd.DataFrame(ticks)

    df = df.rename(columns = {df.columns[0]: 'datetime'})

    df.datetime = pd.to_datetime(df.datetime, unit='s')
    df.datetime = df.datetime - pd.Timedelta('5 hours')
    df = df.set_index(df.datetime, drop=True)       

    # Save volume (count each tick)
    df.volume = 1

    # To avoid spikes due to a widening spread get an average price
    df['price'] = (df.bid + df.ask) / 2 
    df = df[['price', 'volume']]

    open = df.price.resample(period).first()
    high = df.price.resample(period).max()
    low = df.price.resample(period).min()
    close = df.price.resample(period).last()
    volume = df.volume.resample(period).sum()
    
    resampled = pd.DataFrame({'open': open,
                            'high': high,
                            'low': low,
                            'close': close,
                            'volume': volume
                             })
    
    # Check for nan volume rows and set to 0
    resampled.volume.replace(np.nan, 0)

    # Any remaining nans should be forward filled
    resampled = resampled.fillna(method='ffill')

    return resampled

def _normalize(df):
    
    # I want the real volume
    df = df.apply(lambda x: (x - min(x)) / (max(x) - min(x)) if x.name != 'volume' else x)
    
    return df

def _diff(df):
    ''' Rather than deal with the prices, use the diff from one price to the next. '''
    
    vol = df.volume
    df = df.diff()
    df.volume = vol
    df = df.dropna()
    return df

def _cumsum(df):

    df = df.apply(lambda x: x.cumsum() if x.name != 'volume' else x)
    return df

def _invert(df):
    ''' If the currency isn't the base, make its values negative (ie, EURUSD on the USD index)'''

    df.open *= -1
    df.high *= -1
    df.low *= -1
    df.close *= -1
    df = df.dropna()

    return df

def _resample(df, final_period):
    ''' Combine the individual dfs of each pair into one index df and then 
    resample into whatever timeframe is desired. For plotting on MT5, resample to 1min.'''

    # Resample 
    open = df.open.resample(final_period).first()
    high = df.high.resample(final_period).max()
    low = df.low.resample(final_period).min()
    close = df.close.resample(final_period).last()
    volume = df.volume.resample(final_period).sum()
    # voltick = df.voltick.resample(final_period).last()
    
    resampled = pd.DataFrame({'open': open,
                            'high': high,
                            'low': low,
                            'close': close,
                            'volume': volume,
                            # 'voltick': voltick
                             })

    return resampled

def _final_ohlc_cleaning(df, ccy):

    # Set open prices to equal the former close. Then drop nan row.
    df.open = df.close.shift(1)
    df = df[1:]

    # Make sure high is highest and low is lowest 
    df.high = df[['open', 'high', 'low', 'close']].values.max(axis=1)
    df.low = df[['open', 'high', 'low', 'close']].values.min(axis=1)

    # Finally, drop duplicates since it seems resampling creates weekend data
    df = df.drop_duplicates(subset=['open', 'high', 'low', 'close'])

    # This returns a small df similar to describe
    if df.isna().sum().any() > 0:
        print(ccy)
        print('nans found')
        print(df.isna().sum())
        print('dropping...')
    
    df = df.dropna()

    return df

def make_ccy_indexes(pairs, initial_period='1s', final_period='1 min', days=60, _from=None):
    '''  '''

    if not mt5.initialize(login=50341259, server="ICMarkets-Demo",password="ZhPcw6MG"):
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    indexes = {
        'USD': [],
        'EUR': [],
        'GBP': [],
        'JPY': [],
        'AUD': [],
        'NZD': [],
        'CAD': [],
        'CHF': [],
    }
    # Create this timestamp outside of loop so it doesn't change.
    # _from gets passed when a continuous db update is being used and becomes
    # the last timestamp in the db
    _to = datetime.now()

    if _from is None:
        _from = _to - pd.Timedelta(f'{days} day')
    else:
        _from -= pd.Timedelta('3 day')

    for pair in pairs:
        
        df = _request_ticks_and_resample(pair, initial_period, _from, _to)
        
        # Rather than deal with the prices, use the diff from one price to the next
        df = _diff(df)

        inverted = _invert(df.copy())
        inverted = _cumsum(inverted)
        inverted = _normalize(inverted)
        df = _cumsum(df)
        df = _normalize(df)

        # Add the df to its proper dict index
        for ccy in indexes:

            if ccy == pair[:3]:
                if len(indexes[ccy]) == 0:
                    indexes[ccy] = df
                else:
                    indexes[ccy] += df
                continue
            
            # Counter currency
            if ccy == pair[-3:]:
                if len(indexes[ccy]) == 0:
                    indexes[ccy] = inverted
                else:
                    indexes[ccy] += inverted
    
    # Voltick
    # the better way to do this is just to save the 1 second data raw. then run some science.
    # for ccy in indexes:
    #     df = indexes[ccy]
    #     hour_mean = df.volume.rolling(60*60).mean()
    #     rolling_max = df.volume.rolling(60*240).max()

    #     indexes[ccy]['voltick'] = df.close[
    #                                 (df.volume > hour_mean * 3)
    #                                 |
    #                                 (df.volume >= rolling_max)
    #                                 ]

    #     print('number of voltick signals:', len(df[df.voltick.notna()]))

    # Resample into some other timeframe
    for ccy in indexes:

        if final_period != '1s':
            indexes[ccy] = _resample(indexes[ccy], final_period)
    
        indexes[ccy] = _final_ohlc_cleaning(indexes[ccy], ccy)

    return indexes

def _save_data(df, ccy):
       
    # df.to_sql(f'{ccy}', OHLC_CON, if_exists='replace', index=True)
    df.to_sql(f'{ccy}', OHLC_CON, if_exists='append', index=True)
    
    # OHLC_CON.close()
        
def save_index_data_for_mt5(indexes):
    ''' format the data for mt5 and save to csv. '''

    for k in indexes:
        
        df = indexes[k]

        # add the necessary columns
        df['date'] = [d.date() for d in df.index]
        df['time'] = [d.time() for d in df.index]
        df['r_vol'] = 0
        df['spread'] = 0
        
        # reorder
        df = df[['date', 'time', 'open', 'high', 'low', 'close', 'volume', 'r_vol', 'spread']]

        # save to csv
        p = r'C:\Users\ru\AppData\Roaming\MetaQuotes\Terminal\67381DD86A2959850232C0BA725E5966\bases\Custom'
        
        df.to_csv(pathlib.Path(p, f'{k}x.csv'), index=False)

def _read_last_timestamp(tablename, conn):
    ''' Open a db table and get the last timestamp that exists.
    Used to db updates '''

    try:
        df = pd.read_sql(f'''SELECT datetime from {tablename}
                            ORDER BY datetime DESC
                            LIMIT 1''', conn)
        df.datetime = pd.to_datetime(df.datetime)
        timestamp = df.values[0]

        return timestamp
    
    except:
        return None

def continuous_db_update():
    ''' Request ticks, resample into 1 second bars and save to the database '''
    
    ccys = [
        'USD',
        'EUR',
        'GBP',
        'JPY',
        'AUD',
        'NZD',
        'CAD',
        'CHF',
    ]

    timestamps = []
    for ccy in ccys:

        try:

            timestamp = _read_last_timestamp(ccy, OHLC_CON)
            timestamps.append(timestamp)

            # Get the timestamp value
            first = min(timestamps)[0]

        # If this is the first request start with 60 days
        except:
            first = datetime.now() - pd.Timedelta('60 days')

    # Request ticks between 'first' and now
    indexes = make_ccy_indexes(mt5_symbols['majors'], _from=first, final_period='1s')

    # Only append new data
    for last_timestamp, ccy in zip(timestamps, indexes):

        df = indexes[ccy]
        df = df[df.index > last_timestamp[0]]
        _save_data(df, ccy)

# 60 days takes 8.5 mins
if __name__ == '__main__':

    # indexes = make_ccy_indexes(mt5_symbols['majors'], final_period='60 min', days=15)
    while True:
        
        if _market_open:

            second = datetime.now().second
            if second == 0:

                continuous_db_update()  # 3 days takes 30 seconds to update


    usd = indexes['USD']
    eur = indexes['EUR']
    gbp = indexes['GBP']
    cad = indexes['CAD']
    jpy = indexes['JPY']
    aud = indexes['AUD']
    nzd = indexes['NZD']
    chf = indexes['CHF']
    
    import mplfinance as mpf

    mpf.plot(usd,type='candle',show_nontrading=False, volume=True, title='usd')
    mpf.plot(eur,type='candle',show_nontrading=False, volume=True, title='eur')
    mpf.plot(gbp,type='candle',show_nontrading=False, volume=True, title='gbp')
    mpf.plot(cad,type='candle',show_nontrading=False, volume=True, title='cad')
    mpf.plot(jpy,type='candle',show_nontrading=False, volume=True, title='jpy')
    mpf.plot(aud,type='candle',show_nontrading=False, volume=True, title='aud')
    mpf.plot(nzd,type='candle',show_nontrading=False, volume=True, title='nzd')
    mpf.plot(chf,type='candle',show_nontrading=False, volume=True, title='chf')
    

    # df = df.reset_index()  # otherwise plot weekend gaps

    
#     import plotly.graph_objects as go
#     fig = go.Figure(data=[go.Candlestick(x=df.index,
#                                         open=df.open, 
#                                         high=df.high,
#                                         low=df.low, 
#                                         close=df.close,
#     increasing_line_color= 'gray', decreasing_line_color= 'gray'
#     )])

#     # ADD VOLTICK MARKERS
#     fig.add_trace(go.Scatter(
#     x=df.index[df.voltick.notna()],
#     y=df.voltick[df.voltick.notna()],
#     mode="markers",
#     name="voltick",
#     # color='#000000'
#     # text=ratings,
#     # textposition="top center"
#     ))

#     fig.update(layout_xaxis_rangeslider_visible=False)
#     fig.update_layout(
#     # remove this "xaxis" section to show weekend gaps
#     xaxis = dict(  
#                 type="category"),
#     paper_bgcolor="LightSteelBlue",  
# )
#     fig.show()

#     time.sleep(1)
