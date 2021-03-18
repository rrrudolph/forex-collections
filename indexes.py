import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import pathlib
import sqlite3
from ohlc_request import mt5_ohlc_request
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
        print(df.isna())

    return df

def make_ccy_indexes(pairs, initial_period='1s', final_period='1 min', days=60):
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
    # Create this timestamp outside of loop so it doesn't change
    _to = datetime.now()
    _from = _to - pd.Timedelta(f'{days} day')

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

    # Resample into 1 min
    for ccy in indexes:
        indexes[ccy] = _resample(indexes[ccy], final_period)
    
        indexes[ccy] = _final_ohlc_cleaning(indexes[ccy], ccy)

    return indexes

def save_data(indexes):
    ''' Try to append the data from the last row onward.  Otherwise save everything. '''

    # try:
    #     db = pd.read_sql("""SELECT * FROM USD 
    #                         ORDER BY datetime DESC 
    #                         LIMIT 1""", OHLC_CON, parse_dates=True)
    #     db.index = pd.to_datetime(db.index)

    #     for ccy in indexes:
    #         df = indexes[ccy]
    #         df = df[df.index > db.index[0]]
    #         df.to_sql(f'{ccy}', OHLC_CON, if_exists='append', index=True)
    
    # except:
       
    for ccy in indexes:
        df = indexes[ccy]
        df.to_sql(f'{ccy}', OHLC_CON, if_exists='replace', index=True)
        # df.to_sql(f'{ccy}', OHLC_CON, if_exists='append', index=True)
    
    OHLC_CON.close()
        
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


# it took 9 min 33 sec to save 60 days of data (~5mil rows of tick data requested 28 times)
# 10 days will give you 793029 rows of 1s data
if __name__ == '__main__':
    ''' '''
    # while True:
    s = time.time()

    indexes = make_ccy_indexes(mt5_symbols['majors'], final_period='2 min', days=1)

    save_data(indexes)
    print('total mins for 120 days:', (time.time() - s) / 60)
   # save_index_data_for_mt5(indexes)

    # eur = indexes['EUR']
    # usd = indexes['USD']
    # gbp = indexes['GBP']
    # cad = indexes['CAD']
    # jpy = indexes['JPY']
    # aud = indexes['AUD']
    # nzd = indexes['NZD']
    
    import mplfinance as mpf

    # mpf.plot(eur,type='candle',show_nontrading=False, volume=True, title='eur')
    # mpf.plot(usd,type='candle',show_nontrading=False, volume=True, title='usd')
    # mpf.plot(gbp,type='candle',show_nontrading=False, volume=True, title='gbp')
    # mpf.plot(cad,type='candle',show_nontrading=False, volume=True, title='cad')
    # mpf.plot(jpy,type='candle',show_nontrading=False, volume=True, title='jpy')
    # mpf.plot(aud,type='candle',show_nontrading=False, volume=True, title='aud')
    # mpf.plot(nzd,type='candle',show_nontrading=False, volume=True, title='nzd')
    

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
