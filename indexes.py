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

ohlc_con = sqlite3.connect(ohlc_db)
vol_tick = pd.DataFrame()


def _request_ticks_and_resample(pair, period, _from):
    ''' Ticks will be resampled into 1 second bars '''

    _to = datetime.now()

    ticks = mt5.copy_ticks_range(pair, _from, _to, mt5.COPY_TICKS_ALL)
    df = pd.DataFrame(ticks)

    df = df.rename(columns = {df.columns[0]: 'datetime'})

    df.datetime = pd.to_datetime(df.datetime, unit='s')
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
    voltick = df.voltick.resample(final_period).last()
    
    resampled = pd.DataFrame({'open': open,
                            'high': high,
                            'low': low,
                            'close': close,
                            'volume': volume,
                            'voltick': voltick
                             })

    return resampled

def _final_ohlc_cleaning(df):

    # Set open prices to equal the former close
    df.open = df.close.shift(1)

    # Make sure high is highest and low is lowest
    for row in df.itertuples(index=True, name=None):
        i = row[0]
        df.loc[i, 'high'] = max(df.loc[i, ['open', 'high', 'low', 'close']])
        df.loc[i, 'low'] = min(df.loc[i, ['open', 'high', 'low', 'close']])

    # Finally, drop duplicates since it seems resampling creates weekend data
    df = df.drop_duplicates(subset=['open', 'high', 'low', 'close'])
    df = df.dropna(subset=['open'])
    
    return df

def make_ccy_indexes(pairs, initial_period='1s', final_period='1 min', days=60):
    '''  '''

    if not mt5.initialize(login=50341259, server="ICMarkets-Demo",password="ZhPcw6MG"):
        print("initialize() failed, error code =", mt5.last_error())
        quit()
    
    # This dict will store the currency indexes as data is received
    ccys = {
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
    _from = datetime.now() - pd.Timedelta(f'{days} day')

    for pair in pairs:
        
        df = _request_ticks_and_resample(pair, initial_period, _from)
        
        # Rather than deal with the prices, use the diff from one price to the next
        df = _diff(df)

        inverted = _invert(df.copy())
        inverted = _cumsum(inverted)
        inverted = _normalize(inverted)
        df = _cumsum(df)
        df = _normalize(df)

        # Add the df to its proper dict currency
        for ccy in ccys:

            if ccy == pair[:3]:
                if len(ccys[ccy]) == 0:
                    ccys[ccy] = df
                else:
                    ccys[ccy] += df
                continue
            
            # Counter currency
            if ccy == pair[-3:]:
                if len(ccys[ccy]) == 0:
                    ccys[ccy] = inverted
                else:
                    ccys[ccy] += inverted
    
    # vol_tick = pd.DataFrame()   putting this in global for now
    # Save the volume in its 1sec format for plotting later
    for ccy in ccys:
        df = ccys[ccy]
        hour_mean = df.volume.rolling(60*60).mean()
        rolling_max = df.volume.rolling(60*240).max()

        ccys[ccy]['voltick'] = df.close[
                                    (df.volume > hour_mean * 3)
                                    |
                                    (df.volume >= rolling_max)
                                    ]

        print('number of voltick signals:', len(df[df.voltick.notna()]))
    # Resample into 1 min
    for ccy in ccys:
        ccys[ccy] = _resample(ccys[ccy], final_period)
    
        ccys[ccy] = _final_ohlc_cleaning(ccys[ccy])

    return ccys



def save_data(ccys):
    ''' Try to append the data from the last row onward.  Otherwise save everything. '''

    # try:
    #     db = pd.read_sql("""SELECT * FROM USD 
    #                         ORDER BY datetime DESC 
    #                         LIMIT 1""", ohlc_con, parse_dates=True)
    #     db.index = pd.to_datetime(db.index)

    #     for ccy in ccys:
    #         df = ccys[ccy]
    #         df = df[df.index > db.index[0]]
    #         df.to_sql(f'{ccy}', ohlc_con, if_exists='append', index=True)
    
    # except:
       
    for ccy in ccys:
        df = ccys[ccy]
        df.to_sql(f'{ccy}', ohlc_con, if_exists='replace', index=True)
        # df.to_sql(f'{ccy}', ohlc_con, if_exists='append', index=True)
        

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

    indexes = make_ccy_indexes(mt5_symbols['majors'], final_period='15 min', days=8)
    save_data(indexes)
    save_index_data_for_mt5(indexes)

    df = indexes['EUR']
    # df = df.reset_index()  # otherwise plot weekend gaps

    
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df.open, 
                                        high=df.high,
                                        low=df.low, 
                                        close=df.close,
    increasing_line_color= 'gray', decreasing_line_color= 'gray'
    )])

    # ADD VOLTICK MARKERS
    fig.add_trace(go.Scatter(
    x=df.index[df.voltick.notna()],
    y=df.voltick[df.voltick.notna()],
    mode="markers",
    name="voltick",
    # color='#000000'
    # text=ratings,
    # textposition="top center"
    ))

    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(
    # remove this "xaxis" section to show weekend gaps
    xaxis = dict(  
                type="category"),
    paper_bgcolor="LightSteelBlue",  
)
    fig.show()

    time.sleep(1)

    df = indexes['AUD']
    # df = df.reset_index()  # otherwise plot weekend gaps

    
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df.open, 
                                        high=df.high,
                                        low=df.low, 
                                        close=df.close,
    increasing_line_color= 'gray', decreasing_line_color= 'gray'
    )])

    # ADD VOLTICK MARKERS
    fig.add_trace(go.Scatter(
    x=df.index[df.voltick.notna()],
    y=df.voltick[df.voltick.notna()],
    mode="markers",
    name="voltick",
    # color='#000000'
    # text=ratings,
    # textposition="top center"
    ))

    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(
    # remove this "xaxis" section to show weekend gaps
    xaxis = dict(  
                type="category"),
    paper_bgcolor="LightSteelBlue",  
)
    fig.show()

    time.sleep(1)

    df = indexes['NZD']
    # df = df.reset_index()  # otherwise plot weekend gaps

    
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df.open, 
                                        high=df.high,
                                        low=df.low, 
                                        close=df.close,
    increasing_line_color= 'gray', decreasing_line_color= 'gray'
    )])

    # ADD VOLTICK MARKERS
    fig.add_trace(go.Scatter(
    x=df.index[df.voltick.notna()],
    y=df.voltick[df.voltick.notna()],
    mode="markers",
    name="voltick",
    # color='#000000'
    # text=ratings,
    # textposition="top center"
    ))

    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(
    # remove this "xaxis" section to show weekend gaps
    xaxis = dict(  
                type="category"),
    paper_bgcolor="LightSteelBlue",  
)
    fig.show()

    time.sleep(1)

    df = indexes['CAD']
    # df = df.reset_index()  # otherwise plot weekend gaps

    
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df.open, 
                                        high=df.high,
                                        low=df.low, 
                                        close=df.close,
    increasing_line_color= 'gray', decreasing_line_color= 'gray'
    )])

    # ADD VOLTICK MARKERS
    fig.add_trace(go.Scatter(
    x=df.index[df.voltick.notna()],
    y=df.voltick[df.voltick.notna()],
    mode="markers",
    name="voltick",
    # color='#000000'
    # text=ratings,
    # textposition="top center"
    ))

    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.update_layout(
    # remove this "xaxis" section to show weekend gaps
    xaxis = dict(  
                type="category"),
    paper_bgcolor="LightSteelBlue",  
)
    fig.show()

    time.sleep(1)