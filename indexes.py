import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import pathlib
import sqlite3
from pandas.core.indexes.base import ensure_index
import mplfinance as mpf
import concurrent.futures
from symbols_lists import mt5_symbols, indexes
from tokens import mt5_login, mt5_pass

INDEX_OHLC = r'C:\Users\ru\forex\db\indexes'
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

def _request_ticks_and_resample(pair, period, _from, _to) -> list:
    ''' Ticks will be resampled into 1 second bars. Returns a list with
    name of pair in index 0 and dataframe in index 1. '''

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

    # Return the name of pair along with the dataframe
    return [pair, resampled]

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

def _reorder_timeframe_if_needed(timeframe: str) -> str:
    ''' Reorder if needed, number must come first. '''

    if len(timeframe) <= 3: 
        if timeframe[0] == ('H' or 'M'):
            tf = timeframe[::-1]
        else:
            tf = timeframe
    else:
        tf = timeframe
    
    return tf

def _resample(ohlc_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    ''' Just a simple resampling into higher timeframes for whatever
    OHLCV data set gets passed in. '''

    tf = _reorder_timeframe_if_needed(timeframe)

    o = ohlc_df.open.resample(tf).first()
    h = ohlc_df.high.resample(tf).max()
    l = ohlc_df.low.resample(tf).min()
    c = ohlc_df.close.resample(tf).last()
    v = ohlc_df.volume.resample(tf).sum()

    ohlc_df = pd.DataFrame({'open': o,
                        'high': h,
                        'low': l,
                        'close': c,
                        'volume': v,
                        })
    return ohlc_df

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

def build_in_memory_database(pairs: list, initial_period='1s', days=60, _from=None) -> dict:
    ''' Use threading to get all data into memory.  Structured into a dict
    with the name of a pair as the key and the value being a dataframe of
    1 second periods. '''

    if not mt5.initialize(login=mt5_login, server="ICMarkets-Demo",password=mt5_pass):
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    # Create this timestamp outside of loop so it doesn't change.
    _to = datetime.now()
    if _from is None:
        _from = _to - pd.Timedelta(f'{days} day')

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [] 
        for pair in pairs:
            futures.append(executor.submit(_request_ticks_and_resample, pair=pair, period=initial_period, _from=_from, _to=_to))
        
        database = {} 
        for future in concurrent.futures.as_completed(futures):
            data = future.result()
            database[data[0]] = data[1] # unpack name of pair and dataframe

        # confirmed working
        return database

def _functions_handler(database: dict, index: str, final_period: str) -> list:
    ''' This will handle a single index, looping through each key in
    the dict and if the name of the key contains the symbol of the index
    it will process that data appropriately. (If index is EUR, it will
    process EURCAD but not AUDNZD etc)'''

    aggregate_data = pd.DataFrame()
    for pair, df in database.items():
        
        # Base currency
        if pair[:3] == index:
            base = _diff(df.copy())
            base = _cumsum(base)
            base = _normalize(base)

            if len(aggregate_data) == 0:
                aggregate_data = base
            else:
                aggregate_data += base
            continue
        
        # Counter currency
        if pair[-3:] == index:
            counter = _diff(df.copy())
            counter = _invert(counter)
            counter = _cumsum(counter)
            counter = _normalize(counter)

            if len(aggregate_data) == 0:
                aggregate_data = counter
            else:
                aggregate_data += counter

    # Resample into some other timeframe
    if final_period != '1s':
        aggregate_data = _resample(aggregate_data, final_period)

    aggregate_data = _final_ohlc_cleaning(aggregate_data, index)

    return [index, aggregate_data]

def run_calculation_processes(database, indexes, final_period='5 min') -> None:
    ''' This will create the separate processes, one for each index. Ultimately
    the data will be saved to parquet files. '''

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [] 
        for index in indexes:
            futures.append(executor.submit(_functions_handler, database=database, index=index, final_period=final_period))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                data = future.result()
                index = data[0]
                df = data[1]
            except Exception:
                print('A process failed... thats all I know.')
                continue
            if df.empty:
                print(index, 'dataframe is empty')
            df.to_parquet(pathlib.Path(INDEX_OHLC, f'{index}_M5.parquet'), index=True)

if __name__ == '__main__':
    # while True:
    s = time.time()
    print('building databse')
    database = build_in_memory_database(mt5_symbols['majors'], days=20)  # confirmed working
    print('done. now creating indexes')
    run_calculation_processes(database, indexes, final_period='5 min')
    print(f'Finished in {(time.time()-s)/60} minutes.')
    time.sleep(20*60)

# 40 days without futures / with futures: 
#                2.42 min / 1.38 min
    
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
