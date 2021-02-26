import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import date, datetime
import time
import pathlib
from symbols_lists import mt5_symbols, index_ccys
from ohlc_request import mt5_ohlc_request
from tokens import adr_sheet



def _request_ticks_and_resample(pair, days, period):
    ''' Ticks will be resampled into 1 second bars '''

    _from = datetime.now() - pd.Timedelta(f'{days} day')
    _to = datetime.now()

    ticks = mt5.copy_ticks_range(pair, _from, _to, mt5.COPY_TICKS_ALL)
    df = pd.DataFrame(ticks)

    df = df.rename(columns = {df.columns[0]: 'datetime'})

    df.datetime = pd.to_datetime(df.datetime, unit='s')
    df = df.set_index(df.datetime, drop=True)

    # save volume (count each tick)
    df.volume = 1

    # to avoid spikes due to a widening spread get an average price
    df['price'] = (df.bid + df.ask) / 2 
    df = df[['price', 'volume']]

    # Resample, fill blanks, and get rid of the multi level column index
    df = df.resample(period).agg({'price': 'ohlc', 'volume': 'sum'})
    df = df.fillna(method='ffill')
    df.columns = df.columns.get_level_values(1)

    return df


def _normalize(df):
    df.open = (df.open - min(df.open)) / (max(df.open) - min(df.open))
    df.high = (df.high - min(df.high)) / (max(df.high) - min(df.high))
    df.low = (df.low - min(df.low)) / (max(df.low) - min(df.low))
    df.close = (df.close - min(df.close)) / (max(df.close) - min(df.close))
    df.volume = (df.volume - min(df.volume)) / (max(df.volume) - min(df.volume))
    
    return df


def _invert_counter_pairs(df):
    ''' If the currency isn't the base, make its values negative (ie, EURUSD on the USD index)'''

    df.open *= -1
    df.high *= -1
    df.low *= -1
    df.close *= -1

    return df

def _combine_dfs(dfs, base_ccy, final_period, indexes):
    ''' Combine the individual dfs of each pair into one index df and then 
    resample into whatever timeframe is desired. For plotting on MT5, resample to 1min.'''

    # testing out putting normalization here
    # which would be after the price inversions
    for i, df in enumerate(dfs):
        dfs[i] = _normalize(df)

    temp = dfs[0]
    for df in dfs[1:]:
        temp.open += df.open
        temp.high += df.high
        temp.low += df.low
        temp.close += df.close
        temp.volume += df.volume

    # Resample 
    open = temp.open.resample(final_period).first()
    high = temp.high.resample(final_period).max()
    low = temp.low.resample(final_period).min()
    close = temp.close.resample(final_period).last()
    volume = temp.volume.resample(final_period).sum()
    
    combined = pd.DataFrame({'open': open,
                            'high': high,
                            'low': low,
                            'close': close,
                            'volume': volume
                             })

    # Put the df in a dict
    indexes[base_ccy] = combined




def make_ccy_indexes(pairs, final_period, initial_period='1s', days=5):
    ''' timeframe should be passed as a string like '15 min'. 
    To plot the data on MT5 resample to 1min. The platform will handle
    further resampling from there.'''

    if not mt5.initialize(login=50341259, server="ICMarkets-Demo",password="ZhPcw6MG"):
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    
    # This dict will store the dfs of each pair for each ccy (ie, 'USD': 'EURUSD', 'USDJPY', etc)
    groups = {'USD': [],
              'EUR': [],
              'GBP': [],
              'JPY': [],
              'AUD': [],
              'NZD': [],
              'CAD': [],
              'CHF': [],
              }

    for pair in pairs:
        
        # Add the dfs of tick data to the ticks dict
        df = _request_ticks_and_resample(pair, days, initial_period)

        # Assign the df to its proper dict group
        for k in groups:

            if pair[:3] == k:
                groups[k].append(df)
                continue
            
            # Make prices negative if k is in the secondary position
            if pair[-3:] == k:
                df = _invert_counter_pairs(df)
                groups[k].append(df)

    # Now that the groups dict is filled, combine each group into a single index
    # (pass the list of dfs in each group to _combine_dfs which will then combine them)
    indexes = {}
    for k in groups:
        _combine_dfs(groups[k], k, final_period, indexes)

    return indexes 



# start = time.time()
# end = time.time()
# print('time:',end-start)

# Save the output to a csv
def save_index_data_for_mt5(indexes):
    ''' format the data for mt5 and save to csv. '''

    for k in indexes:
        
        df = indexes[k]
        # first subtract 6 hours to get it into CST
        df.index = df.index - pd.Timedelta('6 hours')

        # add the necessary columns
        df['date'] = [d.date() for d in df.index]
        df['time'] = [d.time() for d in df.index]
        df['r_vol'] = np.nan
        df['spread'] = np.nan
        
        # reorder (real volume and spread are ok to be nan i think)
        df = df[['date', 'time', 'open', 'high', 'low', 'close', 'volume', 'r_vol', 'spread']]

        # save to csv
        p = r'C:\Users\ru\AppData\Roaming\MetaQuotes\Terminal\67381DD86A2959850232C0BA725E5966\bases\Custom'
        
        df.to_csv(pathlib.Path(p, f'{k}x.csv'), index=False)


def save_data_for_mpf(indexes):
    
    for k in indexes:
        
        df = indexes[k]
        df.index = df.index - pd.Timedelta('6 hours')
        
        p = r'Desktop'
        df.to_csv(pathlib.Path(p, f'{k}x.csv'), index=True)


# if __name__ == 'main':
    # start = time.time()
    # indexes = make_ccy_indexes(mt5_symbols['majors'], '15min', initial_period='1s', days=5)

    # end = time.time()
    # # save_data_for_mpf(indexes)
    # print('time:',end-start)

    # # send to g sheets
    # df = indexes['USD']
    # df['datetime'] = df.index
    # df.datetime = df.datetime.astype(str)

    # adr_sheet.clear()
    # adr_sheet.update([df.columns.values.tolist()] + df.values.tolist())
