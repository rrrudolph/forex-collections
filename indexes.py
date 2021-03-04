import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
import time
import pathlib
import sqlite3
from symbols_lists import mt5_symbols
import mplfinance as mpf
from create_db import ohlc_db

ohlc_con = sqlite3.connect(ohlc_db)


def _request_ticks_and_resample(pair, days, period):
    ''' Ticks will be resampled into 1 second bars '''

    try:
        df = pd.read_sql(f'SELECT * FROM USD', ohlc_con)
        df.index = pd.to_datetime(df.index)
        _from = df.tail(1).index
        _from -= pd.Timedelta('4 days')
    except:
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
    # df.open = (df.open - min(df.open)) / (max(df.open) - min(df.open))
    # df.high = (df.high - min(df.high)) / (max(df.high) - min(df.high))
    # df.low = (df.low - min(df.low)) / (max(df.low) - min(df.low))
    # df.close = (df.close - min(df.close)) / (max(df.close) - min(df.close))
    # df.volume = (df.volume - min(df.volume)) / (max(df.volume) - min(df.volume))
    df = df.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    
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
    
    resampled = pd.DataFrame({'open': open,
                            'high': high,
                            'low': low,
                            'close': close,
                            'volume': volume
                             })

    return resampled


def make_ccy_indexes(pairs, initial_period='1s', days=40):
    ''' timeframe should be passed as a string like '15 min'. 
    To plot the data on MT5 resample to 1min. The platform will handle
    further resampling from there.'''

    if not mt5.initialize(login=50341259, server="ICMarkets-Demo",password="ZhPcw6MG"):
        print("initialize() failed, error code =", mt5.last_error())
        quit()
    
    # This dict will store the dfs of each pair for each ccy (ie, 'USD': 'EURUSD', 'USDJPY', etc)

    ccys = {'USD': [],
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

        # Adjust to CST
        df.index = df.index - pd.Timedelta('6 hours')
        
        # Rather than deal with the prices, use the diff from one price to the next
        df = _diff(df)

        inverted = _invert(df.copy())
        df = _cumsum(df)
        df = _normalize(df)
        inverted = _cumsum(inverted)
        inverted = _normalize(inverted)

        # Assign the df to its proper dict group
        for ccy in ccys:

            if pair[:3] == ccy:
                # If blank, add in first df as is. else, add
                if not ccys[ccy]:
                    ccys[ccy] = df
                else:
                    ccys[ccy] += df
                continue
            
            # Make values negative if k is in the secondary position
            if pair[-3:] == ccy:
                if not ccys[ccy]:
                    ccys[ccy] = inverted
                else:
                    ccys[ccy] += inverted

    # Resample into 1 min
    for ccy in ccys:
        ccys[ccy] = _resample(ccys[ccy], '1 min')

    return ccys

def save_data(ccys):
    ''' Try to append the data from the last row onward.  Otherwise save everything. '''


    try:
        db = pd.read_sql(f'SELECT * FROM USD ORDER BY datetime DESC LIMIT 1', ohlc_con)
        db.index = pd.to_datetime(db.index)

        for ccy in ccys:
            df = ccys[ccy]
            df = df[df.index > db.index[0]]
            df.to_sql(f'{ccy}', ohlc_con, if_exists='append', index=True)
    
    except:
       
        for ccy in ccys:
            df = ccys[ccy]
            df.to_sql(f'{ccy}', ohlc_con, if_exists='append', index=True)
        


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


if __name__ == '__main__':
    start = time.time()
    indexes = make_ccy_indexes(mt5_symbols['majors'], '15min', initial_period='1s', days=3)
    df = indexes['JPY']
    mpf.plot(df, type='candle', volume=True, show_nontrading=False)

    # print(indexes)
    end = time.time()
    # # save_data_for_mpf(indexes)
    print('time:',end-start)

    # # send to g sheets
    # df = indexes['USD']
    # df['datetime'] = df.index
    # df.datetime = df.datetime.astype(str)

    # adr_sheet.clear()
    # adr_sheet.update([df.columns.values.tolist()] + df.values.tolist())
