import MetaTrader5 as mt5
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import time
from create_db import econ_db
from symbols_lists import mt5_symbols
from ohlc_request import mt5_ohlc_request
from order_params import enter_trade
from tokens import bot

# import sys
# sys.path.append('C:/Program Files (x86)/Python38-32/code')
# import format_functions
# import trades


def send_trade_alert(symbol, timeframe, pattern, bot=bot):
    bot.send_message(chat_id=446051969, text=f'{symbol} {timeframe} {pattern}')
    
econ_con = sqlite3.connect(econ_db)

df = pd.DataFrame()
timeframe = None
symbol = None

def spread_is_ok(df, i, symbol):

    info = mt5.symbol_info(symbol)

    # Get the spread in pips rather than points
    spread = info.spread * info.point

    if spread <= df.loc[i, 'atr'] * 1/3:
        return True
    else:
        return False

##### Format functions ######
def format_data(df):
    # df = df[:-1] # sometimes calculating the current bar throws errors
    df = df.rename(columns={'time': 'dt', 'tick_volume': 'volume'})
    df.dt = pd.to_datetime(df.dt, unit='s')
    df['pattern'] = np.nan

    return df

def set_bar_type(df):

    df.loc[df.close > df.open, 'bar_type'] = 'up'
    df.loc[df.close < df.open, 'bar_type'] = 'down'
    
def hlc3(df):
    df['hlc3'] = (df.high + df.low + df.close) / 3

def pinbar(df):
    top = []
    bot = []
    df['pinbar'] = np.nan
    
    # Only calculate the last couple bars because this code is slow af
    dfdf = df.tail(3).copy().reset_index()

    # get the highest of either the open or close
    for row in dfdf.itertuples(index=True, name=None):
        i = row[0]
        top.append(max(dfdf.loc[i, 'close'], dfdf.loc[i, 'open']))
        bot.append(min(dfdf.loc[i, 'close'], dfdf.loc[i, 'open']))

        # get the size of upper and lower wicks to compare
        upper = dfdf.loc[i, 'high'] - top[i]
        lower = bot[i] - dfdf.loc[i, 'low']
        if dfdf.loc[i, 'high'] - dfdf.loc[i, 'low'] > dfdf.loc[i, 'atr']:
            try:
                if upper > lower:
                    size = upper / (dfdf.loc[i, 'high'] - dfdf.loc[i, 'low']) # wick needs to be a min %
                    if size > 0.25:
                        dfdf.loc[i, 'pinbar'] = size * -1  # use a negative value for sell setups 
                else:
                    size = lower / (dfdf.loc[i, 'high'] - dfdf.loc[i, 'low']) # wick needs to be a min %
                    if size > 0.25:
                        df.loc[i, 'pinbar'] = size 
            except: 
                print('pinbar error')
                df.loc[i, 'pinbar'] = np.nan
                pass
    
def find_peaks(df):
    df['peak_hi'] = np.nan
    df['peak_lo'] = np.nan
    fade = 3  # these are used for a minimum swing size filter from a peak
    spring = 2 

    ############### SWING LOWS
    temp = df[(df.low < df.low.shift(1)) &
                (df.low <= df.low.shift(-1)) &
                (df.low < df.low.shift(2)) &
                (df.low <= df.low.shift(-2)) &
                (df.low < df.low.shift(3)) &
                (df.low <= df.low.shift(-3)) &
                (df.low < df.low.shift(4)) 
                ]

    win = 4
    for row in temp.itertuples(name=None, index=True):
        i = row[0]
        # Price move from the peak (forwards then backwards)
        if (df.loc[i:i+win, 'high'].max() - df.loc[i, 'low']) > fade * df.loc[i, 'atr']:
            if (df.loc[i-win:i, 'high'].max() - df.loc[i, 'low']) > fade * df.loc[i, 'atr']:
                df.loc[i, 'peak_lo'] = 'fade'
                pass
        # If not found look for the spring sized trades
        if (df.loc[i:i+win, 'high'].max() - df.loc[i, 'low']) > spring * df.loc[i, 'atr']:
            if (df.loc[i-win:i, 'high'].max() - df.loc[i, 'low']) > spring * df.loc[i, 'atr']:
                df.loc[i, 'peak_lo'] = 'spring'


    ################ SWING HIGHS
    temp = df[(df.high > df.high.shift(1)) &
            (df.high >= df.high.shift(-1)) &
            (df.high > df.high.shift(2)) &
            (df.high >= df.high.shift(-2)) &
            (df.high > df.high.shift(3)) &
            (df.high >= df.high.shift(-3)) &
            (df.high > df.high.shift(4)) 
            ]

    win = 4
    for row in temp.itertuples(name=None, index=True):
        i = row[0]
        # Price move from the peak (forwards then backwards)
        if (df.loc[i, 'high'] - df.loc[i:i+win, 'low'].min()) > fade * df.loc[i, 'atr']:
            if (df.loc[i, 'high'] - df.loc[i-win:i, 'low'].min()) > fade * df.loc[i, 'atr']:
                df.loc[i, 'peak_hi'] = 'fade'
                pass
        # If not found look for the swing sized swings
        if (df.loc[i, 'high'] - df.loc[i:i+win, 'low'].min()) > spring * df.loc[i, 'atr']:
            if (df.loc[i, 'high'] - df.loc[i-win:i, 'low'].min()) > spring * df.loc[i, 'atr']:
                df.loc[i, 'peak_hi'] = 'spring'
    # print(len(df[df.peak.notna()]), 'atr filtered peaks')

def set_ema(df):
    df['ema_7'] = df.hlc3.ewm(span=7,adjust=False).mean()
    df['ema_9'] = df.hlc3.ewm(span=9,adjust=False).mean()
    df['ema_20'] = df.hlc3.ewm(span=20,adjust=False).mean()
    df['ema_50'] = df.hlc3.ewm(span=50,adjust=False).mean()
    df['ema_100'] = df.hlc3.ewm(span=100,adjust=False).mean()

def wwma(values, n):
    return values.ewm(alpha=1/n, adjust=False).mean()

def atr(df, n=14):
    data = pd.DataFrame()
    data['tr0'] = abs(df.high - df.low)
    data['tr1'] = abs(df.high - df.close.shift())
    data['tr2'] = abs(df.low - df.close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = wwma(tr, n)
    df['atr'] = atr

def avg_vol(df):
    # Relative volume (normalized)
    df['avg_vol'] = df.volume / (df.high - df.low)
    df.avg_vol = (df.avg_vol - df.avg_vol.rolling(30).min()) / (df.avg_vol.rolling(30).max() - df.avg_vol.rolling(30).min())

def avg_bar_size(df):
    # Relative bar size (normalized)
    # (this is something I'd like to test but haven't used)
    df['avg_bar_size'] = (df.high - df.low)
    df.avg_bar_size = (df.avg_bar_size - df.avg_bar_size.rolling(30).min()) / (df.avg_bar_size.rolling(30).max() - df.avg_bar_size.rolling(30).min())

def auction_vol(df):
    ''' Price trending with lessening volume '''
    temp = df[((df.hlc3.shift(2) < df.hlc3.shift(1)) &  # up trend
            (df.high == df.high.rolling(10).max()) & 
            (df.volume.shift(1) > df.volume)) | 
            ((df.hlc3.shift(2) > df.hlc3.shift(1)) & # down trend
            (df.low == df.low.rolling(10).min()) & 
            (df.volume.shift(1) > df.volume))
            ]
    df.loc[temp.index, 'auction_vol'] = 1 # one bar with less volume

    # twos
    temp = df[((df.hlc3.shift(2) < df.hlc3.shift(1)) &
            (df.high == df.high.rolling(10).max()) & 
            (df.volume.shift(2) > df.volume.shift(1)) & # extra volume bar required
            (df.volume.shift(1) > df.volume)) | 
            ((df.hlc3.shift(2) > df.hlc3.shift(1)) &
            (df.low == df.low.rolling(10).min()) & 
            (df.volume.shift(2) > df.volume.shift(1)) &
            (df.volume.shift(1) > df.volume))
            ]
    df.loc[temp.index, 'auction_vol'] = 2 # two bars with less volume

def save_trade_info(df, i):
    '''
    Save the trade info to the df
    '''
    cur_atr = df.loc[i, 'atr']
    sl_r = 2
    tp1_r = 1.5
    tp2_r = 4
    entry_mult = 0.2

    if df.loc[i, 'pattern'][-2:] == '_b':

        entry = df.loc[i, 'close'] + cur_atr * entry_mult
        sl_pips = cur_atr * sl_r
        sl = entry - sl_pips
        tp1_pips = cur_atr * tp1_r
        tp1 = entry + tp1_pips
        tp2_pips = cur_atr * tp2_r
        tp2 = entry + tp2_pips
        df.loc[i, 'entry'] = entry
        df.loc[i, 'sl'] = sl
        df.loc[i, 'tp1'] = tp1
        df.loc[i, 'tp2'] = tp2
    
    if df.loc[i, 'pattern'][-2:] == '_s':

        entry = df.loc[i, 'close'] - cur_atr * entry_mult
        sl_pips = cur_atr * sl_r
        sl = entry + sl_pips
        tp1_pips = cur_atr * tp1_r
        tp1 = entry - tp1_pips
        tp2_pips = cur_atr * tp2_r
        tp2 = entry - tp2_pips
        df.loc[i, 'entry'] = entry
        df.loc[i, 'sl'] = sl
        df.loc[i, 'tp1'] = tp1
        df.loc[i, 'tp2'] = tp2



#### TRADE SETUPS ######

def spring_b(df):  
    ''' Low volume springs in an up trend. '''
    # Main entry filters 
    temp = df[(df.low == df.low.rolling(10).min()) &
            (df.ema_9 > df.ema_20) &
            (df.ema_20 > df.ema_50) &
            ((df.auction_vol > 1) |
            ((df.auction_vol > 0) &
            (df.pinbar > 0.5)))
            ]
    # print(len(temp))
    # Verify exactly 1 peak exists in the 15 bars prior to the entry signal
    for row in temp.itertuples(name=None, index=True):
        i = row[0]
        p = df.loc[i-20:i-1, 'peak_lo'][(df.loc[i-20:i-1, 'peak_lo'] == 'spring') |
                                        (df.loc[i-20:i-1, 'peak_lo'] == 'fade')]
        if len(p) == 1:
            # A low must be made below the peak_lo but not more than 1.5 atr away
            if df.loc[i, 'low'] < df.loc[p.index[0], 'low']:
                if df.loc[i, 'low'] > df.loc[p.index[0], 'low'] - df.loc[i, 'atr'] * 1.5:
                    df.loc[i, 'pattern'] = 'spring_b'
                    save_trade_info(df, i)

def spring_s(df):   
    ''' Low volume springs in a down trend. '''
    # Main entry filters 
    temp = df[(df.high == df.high.rolling(10).max()) &
            #   (df.ema_7 > df.ema_9) &
            (df.ema_9 < df.ema_20) &
            (df.ema_20 < df.ema_50) &
            ((df.auction_vol > 1) |
            ((df.auction_vol > 0) &
            (df.pinbar < -0.5)))
            ]
    # print(len(temp))
    # Verify exactly 1 peak exists in the 15 bars prior to the entry signal
    for row in temp.itertuples(name=None, index=True):
        i = row[0]
        p = df.loc[i-20:i-1, 'peak_hi'][(df.loc[i-20:i-1, 'peak_hi'] == 'spring') |
                                        (df.loc[i-20:i-1, 'peak_hi'] == 'fade')]
        if len(p) == 1:
            if df.loc[i, 'high'] > df.loc[p.index[0], 'high']:
                if df.loc[i, 'high'] < df.loc[p.index[0], 'high'] + df.loc[i, 'atr'] * 1.5:
                    df.loc[i, 'pattern'] = 'spring_s'
                    save_trade_info(df, i)

def fade_b(df): 
    # Main entry filters 
    temp = df[(df.low == df.low.rolling(20).min()) &
            (df.ema_20 < df.ema_50) 
            ]

    for row in temp.itertuples(name=None, index=True):
        i = row[0]
        p = df.loc[i-20:i-1][df.loc[i-20:i-1, 'peak_lo'] == 'fade']
        if len(p) > 0:
            # Get the last peak's index loc
            prior_peak = p[p.low == p.low.min()].index[-1]
            # Lowest low in the last 20 bars (besides the signal bar and the bar preceeding signal bar)
            if df.loc[prior_peak, 'low'] == df.loc[i-20:i-2, 'low'].min():
                # Signal low must not be more than 1.5 atr away from prior lows
                if df.loc[i, 'low'] > p.low.any() - (1.6 * df.loc[i, 'atr']):
                    # EMA alignment
                    if df.loc[prior_peak, 'ema_9'] < df.loc[prior_peak, 'ema_20'] and df.loc[prior_peak, 'ema_20'] < df.loc[prior_peak, 'ema_50']:
                        # Less vol than prev preak or auc vol
                        if df.loc[i, 'volume'] < df.loc[prior_peak, 'volume'] or df.loc[i, 'auction_vol'] == 2:
                            df.loc[i, 'pattern'] = 'fade_b'
                            save_trade_info(df, i)

def fade_s(df):   
    # Main entry filters 
    temp = df[(df.high == df.high.rolling(20).max()) &
            (df.ema_20 > df.ema_50) 
            ]

    for row in temp.itertuples(name=None, index=True):
        i = row[0]
        p = df.loc[i-20:i-1][df.loc[i-20:i-1, 'peak_hi'] == 'fade']
        if len(p) > 0:
            # Get the last peak's index loc
            prior_peak = p[p.high == p.high.max()].index[-1]
            # Ensure it is highest high in the last 20 bars (besides the signal bar and the bar preceeding signal bar)
            if df.loc[prior_peak, 'high'] == df.loc[i-20:i-2, 'high'].max():
                # Signal high must not be more than 1.5 atr away from last peak
                if df.loc[i, 'high'] < p.high.any() + (1.6 * df.loc[i, 'atr']):
                    # EMA alignment
                    if df.loc[prior_peak, 'ema_9'] > df.loc[prior_peak, 'ema_20'] and df.loc[prior_peak, 'ema_20'] > df.loc[prior_peak, 'ema_50']:
                        # Ensure this bar has less vol than prev preak or auc vol
                        if df.loc[i, 'volume'] < df.loc[prior_peak, 'volume'] or df.loc[i, 'auction_vol'] == 2:
                            df.loc[i, 'pattern'] = 'fade_s'
                            save_trade_info(df, i)

def stoprun_b(df):  

    # Main entry filters 
    temp = df[(df.low == df.low.rolling(20).min()) &
            #   (df.avg_bar_size > .8) &
            (df.close < df.ema_50) &
            ((df.auction_vol > 1) |
            ((df.auction_vol > 0) &
            (df.pinbar > 0.5)))
            ]
    # Verify > 1 peak exists within n bars prior to the entry signal
    for row in temp.itertuples(name=None, index=True):
        i = row[0]
        p = df.loc[i-25:i-1, 'peak_lo'][df.loc[i-25:i-1, 'peak_lo'].notna()]
        if len(p) > 1:        
            if df.loc[p.index[0], 'low'] == df.loc[p.index[0]-20:p.index[0], 'low'].min():
                if df.loc[p.index[0], 'ema_20'] < df.loc[p.index[0], 'ema_50']:
                    if df.loc[p.index[1], 'low'] > df.loc[p.index[0], 'low'] - df.loc[i, 'atr'] * 1.5:
                        if df.loc[p.index[1], 'low'] < df.loc[p.index[0], 'low'] + df.loc[i, 'atr'] * 1:
                            if df.loc[i, 'low'] > df.loc[p.index[0], 'low'] - df.loc[i, 'atr'] * 1.8:
                                df.loc[i, 'pattern'] = 'stoprun_b'
                                save_trade_info(df, i)

def stoprun_s(df):  
    # Main entry filters 
    temp = df[(df.high == df.high.rolling(25).max()) &
            #   (df.avg_bar_size > .8) &
            (df.close > df.ema_50) &
            ((df.auction_vol > 1) |
            ((df.auction_vol > 0) &
            (df.pinbar < -0.5)))
            ]

    # Verify exactly 1 peak exists in the 15 bars prior to the entry signal
    for row in temp.itertuples(name=None, index=True):
        i = row[0]
        p = df.loc[i-25:i-1, 'peak_hi'][df.loc[i-25:i-1, 'peak_hi'].notna()]
        if len(p) > 1:
            if df.loc[p.index[0], 'high'] == df.loc[p.index[0]-20:p.index[0], 'high'].max():
                if df.loc[p.index[0], 'ema_20'] > df.loc[p.index[0], 'ema_50']:
                    if df.loc[p.index[1], 'high'] < df.loc[p.index[0], 'high'] + df.loc[i, 'atr'] * 1.5:
                        if df.loc[p.index[1], 'high'] > df.loc[p.index[0], 'high'] - df.loc[i, 'atr'] * 1:
                            if df.loc[i, 'high'] < df.loc[p.index[0], 'high'] + df.loc[i, 'atr'] * 1.8:
                                df.loc[i, 'pattern'] = 'stoprun_s'
                                save_trade_info(df, i)


def trade_scanner(timeframe):
    ''' Read the database for the current timeframe. This function gets
    called within a loop so only a single timeframe will be passed. '''
    
    b = []
    b.extend(mt5_symbols['majors'])
    b.extend(mt5_symbols['others'])
    for symbol in b:

        df = mt5_ohlc_request(symbol, timeframe)

        atr(df)
        hlc3(df)
        set_ema(df)
        auction_vol(df)
        avg_vol(df)
        pinbar(df)
        avg_bar_size(df)
        find_peaks(df)
        fade_b(df)
        fade_s(df)
        spring_b(df) 
        spring_s(df)
        stoprun_b(df)
        stoprun_s(df)

        i = df.tail(1).index[0]
        pattern = df.loc[i, 'pattern']
        
        if not pd.isnull(pattern) and spread_is_ok(df, i, symbol):
            print(symbol, timeframe, pattern)
            # rather than enter the trade here, send that df row back to 'main'
            # ... but i can't just return the row andbreak the loop
            send_trade_alert(symbol, timeframe, pattern)
            enter_trade(df, i, symbol, timeframe)
            # forecast_df = pd.read_sql('outlook', econ_con)



def trade_scan_handler():
    ''' A continuous loop that checks the database for the last
    timestamp of each timeframe.  If a newer timestamp appears
    than what's been saved, the database has been updated and 
    it's time to scan for trades on that timeframe. '''

    # Create the dict which will hold the latest timestamp of each timeframe
    latest_timestamps = {}

    # Read the database (symbol doesn't matter, could be anything)
    db = pd.read_sql(r"SELECT datetime, timeframe FROM 'EUR/USD'", ohlc_con, parse_dates=['datetime'])
    timeframes = db.timeframe.unique()

    # Check the last timestamp for each timeframe in the database
    for tf in timeframes:
        datetime = db.datetime[db.timeframe == tf]
        latest_datetime = datetime.values[-1]

        # Add the value to the dict
        latest_timestamps[tf] = latest_datetime
                
    # Now continually check for newer timestamps in the database for each timeframe
    while True:

        # Read the database 
        db = pd.read_sql(r"SELECT datetime, timeframe FROM 'EUR/USD'", ohlc_con, parse_dates=['datetime'])

        for tf in timeframes:
            datetime = db.datetime[db.timeframe == tf]
            latest_datetime = datetime.values[-1]
            
            # See if any are newer than what's in the 'times' dict
            if latest_datetime > latest_timestamps[tf]:

                # Update dict
                latest_timestamps[tf] = latest_datetime

                # Scan for trade setups on that timeframe
                return _trade_scanner(tf)