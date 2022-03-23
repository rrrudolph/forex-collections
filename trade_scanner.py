import pandas as pd
import numpy as np
import telegram
import mplfinance as mpf
from datetime import datetime
from symbols_lists import mt5_symbols, indexes, candles_per_day, etf_to_htf, etf_to_int, basket_pairs
from ohlc_request import mt5_ohlc_request
from pathlib import Path
import time
from indexes import INDEXES_PATH
import json

bot = telegram.Bot(token='1777249819:AAGiRaYqHVTCwYZMGkjEv6guQ1g3NN9LOGo')
p = r'C:\Users\ru\forex'

df = pd.DataFrame()
timeframe = None
symbol = None

##### Format functions #####

def _resample(ohlc_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    ''' Just a simple resampling into higher timeframes for whatever
    OHLCV data set gets passed in. '''


    # Reorder if needed. Number must come first
    if any(timeframe[0] == x for x in ['D','H','M','W']):
        if len(timeframe) == 2: 
            tf = timeframe[::-1]
        if len(timeframe) == 3: 
            tf = timeframe[1] + timeframe[2] + timeframe[0]
    else:
        tf = timeframe
    
    # monthly -> minute
    if tf[-1] == 'M':
        tf = tf[:-1] + ' min'

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
    assert len(ohlc_df) > 0, "\n~~~ Resample error. ~~~"
    return ohlc_df

def set_bar_type(df):

    df.loc[df.close > df.open, 'bar_type'] = 'up'
    df.loc[df.close < df.open, 'bar_type'] = 'down'
    
def pinbar(df):
    
    df['pinbar'] = np.nan

    ups = df[
        (df.bar_type == 'up')
        &
        (df.open - df.low > (df.close - df.open) * 2)
        ]
    df.loc[ups.index, 'pinbar'] = 'up'

    downs = df[
        (df.bar_type == 'down')
        &
        (df.high - df.open > (df.open - df.close) * 2)
        ]
    df.loc[downs.index, 'pinbar'] = 'down'
      
def find_simple_peaks(df):
    ''' This version will not use ATR filtering and will find
    big and small peaks.'''

    df['high_peak'] = np.nan
    df['low_peak'] = np.nan

    small_highs = df[
            (df.high > df.high.shift(1)) 
            &
            (df.high >= df.high.shift(-1)) 
            &
            (df.high > df.high.shift(2)) 
            &
            (df.high >= df.high.shift(-2)) 
            &
            (df.high > df.high.shift(3)) 
            &
            (df.high >= df.high.shift(-3)) 
            &
            (df.high > df.high.shift(4)) 
            &
            (df.high >= df.high.shift(-4)) 
            &
            (df.high > df.high.shift(5)) 
            &
            (df.high >= df.high.shift(-5))
            ].index 
    df.loc[small_highs, 'high_peak']= 'small'
    
    small_lows = df[
            (df.low < df.low.shift(1)) 
            &
            (df.low <= df.low.shift(-1)) 
            &
            (df.low < df.low.shift(2)) 
            &
            (df.low <= df.low.shift(-2)) 
            &
            (df.low < df.low.shift(3)) 
            &
            (df.low <= df.low.shift(-3)) 
            &
            (df.low < df.low.shift(4)) 
            &
            (df.low <= df.low.shift(-4)) 
            &
            (df.low < df.low.shift(5)) 
            &
            (df.low <= df.low.shift(-5))
            ].index
    df.loc[small_lows, 'low_peak'] = 'small'

    # Now add some shift values to find bigger swings
    big_highs = df[
            (df.high > df.high.shift(1)) 
            &
            (df.high >= df.high.shift(-1)) 
            &
            (df.high > df.high.shift(2)) 
            &
            (df.high >= df.high.shift(-2)) 
            &
            (df.high > df.high.shift(3)) 
            &
            (df.high >= df.high.shift(-3)) 
            &
            (df.high > df.high.shift(4)) 
            &
            (df.high >= df.high.shift(-4)) 
            &
            (df.high > df.high.shift(5)) 
            &
            (df.high >= df.high.shift(-5)) 
            &
            (df.high > df.high.shift(6)) 
            &
            (df.high >= df.high.shift(-6)) 
            &
            (df.high > df.high.shift(7)) 
            &
            (df.high >= df.high.shift(-7)) 
            &
            (df.high > df.high.shift(8)) 
            &
            (df.high >= df.high.shift(-8)) 
            &
            (df.high > df.high.shift(9)) 
            &
            (df.high >= df.high.shift(-9)) 
            &
            (df.high > df.high.shift(10)) 
            &
            (df.high >= df.high.shift(-10))  
            &
            (df.high > df.high.shift(11)) 
            &
            (df.high >= df.high.shift(-11)) 
            &
            (df.high > df.high.shift(12)) 
            &
            (df.high >= df.high.shift(-12)) 
            ].index
    df.loc[big_highs, 'high_peak'] = 'big'

    big_lows = df[
            (df.low < df.low.shift(1)) 
            &
            (df.low <= df.low.shift(-1)) 
            &
            (df.low < df.low.shift(2)) 
            &
            (df.low <= df.low.shift(-2)) 
            &
            (df.low < df.low.shift(3)) 
            &
            (df.low <= df.low.shift(-3)) 
            &
            (df.low < df.low.shift(4)) 
            &
            (df.low <= df.low.shift(-4)) 
            &
            (df.low < df.low.shift(5)) 
            &
            (df.low <= df.low.shift(-5)) 
            &
            (df.low < df.low.shift(6)) 
            &
            (df.low <= df.low.shift(-6)) 
            &
            (df.low < df.low.shift(7)) 
            &
            (df.low <= df.low.shift(-7)) 
            &
            (df.low < df.low.shift(8)) 
            &
            (df.low <= df.low.shift(-8)) 
            &
            (df.low < df.low.shift(9)) 
            &
            (df.low <= df.low.shift(-9)) 
            &
            (df.low < df.low.shift(10)) 
            &
            (df.low <= df.low.shift(-10))  
            &
            (df.low < df.low.shift(11)) 
            &
            (df.low <= df.low.shift(-11)) 
            &
            (df.low < df.low.shift(12)) 
            &
            (df.low <= df.low.shift(-12)) 
            ].index
    df.loc[big_lows, 'low_peak'] = 'big'

def set_atr(df, n=14):
    data = pd.DataFrame()
    data['tr0'] = abs(df.high - df.low)
    data['tr1'] = abs(df.high - df.close.shift())
    data['tr2'] = abs(df.low - df.close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean() 
    df['atr'] = atr

def find_thrusts(df: pd.DataFrame, roll_window:int = 1) -> None:
    ''' Identify periods where there has been an impulsive price move,
    either up or down, and locate both the start and end of that move.'''

    # I need separate thrust cols so that a single peak can be both a 
    # "end up" and a "start down"
    df['thrust_up'] = np.nan
    df['thrust_down'] = np.nan
    
    up_thrusts = df[
        (df.close.diff().rolling(2).mean() >= df.close.diff().rolling(40).mean() * roll_window)
        &
        (df.close.diff().rolling(2).mean() >= df.atr.shift(2).rolling(40).mean())
        ].index
    df.loc[up_thrusts, 'thrust_up'] = 'up'

    down_thrusts = df[
        (df.close.diff().rolling(2).mean() <= df.close.diff().rolling(40).mean() * roll_window)
        &
        (df.close.diff().rolling(2).mean() <= -1 * df.atr.shift(2).rolling(40).mean())
        ].index
    df.loc[down_thrusts, 'thrust_down'] = 'down'

    # print(df.thrust_up.notna().sum())
    # print(df.thrust_down.notna().sum())

    # Now locate the actual start and end points of those thrust periods by looking
    # for nearby peaks.  For a down thrust, the start peak will be some number of
    # rows prior to the thrust occuring. The end peak could occur somewhere in
    # the thrust period or afterwards. 

    up_starts = df.thrust_up[
            (   
                (df.low_peak == 'small')
                |
                (df.low_peak == 'big')
            )
            &
            (
                (df.thrust_up == 'up') 
                |
                (df.thrust_up.shift(-1) == 'up') # look forwards for the thrust
                |
                (df.thrust_up.shift(-2) == 'up') 
                |
                (df.thrust_up.shift(-3) == 'up') 
                |
                (df.thrust_up.shift(-4) == 'up')
                |
                (df.thrust_up.shift(-5) == 'up') 
                |
                (df.thrust_up.shift(-6) == 'up')
                |
                (df.thrust_up.shift(-7) == 'up')
                |
                (df.thrust_up.shift(-8) == 'up')
                |
                (df.thrust_up.shift(-9) == 'up')
            )
            ].index
    df.loc[up_starts, 'thrust_up'] = 'start'

    # Finding the end will be essentially the same but I'll be looking for the opposite
    # type of peak, and in the opposite direction time-wise

    up_ends = df[
            (   
                (df.high_peak == 'small')
                |
                (df.high_peak == 'big')
            )
            &
            (
                (df.thrust_up == 'up') 
                |
                (df.thrust_up.shift(1) == 'up') # look backwards for the thrust
                |
                (df.thrust_up.shift(2) == 'up') 
                |
                (df.thrust_up.shift(3) == 'up') 
                |
                (df.thrust_up.shift(4) == 'up')
                |
                (df.thrust_up.shift(5) == 'up') 
                |
                (df.thrust_up.shift(6) == 'up')
                |
                (df.thrust_up.shift(7) == 'up')
                |
                (df.thrust_up.shift(8) == 'up')
                |
                (df.thrust_up.shift(9) == 'up')
            )
            ].index
    df.loc[up_ends, 'thrust_up'] = 'end'

    # Now do the down thrusts
    down_starts = df[
            (   
                (df.high_peak == 'small')
                |
                (df.high_peak == 'big')
            )
            &
            (
                (df.thrust_down == 'down') 
                |
                (df.thrust_down.shift(-1) == 'down') 
                |
                (df.thrust_down.shift(-2) == 'down') 
                |
                (df.thrust_down.shift(-3) == 'down') 
                |
                (df.thrust_down.shift(-4) == 'down')
                |
                (df.thrust_down.shift(-5) == 'down') 
                |
                (df.thrust_down.shift(-6) == 'down')
                |
                (df.thrust_down.shift(-7) == 'down')
                |
                (df.thrust_down.shift(-8) == 'down') 
                |
                (df.thrust_down.shift(-9) == 'down')
            )
            ].index
    df.loc[down_starts, 'thrust_down'] = 'start'

    down_ends = df[
            (   
                (df.low_peak == 'small')
                |
                (df.low_peak == 'big')
            )
            &
            (
                (df.thrust_down == 'down') 
                |
                (df.thrust_down.shift(1) == 'down') 
                |
                (df.thrust_down.shift(2) == 'down') 
                |
                (df.thrust_down.shift(3) == 'down') 
                |
                (df.thrust_down.shift(4) == 'down')
                |
                (df.thrust_down.shift(5) == 'down') 
                |
                (df.thrust_down.shift(6) == 'down')
                |
                (df.thrust_down.shift(7) == 'down')
                |
                (df.thrust_down.shift(8) == 'down') 
                |
                (df.thrust_down.shift(9) == 'down')
            )
            ].index            
    df.loc[down_ends, 'thrust_down'] = 'end'

def find_brl_levels(df: pd.DataFrame, lookback:int = 10, extension:int = 100) -> None:
    ''' This will locate SR levels where a fast and fairly large price
    movement has happened well beyond a previous peak. The row which 
    the BRL data will get saved to will be a thrust's "end", this will 
    make backtesting safe as the level will only appear after the 
    thrust which validated it.
    Note: atr() and find_thrusts() needs to be called first.''' 

    # I need to identify each unique thrust by getting start and end points,
    # then get the 50% level between those prices and use that to then find
    # peaks which have occured in the recent past, on a certain side of that 50%

    df['level_type'] = np.nan
    df['level_direction'] = np.nan
    df['level_start'] = np.nan
    df['level_end'] = np.nan
    df['level_plot_price'] = np.nan
    df['level_min_price'] = np.nan
    df['level_max_price'] = np.nan


    up_end_idxs = df[df.thrust_up == 'end'].index
    down_end_idxs = df[df.thrust_down == 'end'].index

    #   -- FIND 50% PRICES --

    # Iter thru the ends to find the nearest previous "start." Using the high
    # and low of those indexes, split the difference to find the 50% level.
    for end_idx in up_end_idxs:
        start_idx = df[
                    (df.thrust_up == 'start')
                    &
                    (df.index < end_idx)
                    ].index.max()

        # Set the mid price between start and end
        # If no start was found because you're at the df bounds, just skip it
        if pd.isnull(start_idx):
            continue
        df.loc[end_idx, 'thrust_50'] = df.loc[end_idx, 'high'] - (df.loc[end_idx, 'high'] - df.loc[start_idx, 'low']) / 2

        #   -- FIND MAXIMIUM LOOK BACK --

        # The price range in which BRLs will be searched for is between the start of a thrust
        # and the 50% level. Now I also need to get the time range to know how much historical data to scan

        atr_high_to_50 = (df.loc[end_idx, 'high'] - df.loc[end_idx, 'thrust_50']) / df.loc[end_idx, 'atr']
        historical_limit = round(abs(atr_high_to_50) * lookback)
        historical_limit *=  df.index[1] - df.index[0]  # historical limit converted to num rows
        # Now find a peak (BRL) that exists between the start 'price' and 50, 
        # and start 'time' and historical_limit
        brl_idxs = df[
                    (df.high_peak.notna())                      # BRL peak type
                    &
                    (df.high < df.loc[end_idx, 'thrust_50'])    # price < than 50 price
                    &
                    (df.high > df.loc[start_idx, 'low'])        # price > than thrust start
                    &
                    (df.index < start_idx)                      # exists before thrust start
                    &
                    (df.index > start_idx - historical_limit)   # exists after historical limit   
                    # &
                    # (df.high == df.high.rolling(20).max())  # << trying this out cuz I CANNOT FUCKING GET BAD ONES TO GO AWAY  
                    ].index
                    
        #   -- SET BRLs --
        # Before I confirm a BRL I need to verify its the highest high between itself
        # and the start.  If confirmed I'll save the idx_loc of the actual BRL peaks,
        # the price of the BRL (to be used for plotting) and the price threshold (to be
        # used for activating trade scanning)
        for brl_idx in brl_idxs:
            if df.loc[brl_idx, 'high'] == df.loc[brl_idx : start_idx, 'high'].max():
                df.loc[end_idx, 'level_type'] = 'brl'
                df.loc[end_idx, 'level_direction'] = 'buy'
                df.loc[end_idx, 'level_start'] = end_idx  # trying this out, might make queries later on more clear
                df.loc[end_idx, 'level_plot_price'] = df.loc[brl_idx, 'high']
                df.loc[end_idx, 'level_upper_price'] = df.loc[brl_idx, 'high'] + df.atr.mean()
                df.loc[end_idx, 'level_lower_price'] = df.loc[brl_idx, 'high'] - df.atr.mean()

    #   -- SET TERMINATION POINTS  -- 

    # The last thing to do is to find the termination points of a BRL. Once price
    # has pierced a level by some amount I will want that level to become inactive.
    # Price needs to not only pierce by some distance but also for multiple candles
    for i in df[(df.level_type == 'brl') & (df.level_direction == 'buy')].index:
        index_loc = df[
                    (df.close < (df.loc[i, 'level_plot_price'] - df.loc[i, 'atr'] * 1))
                    &
                    (df.index > i)
                    ].index
        df.loc[i, 'level_end'] = index_loc[2] if len(index_loc) > 3 else i + (df.index[1] - df.index[0]) * extension
        # I set index_loc to 2 so that visual plots will be accurate with what happened live
    

    # Now all the same thing but for the lows
    
    #   -- FIND 50% PRICES --
    
    for end_idx in down_end_idxs:
        start_idx = df[
                    (df.thrust_down == 'start')
                    &
                    (df.index < end_idx)
                    ].index.max()

        if pd.isnull(start_idx):
            continue
        df.loc[end_idx, 'thrust_50'] = df.loc[start_idx, 'high'] - (df.loc[start_idx, 'high'] - df.loc[end_idx, 'low']) / 1.6

        #   -- FIND MAXIMIUM LOOK BACK --
        
        atr_high_to_50 = (df.loc[start_idx, 'high'] - df.loc[end_idx, 'thrust_50']) / df.loc[end_idx, 'atr']
        historical_limit = round(abs(atr_high_to_50) * lookback)
        historical_limit *=  df.index[1] - df.index[0]  # historical limit converted to num rows
        
        brl_idxs = df[
                (df.low_peak.notna())                       # BRL peak type
                &
                (df.low > df.loc[end_idx, 'thrust_50'])    # price > than 50 price
                &
                (df.low < df.loc[start_idx, 'high'])        # price < than thrust start
                &
                (df.index < start_idx)                      # exists before thrust start
                &
                (df.index > start_idx - historical_limit)   # exists after historical limit
                # &
                # (df.low == df.low.rolling(20).min())      
                ].index
                    
        #   -- SET BRLs --
        for brl_idx in brl_idxs:
            if df.loc[brl_idx, 'low'] == df.loc[brl_idx : start_idx, 'low'].min():
                # df.loc[end_idx, 'brl_sell_idx_loc'] = brl_idx
                df.loc[end_idx, 'level_type'] = 'brl'
                df.loc[end_idx, 'level_direction'] = 'sell'
                df.loc[end_idx, 'level_start'] = end_idx  # trying this out, might make queries later on more clear
                df.loc[end_idx, 'level_plot_price'] = df.loc[brl_idx, 'low']
                df.loc[end_idx, 'level_upper_price'] = df.loc[brl_idx, 'low'] + df.atr.mean()
                df.loc[end_idx, 'level_lower_price'] = df.loc[brl_idx, 'low'] - df.atr.mean()
    
    #   -- SET TERMINATION POINTS  -- 

    for i in df[(df.level_type == 'brl') & (df.level_direction == 'sell')].index:
        index_loc = df[
                    (df.close > (df.loc[i, 'level_plot_price'] + df.loc[i, 'atr'] * 1))
                    &
                    (df.index > i)
                    ].index
        df.loc[i, 'level_end'] = index_loc[2] if len(index_loc) > 3 else i + (df.index[1] - df.index[0]) * extension

def find_sd_zones(df, extension=200):
    ''' sup w/ dem zones'''

    df['level_type'] = np.nan
    df['level_direction'] = np.nan

    rows = df.index[1] - df.index[0]   

    # Demand
    for i in df.index[df.thrust_up == 'start']:
        next_high = df[

            # A strong thrust needs to be made within n bars 
            (df.index > i)
            &
            (df.index < i + 10 * rows)
            &
            (df.high > df.loc[i, 'low'] + df.loc[i, 'atr'] * 4)
        ]

        # Save the info which will be used for htf level proximity scanning
        if not next_high.empty:
            df.loc[i, 'level_start'] = i
            df.loc[i, 'level_end'] = i + extension * rows
            df.loc[i, 'level_plot_price'] = df.loc[i, 'low']
            df.loc[i, 'level_upper_price'] = df.loc[i, 'low'] + df.loc[i, 'atr'] * 1.5
            df.loc[i, 'level_lower_price'] = df.loc[i, 'low'] - df.loc[i, 'atr'] * 0.5
            df.loc[i, 'level_direction'] = 'buy'
            df.loc[i, 'level_type'] = 'supdem'

    # Supply
    for i in df.index[df.thrust_down == 'start']:
        next_low = df[
            (df.index > i)
            &
            (df.index < i + 10 * rows)
            &
            (df.low < df.loc[i, 'high'] - df.loc[i, 'atr'] * 4)
        ]
        if not next_low.empty:
            df.loc[i, 'level_start'] = i
            df.loc[i, 'level_end'] = i + extension * rows
            df.loc[i, 'level_plot_price'] = df.loc[i, 'high']
            df.loc[i, 'level_upper_price'] = df.loc[i, 'high'] + df.loc[i, 'atr'] * 0.5
            df.loc[i, 'level_lower_price'] = df.loc[i, 'high'] - df.loc[i, 'atr'] * 1.5
            df.loc[i, 'level_direction'] = 'sell'
            df.loc[i, 'level_type'] = 'supdem'
    

    # Now adjust termination points for zones that get taken out
    for i in df[(df.level_type == 'supdem') & (df.level_direction == 'buy')].index:
        index_loc = df[
                    (df.close < (df.loc[i, 'level_plot_price'] - df.loc[i, 'atr'] * 0.5))
                    &
                    (df.index > i)
                    ].index
        df.loc[i, 'level_end'] = index_loc[2] if len(index_loc) > 3 else i + rows * extension

    for i in df[(df.level_type == 'supdem') & (df.level_direction == 'sell')].index:
        index_loc = df[
                    (df.close > (df.loc[i, 'level_plot_price'] + df.loc[i, 'atr'] * 0.5))
                    &
                    (df.index > i)
                    ].index
        df.loc[i, 'level_end'] = index_loc[2] if len(index_loc) > 3 else i + rows * extension

def find_accumulations(df: pd.DataFrame, lookback:int = 10) -> None:
    '''   '''

    # General variables used for both buy and sell patterns
    atr = df.atr.rolling(lookback).mean()
    high_slope = df.high.diff().rolling(lookback).mean() / atr
    low_slope = df.low.diff().rolling(lookback).mean() / atr
    recent_avg_low_price = df.low.rolling(lookback).mean() 
    recent_avg_high_price = df.high.rolling(lookback *2).mean() 
    long_term_avg_price = df.close.rolling(lookback * 5).mean() 
    close_slope = df.close.diff().rolling(lookback * 3).mean() / atr
    
    print(high_slope.describe())
    print(low_slope.describe())

    #   --- BUY PATTERN ---
    buy_no_vol = df[
        (low_slope < low_slope.shift(lookback))
        &
        (high_slope > high_slope.shift(lookback))
        &
        (low_slope < -0.02)
        &
        (high_slope - low_slope > (high_slope - low_slope).shift(lookback))
        &
        (df.low < recent_avg_low_price)
        &
        (df.low < long_term_avg_price)
        &
        (close_slope > close_slope.shift(lookback))
        &
        (close_slope.shift(lookback) < 0)
        ].index
    
    if len(buy_no_vol) < 1:
        print(len(buy_no_vol))
        df.loc[df.index.min(), 'buys'] = df.loc[df.index.min(), 'close']
    print(len(buy_no_vol))
    df.loc[buy_no_vol, 'buys'] = df.loc[buy_no_vol, 'close']

    sell_no_vol = df[
        (low_slope < low_slope.shift(lookback))
        &
        (high_slope < high_slope.shift(lookback))
        &
        (high_slope > 0.02)
        &
        (high_slope - low_slope > (high_slope - low_slope).shift(lookback))
        &
        (df.high > recent_avg_high_price)
        &
        (df.high > long_term_avg_price)
        &
        (close_slope < close_slope.shift(lookback))
        &
        (close_slope.shift(lookback) > 0)
        ].index


    if len(sell_no_vol) < 1:
        print(len(sell_no_vol))
        df.loc[df.index.min(), 'sells'] = df.loc[df.index.min(), 'close']
    print(len(sell_no_vol))
    df.loc[sell_no_vol, 'sells'] = df.loc[sell_no_vol, 'close']

def save_levels_data(htf_levels, df, symbol, timeframe):

    # save any levels to the brl_df
    new_levels = pd.DataFrame() 
    for level_type in df.level_type[df.level_type.notna()].unique():
        for level_direction in df.level_direction[df.level_direction.notna()].unique():
            df_levels = df[
                            (df.level_type == level_type) 
                            & 
                            (df.level_direction == level_direction)
                        ]

            new_levels['start'] = df_levels.level_start
            new_levels['end'] = df_levels.level_end
            new_levels['plot_price'] = df_levels.level_plot_price
            new_levels['upper_price'] = df_levels.level_upper_price
            new_levels['lower_price'] = df_levels.level_lower_price
            new_levels['symbol'] = symbol
            new_levels['timeframe'] = timeframe
            new_levels['direction'] = level_direction
            new_levels['type'] = level_type

            # Append the levels and clear then erase them
            htf_levels = htf_levels.append(new_levels)
            new_levels = new_levels[0:0]

    # Ensure this is sent out with unique index values
    return htf_levels.reset_index(drop=True)

def init_sr_levels(symbols:list=mt5_symbols['majors'], timeframes:list=None) -> pd.DataFrame:
    ''' Upon program start, fill the data for all HTF levels.
    This would normally not happen till htf candles close.'''

    # Add in the htf's so they will get scanned
    all_tfs = timeframes.copy()
    for tf in timeframes:
        htfs = etf_to_htf.get(tf, None)
        if htfs:
            if htfs[0] not in timeframes:
                all_tfs.append(htfs[0])
            if htfs[1] not in timeframes:
                all_tfs.append(htfs[1])

    htf_levels = pd.DataFrame()

    if symbols == indexes:
        for symbol in symbols:
            for timeframe in timeframes:
                df = pd.read_parquet(Path(INDEXES_PATH, f'temp_{symbol}.parquet'))
                df = _resample(df, timeframe)

                # If resampling too high it will return one bar
                if len(df) > 1:
                    format(df)
                    htf_levels = save_levels_data(htf_levels, df, symbol, timeframe)
        return htf_levels

    for symbol in symbols:
        for timeframe in timeframes:
            df = mt5_ohlc_request(symbol, timeframe, num_candles=350)
            format(df)
            htf_levels = save_levels_data(htf_levels, df, symbol, timeframe)

    return htf_levels


#### TRADE SETUPS ######
def flying_buddha(df, rrr:int = 5):
    ''' long term down trend with fast pullback. Im intending 
    these signals to act as triggers to allow for LTF reversal trades'''
    
    # Main entry filters 
    buys = df[
        (df.high < df.close.ewm(span=4).mean()) 
        # &
        # (df.low < df.low.shift(1).rolling(12).min())
        &
        (df.close.ewm(span=4).mean() < df.close.ewm(span=30).mean()) 
        &
        (df.close.ewm(span=4).mean() - df.close.shift(5).ewm(span=4).mean() > df.atr) 
       # &
        # This is just to filter out signals that wouldn't get filled
        #(df.high.shift(-1) > df.high)
        ]

    sells = df[
        (df.low > df.close.ewm(span=4).mean()) 
        # &
        # (df.low < df.low.shift(1).rolling(12).min())
        &
        (df.close.ewm(span=4).mean() > df.close.ewm(span=30).mean()) 
        &
        (df.close.shift(5).ewm(span=30).mean() - df.close.ewm(span=4).mean() > df.atr) 
       # &
        # This is just to filter out signals that wouldn't get filled
        #(df.high.shift(-1) > df.high)
        ]

    if not buys.empty:
        df.loc[buys.index, 'pattern'] = 'flying_buddha'  
        df.loc[buys.index, 'buy_entry'] = df.loc[buys.index, 'high']
        df.loc[buys.index, 'sl'] = df.loc[buys.index, 'low']  
        df.loc[buys.index, 'tp'] = df.loc[buys.index, 'buy_entry'] + rrr * (df.loc[buys.index, 'buy_entry'] - df.loc[buys.index, 'sl'])  
    if not sells.empty:
        df.loc[sells.index, 'pattern'] = 'flying_buddha'  
        df.loc[sells.index, 'sell_entry'] = df.loc[sells.index, 'low']
        df.loc[sells.index, 'sl'] = df.loc[sells.index, 'high']  
        df.loc[sells.index, 'tp'] = df.loc[sells.index, 'sell_entry'] - rrr * (df.loc[sells.index, 'sl'] - df.loc[sells.index, 'sell_entry'])

def trade_spring(df, rrr:int = 5):  
    ''' Low volume springs in an up trend. '''

    # Main entry filters 
    buys = df[
        (df.bar_type.shift(1) == 'down')
        &
        (df.close > df.open.shift(1))
        &
        (df.thrust_up.shift(2).notna().rolling(15).sum() > 0)
        &
        # This is just to filter out signals that wouldn't get filled
        (df.high.shift(-1) > df.high)
        ]

    sells = df[
        (df.bar_type.shift(1) == 'up')
        &
        (df.close < df.open.shift(1))
        &
        (df.thrust_down.shift(2).notna().rolling(15).sum() > 0)
        &
        # This is just to filter out signals that wouldn't get filled
        (df.low.shift(-1) < df.low)
        ]
    if not buys.empty:
        df.loc[buys.index, 'pattern'] = 'spring'  
        df.loc[buys.index, 'buy_entry'] = df.loc[buys.index, 'high']
        df.loc[buys.index, 'sl'] = df.loc[buys.index, 'low']  
        df.loc[buys.index, 'tp'] = df.loc[buys.index, 'buy_entry'] + rrr * (df.loc[buys.index, 'buy_entry'] - df.loc[buys.index, 'sl'])  
    if not sells.empty:
        df.loc[sells.index, 'pattern'] = 'spring'  
        df.loc[sells.index, 'sell_entry'] = df.loc[sells.index, 'low']
        df.loc[sells.index, 'sl'] = df.loc[sells.index, 'high']  
        df.loc[sells.index, 'tp'] = df.loc[sells.index, 'sell_entry'] - rrr * (df.loc[sells.index, 'sl'] - df.loc[sells.index, 'sell_entry'])

def trade_spring_new(df, rrr:int = 5):  
    ''' Low volume springs in an up trend. '''

    # Main entry filters 
    buys = df[
        (df.bar_type == 'down')
        &
        (df.bar_type.shift(1) == 'up')
        &
        (df.low > df.low.shift(1))
        &
        (df.close > df.high.shift(1))
        &
        (df.thrust_up.shift(2).notna().rolling(15).sum() > 0)
        # &
        # This is just to filter out signals that wouldn't get filled
        # (df.high.shift(-1) > df.high)
        ]

    sells = df[
        (df.bar_type == 'up')
        &
        (df.bar_type.shift(1) == 'down')
        &
        (df.high < df.high.shift(1))
        &
        (df.thrust_down.shift(2).notna().rolling(15).sum() > 0)
        # &
        # This is just to filter out signals that wouldn't get filled
        # (df.low.shift(-1) < df.low)
        ]
    if not buys.empty:
        df.loc[buys.index, 'pattern'] = 'spring'  
        df.loc[buys.index, 'buy_entry'] = df.loc[buys.index, 'high']
        df.loc[buys.index, 'sl'] = df.loc[buys.index, 'low']  
        df.loc[buys.index, 'tp'] = df.loc[buys.index, 'buy_entry'] + rrr * (df.loc[buys.index, 'buy_entry'] - df.loc[buys.index, 'sl'])  
    if not sells.empty:
        df.loc[sells.index, 'pattern'] = 'spring'  
        df.loc[sells.index, 'sell_entry'] = df.loc[sells.index, 'low']
        df.loc[sells.index, 'sl'] = df.loc[sells.index, 'high']  
        df.loc[sells.index, 'tp'] = df.loc[sells.index, 'sell_entry'] - rrr * (df.loc[sells.index, 'sl'] - df.loc[sells.index, 'sell_entry'])

def trade_reversal(df, rrr=5):
    ''' this is currently my best code.
    6 week back test: 256 trades, 284R w00t
    
    it include the price_thresh section 
    has roll win = 20 and half win = 10
    and no trend filtering'''

    roll_win = 20
    half_win = 10
    # These thresh values are meant to target the recent peak
    max_price_thresh = df.low.shift(half_win).rolling(roll_win).min() + (df.atr.rolling(roll_win).mean() * 1)
    min_price_thresh = df.low.shift(half_win).rolling(roll_win).min() - (df.atr.rolling(roll_win).mean() * 2)
    # This one is a safety measure in case that peak was breached alot, I dont wanna be too far away from the lowest low
    max_dist_from_lows = df.low.shift(1).rolling(half_win).min() + (df.atr.rolling(roll_win).mean() * 1)
    # I need a lowest to highest peak size filter. to ensure theres been  a decent bounce.
    # Or just like a diff filter based on atr
    # Buys
    buys = df[
        # Buyers have already appeared making an up-thrust (this one Im not sure about)
        (df.thrust_up.notna().rolling(12).sum() > 0)   
        &        
        # (df.thrust_down.notna().rolling(12).sum() > 0)   
        # &
        # At least one previous peak already exists nearby 
        # (need a shift value to not cheat)
        # should prob increase peak size and increase this window 
        (df.low_peak.shift(5).notna().rolling(roll_win).sum() > 0)
        &
        # This high peak will only exist if enough of a bounce occured off the low peak
        (df.high_peak.shift(5).notna().rolling(roll_win).sum() > 0)
        &
        # Price is low, but not too low
        (df.close < max_price_thresh)
        &
        (df.close > min_price_thresh)
        &
        (df.close < max_dist_from_lows)
        &
        # Impulse into HTF BRL (this was something that I cut out after meeting mike but I think is actually good)
        (df.close.diff().rolling(3).mean() < df.close.diff().rolling(roll_win).mean())
        &
        # Looking for a big bar (this will work for both buy and sell setups)
        (abs(df.open - df.close) > abs(df.open - df.close).rolling(roll_win).mean())
        &
        (
            (df.bar_type == 'down')
            |
            (df.pinbar == 'up')
        )
        # Backtest prefilter. Comment this out for live trading
        # &
        # (df.high.shift(-1) > df.high)
    ]

    ### Sells ###
    max_price_thresh = df.high.shift(half_win).rolling(roll_win).max() + (df.atr.rolling(roll_win).mean() * 2)
    min_price_thresh = df.high.shift(half_win).rolling(roll_win).max() - (df.atr.rolling(roll_win).mean() * 1)
    max_dist_from_highs = df.high.shift(1).rolling(half_win).max() - (df.atr.rolling(roll_win).mean() * 1)

    # Sells
    sells = df[
        
        (df.thrust_down.notna().rolling(12).sum() > 0)   
        # &
        # (df.thrust_up.notna().rolling(12).sum() > 0  )
        &
        (df.high_peak.shift(5).notna().rolling(roll_win).sum() > 0)
        &
        (df.low_peak.shift(5).notna().rolling(roll_win).sum() > 0)
        &
        (df.close > max_dist_from_highs)
        &
        (df.close < max_price_thresh)
        &
        (df.close > min_price_thresh)
        &
        # Impulse into HTF BRL (this was something that I cut out after meeting mike but I think is actually good)
        (df.close.diff().rolling(3).mean() > df.close.diff().rolling(roll_win).mean())
        &
        # Looking for a big bar (this will work for both buy and sell setups)
        (abs(df.open - df.close) > abs(df.open - df.close).rolling(roll_win).mean())
        &
        (
            (df.bar_type == 'up')
            |
            (df.pinbar == 'down')
        )
        # Backtest prefilter. Comment this out for live trading
        # &
        # (df.low.shift(-1) < df.low)
    ]
    
    if not buys.empty:
        df.loc[buys.index, 'pattern'] = 'my_reversal'  
        df.loc[buys.index, 'buy_entry'] = df.loc[buys.index, 'high']
        df.loc[buys.index, 'sl'] = df.loc[buys.index, 'low']  
        df.loc[buys.index, 'tp'] = df.loc[buys.index, 'buy_entry'] + rrr * (df.loc[buys.index, 'buy_entry'] - df.loc[buys.index, 'sl'])  
    if not sells.empty:
        df.loc[sells.index, 'pattern'] = 'my_reversal'  
        df.loc[sells.index, 'sell_entry'] = df.loc[sells.index, 'low']
        df.loc[sells.index, 'sl'] = df.loc[sells.index, 'high']  
        df.loc[sells.index, 'tp'] = df.loc[sells.index, 'sell_entry'] - rrr * (df.loc[sells.index, 'sl'] - df.loc[sells.index, 'sell_entry'])

def trade_simple_volume(df, rrr:int = 5):
    ''' high vol up bar, low vol down bar, high vol up bar'''


    buys = df[
        (df.bar_type.shift(2) == 'up')
        &
        (df.bar_type.shift(1) == 'down')
        &
        (df.bar_type == 'up')
        &
        (df.low.shift(1) > df.low.shift(2))

        # &
        # (df.high.shift(-1) > df.high)
    ]

    sells = df[
        (df.bar_type.shift(2) == 'down')
        &
        (df.bar_type.shift(1) == 'up')
        &
        (df.bar_type == 'down')
        &
        (df.high.shift(1) < df.high.shift(2))

        # &
        # (df.low.shift(-1) < df.low)
    ]

    if not buys.empty:
        df.loc[buys.index, 'pattern'] = 'simple_volume'  
        df.loc[buys.index, 'buy_entry'] = df.loc[buys.index, 'high']
        df.loc[buys.index, 'sl'] = df.loc[buys.index, 'low']  
        df.loc[buys.index, 'tp'] = df.loc[buys.index, 'buy_entry'] + rrr * (df.loc[buys.index, 'buy_entry'] - df.loc[buys.index, 'sl'])  
    if not sells.empty:
        df.loc[sells.index, 'pattern'] = 'simple_volume'  
        df.loc[sells.index, 'sell_entry'] = df.loc[sells.index, 'low']
        df.loc[sells.index, 'sl'] = df.loc[sells.index, 'high']  
        df.loc[sells.index, 'tp'] = df.loc[sells.index, 'sell_entry'] - rrr * (df.loc[sells.index, 'sl'] - df.loc[sells.index, 'sell_entry'])

    # # Filter for levels
    # filter_level_proximity(sr_levels, df, symbol, timeframe)

    # # Once those root positions have been set, iter thru looking at a fwd window
    # row = df.index[1] - df.index[0]
    # for i in df.index[df.buy_entry.notna()]:
    #     temp = df[i: i + 6 * row]
    
    #     buys = temp[
    #         (temp.bar_type.shift(2) == 'up')
    #         &
    #         (temp.bar_type.shift(1) == 'down')
    #         &
    #         (temp.bar_type == 'up')
    #         &
    #         (temp.low.shift(1) > temp.low.shift(2))

    #         &
    #         (temp.high.shift(-1) > temp.high)
    #     ]

    #     if not buys.empty:
    #         df.loc[buys.index, 'pattern'] = 'simple_volume'  
    #         df.loc[buys.index, 'buy_entry'] = df.loc[buys.index, 'high']
    #         df.loc[buys.index, 'sl'] = df.loc[buys.index, 'low']  
    #         df.loc[buys.index, 'tp'] = df.loc[buys.index, 'buy_entry'] + rrr * (df.loc[buys.index, 'buy_entry'] - df.loc[buys.index, 'sl'])  
        
    # for i in df.index[df.sell_entry.notna()]:
    #     temp = df[i: i + 6 * row]

    #     sells = temp[
    #         (temp.bar_type.shift(2) == 'down')
    #         &
    #         (temp.bar_type.shift(1) == 'up')
    #         &
    #         (temp.bar_type == 'down')
    #         &
    #         (temp.high.shift(1) < temp.high.shift(2))

    #         &
    #         (temp.low.shift(-1) < temp.low)
    #     ]
    #     if not sells.empty:
    #         df.loc[sells.index, 'pattern'] = 'simple_volume'  
    #         df.loc[sells.index, 'sell_entry'] = df.loc[sells.index, 'low']
    #         df.loc[sells.index, 'sl'] = df.loc[sells.index, 'high']  
    #         df.loc[sells.index, 'tp'] = df.loc[sells.index, 'sell_entry'] - rrr * (df.loc[sells.index, 'sl'] - df.loc[sells.index, 'sell_entry'])


def mike_trade_basic_engulf(df, rrr:int = 5):
    sells = df[
        # Last bar was up
        (df.bar_type.shift(1) == 'up')
        &
        # Signal bar is down
        (df.bar_type == 'down')
        &
        # Signal bar made a HH than previous bar
        (df.high > df.high.shift(1))
        &
        (df.close < df.low.shift(1))
        &
        # MA down
        (df.close < df.close.ewm(span=20).mean())     
        # &
        # This is just to set the signal on bars that would have actually triggered a trade
        # (df.low.shift(-1) < df.low)
    ]

    buys = df[
        (df.bar_type.shift(1) == 'down')
        &
        (df.bar_type == 'up')
        &
        (df.low < df.low.shift(1))
        &
        (df.close > df.high.shift(1))
        &
        (df.close > df.close.ewm(span=20).mean())   
        # &
        # (df.high.shift(-1) > df.high)
    ]

    if not buys.empty:
        df.loc[buys.index, 'pattern'] = 'engulf'  
        df.loc[buys.index, 'buy_entry'] = df.loc[buys.index, 'high']
        df.loc[buys.index, 'sl'] = df.loc[buys.index, 'low']  
        df.loc[buys.index, 'tp'] = df.loc[buys.index, 'buy_entry'] + rrr * (df.loc[buys.index, 'buy_entry'] - df.loc[buys.index, 'sl'])  
    if not sells.empty:
        df.loc[sells.index, 'pattern'] = 'engulf'  
        df.loc[sells.index, 'sell_entry'] = df.loc[sells.index, 'low']
        df.loc[sells.index, 'sl'] = df.loc[sells.index, 'high']  
        df.loc[sells.index, 'tp'] = df.loc[sells.index, 'sell_entry'] - rrr * (df.loc[sells.index, 'sl'] - df.loc[sells.index, 'sell_entry'])

def mike_trade_reversal(df, rrr:int = 5):

    roll_win, half_win = 20,10
    
    sells = df[
        # Some trend stuff
        (df.low < df.close.rolling(5).mean())
        &
        (df.close.rolling(5).mean() < df.close.rolling(50).mean())
        &
        # Last bar was up
        (df.bar_type.shift(1) == 'up')
        &
        # Signal bar is down
        (
            (df.bar_type == 'down')
            |
            (df.pinbar == 'down')
        )
        &
        (
            # Signal bar made a HH than previous bar
            (df.high > df.high.shift(1))
            |
            # Engulf bar
            (df.close < df.low.shift(1))
        )
        # &
        # This is just to set the signal on bars that would have actually triggered a trade
        # (df.low.shift(-1) < df.low)
    ]
    
    buys = df[
        # Some trend stuff
        (df.high > df.close.rolling(5).mean())
        &
        (df.close.rolling(5).mean() > df.close.rolling(50).mean())
        &
        (df.bar_type.shift(1) == 'down')
        &
        (
            (df.bar_type == 'up')
            |
            (df.pinbar == 'up')
        )
        &
        (df.low < df.low.shift(1))
        # &
        # Engulf
        # (df.close > df.high.shift(1))
        # &
        # (df.low_peak.notna().rolling(25).sum() > 0)
        # &
        # (df.high.shift(-1) > df.high)
    ]

    if not buys.empty:
        df.loc[buys.index, 'pattern'] = 'reversal'  
        df.loc[buys.index, 'buy_entry'] = df.loc[buys.index, 'high']
        df.loc[buys.index, 'sl'] = df.loc[buys.index, 'low']  
        df.loc[buys.index, 'tp'] = df.loc[buys.index, 'buy_entry'] + rrr * (df.loc[buys.index, 'buy_entry'] - df.loc[buys.index, 'sl'])  
    if not sells.empty:
        df.loc[sells.index, 'pattern'] = 'reversal'  
        df.loc[sells.index, 'sell_entry'] = df.loc[sells.index, 'low']
        df.loc[sells.index, 'sl'] = df.loc[sells.index, 'high']  
        df.loc[sells.index, 'tp'] = df.loc[sells.index, 'sell_entry'] - rrr * (df.loc[sells.index, 'sl'] - df.loc[sells.index, 'sell_entry']) 

def trade_3_bar(df, rrr:int = 5):

    avg_bar_size = abs(df.close - df.open).rolling(20).mean()
    
    buys = df[
        # MA Up
        (df.close > df.close.ewm(span=20).mean())    
        &
        # last bar big
        (abs(df.close.shift(1) - df.open.shift(1)) > avg_bar_size * 1.5)
        &
        # this bar small
        (abs(df.close - df.open) < avg_bar_size)
        &
        # last bar small upper wick
        (df.high.shift(1) - df.close.shift(1) < (df.high.shift(1) - df.low.shift(1)) * 0.25)
        &
        # last bar up
        (df.bar_type.shift(1) == 'up')
        &
        (df.bar_type == 'down')
        # &
        # (df.high.shift(-1) > df.high)
    ]

    sells = df[
        (df.close < df.close.ewm(span=20).mean())    
        &
        (abs(df.close.shift(1) - df.open.shift(1)) > avg_bar_size * 1.5)
        &
        (abs(df.close - df.open) < avg_bar_size)
        &
        (df.close.shift(1) - df.low.shift(1) < (df.high.shift(1) - df.low.shift(1)) * 0.25)
        &
        (df.bar_type.shift(1) == 'down')
        &
        (df.bar_type == 'up')
        # &
        # (df.low.shift(-1) < df.low)
    ]

    if not buys.empty:
        df.loc[buys.index, 'pattern'] = '3bar'  
        df.loc[buys.index, 'buy_entry'] = df.loc[buys.index, 'high']
        df.loc[buys.index, 'sl'] = df.loc[buys.index, 'low']  
        df.loc[buys.index, 'tp'] = df.loc[buys.index, 'buy_entry'] + rrr * (df.loc[buys.index, 'buy_entry'] - df.loc[buys.index, 'sl'])  
    if not sells.empty:
        df.loc[sells.index, 'pattern'] = '3bar'  
        df.loc[sells.index, 'sell_entry'] = df.loc[sells.index, 'low']
        df.loc[sells.index, 'sl'] = df.loc[sells.index, 'high']  
        df.loc[sells.index, 'tp'] = df.loc[sells.index, 'sell_entry'] - rrr * (df.loc[sells.index, 'sl'] - df.loc[sells.index, 'sell_entry']) 


#### TRADE FILTERS #####

def filter_level_proximity(htf_levels, df, symbol, timeframe):
    '''this identfies closes that are within a htf levels proximity. 
    any trade that is not near a htf level gets deleted. 
    a new column is created in the df to track it because
    another filter function refermces the proximity data '''

    # Skip this function and erase all the trades if this dataframes empty
    if htf_levels.empty:
        df[['buy_entry', 'sell_entry', 'sl', 'tp', 'pattern']] = np.nan
        return

    trade_directions = [
        'buy',
        'sell'
    ]

    htfs = etf_to_htf.get(timeframe, None)
    
    if not htfs:
        return

    htf_1 = htfs[0]
    htf_2 = htfs[1]

    df['level_proximity'] = np.nan

    # i will need to filter the htf_levels based on direvtion
    # and timeframe and then iter thru each one to grab rach rows data
    for direction in trade_directions:
        levels = htf_levels[
            (
                (htf_levels.timeframe == htf_1)
                |
                (htf_levels.timeframe == htf_2)
                # (htf_levels.timeframe == any(htf for htf in etf_to_htf))
            )
            &
            (htf_levels.symbol == symbol)
            &
            (htf_levels.direction == direction)
            ]

        # Erase the trade entries if nothing is found
        if levels.empty:
            if direction == 'buy':
                df.loc[df[df.buy_entry.notna()].index, [
                    'buy_entry', 'sl', 'tp', 'pattern']] = np.nan
            
            if direction == 'sell':
                df.loc[df[df.sell_entry.notna()].index, [
                    'sell_entry', 'sl', 'tp', 'pattern']] = np.nan
            continue 

        # See if any trades in the df match these params
        # accessing rows via index like this is dangerous 

        for row in levels.itertuples(index=None):
            if direction == 'buy':
                buy_trades = df[
                                (df.index > row.start)
                                &
                                (df.index < row.end)
                                &
                                (df.close < row.upper_price)
                                &
                                (df.close > row.lower_price)
                            ].index
                df.loc[buy_trades, 'level_proximity'] = 'buy'

            if direction == 'sell':
                sell_trades = df[
                                (df.index > row.start)
                                &
                                (df.index < row.end)
                                &
                                (df.close < row.upper_price)
                                &
                                (df.close > row.lower_price)
                            ].index
                df.loc[sell_trades, 'level_proximity'] = 'sell'

    # Delete whatever didnt get confirmed with "level_proximity" 
    df.loc[df.index[
            (df.level_proximity != 'buy') 
            & 
            (df.buy_entry.notna())
            ], ['buy_entry', 'sl', 'tp', 'pattern']] = np.nan
        
    df.loc[df.index[
            (df.level_proximity != 'sell') 
            & 
            (df.sell_entry.notna())
            ], ['sell_entry', 'sl', 'tp', 'pattern']] = np.nan

def filter_time_of_day(df,  start_trading:int=23, stop_trading:int=12):
    ''' limit trade signals to fire only between certain hours'''

    no_trading = df[
        (df.index.hour < start_trading)
        &
        (df.index.hour > stop_trading)
    ].index

    df.loc[no_trading, ['sell_entry', 'buy_entry', 'pattern', 'sl', 'tp']] = np.nan

def filter_daily_range(df, symbol, timeframe):
    ''' if price is near the ADR only take reversals'''

    if timeframe == 'D1' or timeframe == 'W1':
        return

    daily = mt5_ohlc_request(symbol, 'D1', num_candles=11)
    adr = (daily.high - daily.low).rolling(10).mean()[-1]

    # Based on the timeframe get a roll value equaling 1 day
    # candles_per_day = df.index[1] - df.index[0] / pd.to_datetime('00:24:00')

    price_range = df.close.diff().rolling(candles_per_day[timeframe]).sum()

    delete_buys = df.index[
            (price_range > adr * 0.9)
            &
            (df.buy_entry.notna())
        ]
    df.loc[delete_buys, 'buy_entry'] = np.nan

    
    delete_sells = df.index[
            (price_range < adr * -0.9)
            &
            (df.sell_entry.notna())
        ]
    df.loc[delete_sells, 'sell_entry'] = np.nan

def filter_bigdaddy_bar(df):
    ''' This doesn't filter all large bars, only those that are trend changing.
    Example: if the trend is moving up and then there is massive down bar, that's
    when I want to filter any signals. And not just on that bar, but also for a few
    bars following 
    
    Edit: it now includes filtering godzilla bars regardless of trend'''

    for shift_val in range(4):

        buys = df[
            # A trade exists
            (df.buy_entry.notna())
            &
            (   # The trend has been up
                (df.close.shift(shift_val).diff().rolling(15).mean() > df.atr.shift(shift_val).rolling(10).mean() * 0.5)
                &
                # Theres a massive down bar
                (df.high.shift(shift_val) - df.close.shift(shift_val) < df.atr.shift(shift_val).rolling(30).mean() * -3)
            )
        ]
        if not buys.empty:
            df.loc[buys.index, 'buy_entry'] = np.nan

        sells = df[
            (df.sell_entry.notna())
            &
            (   
                (df.close.shift(shift_val).diff().rolling(15).mean() < df.atr.shift(shift_val).rolling(10).mean() * -0.5)
                &
                (df.close.shift(shift_val) - df.low.shift(shift_val) > df.atr.shift(shift_val).rolling(30).mean() * 3)
            )
        ]
        if not sells.empty:
            df.loc[sells.index, 'sell_entry'] = np.nan

        # The godzilla bar (open to close > 4 ATR)
        godzilla = df[
            (
                (df.buy_entry.notna())
                |
                (df.sell_entry.notna())
            )
            &
            (abs(df.close.shift(shift_val) - df.open.shift(shift_val)) > df.atr.shift(1 + shift_val).rolling(30).mean() * 4)
        ] 
        if not godzilla.empty:
            df.loc[godzilla.index, ['buy_entry', 'sell_entry']] = np.nan

def timeframes_to_request() -> list:
    ''' This returns a list of timeframes like 'H1', 'M5' '''

    minute = datetime.now().minute
    hour = datetime.now().hour
    timeframes = []

    # top of the hour check hourly timeframes
    if minute == 0:
        for tf in etf_to_htf:
            if hour % etf_to_int[tf] == 0:
                timeframes.append(tf)
        return timeframes

    # else minutes
    for tf in etf_to_htf:
        if minute % etf_to_int[tf] == 0:
            timeframes.append(tf)
    return timeframes


def format(df):

    df['buy_entry'] = np.nan
    df['sell_entry'] = np.nan 
    df['pattern'] = np.nan 
    df['sl'] = np.nan
    df['tp'] = np.nan

    set_atr(df)
    set_bar_type(df)
    pinbar(df)
    find_simple_peaks(df)
    find_thrusts(df)
    find_brl_levels(df, 30)
    find_sd_zones(df)

 
# bot.send_photo(chat_id=446051969, photo=open(pathlib.Path(r'C:\Users\ru'+ f'\EURUSD_H1.png'), 'rb'))
#bot.send_message(chat_id=446051969, text='test')
def trade_3_candle(df, rrr:int = 3):

    buys = df[
        # Some trend stuff
        (df.high > df.close.rolling(5).mean())
        &
        (df.close.rolling(5).mean() > df.close.rolling(50).mean())
        &
        (df.bar_type.shift(1) == 'up')
        &
        (df.bar_type == 'down')
        &
        (df.low > df.low.shift(1))
        &
        (df.high.shift(-1) > df.high)
    ]

    sells = df[
        # Some trend stuff
        (df.low < df.close.rolling(5).mean())
        &
        (df.close.rolling(5).mean() < df.close.rolling(50).mean())
        &
        (df.bar_type.shift(1) == 'down')
        &
        (df.bar_type == 'up')
        &
        (df.high < df.high.shift(1))
        &
        (df.low.shift(-1) < df.low)
    ]

    if not buys.empty:
        df.loc[buys.index, 'pattern'] = '3_bar'  
        df.loc[buys.index, 'buy_entry'] = df.loc[buys.index, 'high']  
        df.loc[buys.index, 'sl'] = df.loc[buys.index, 'low']  
        df.loc[buys.index, 'tp'] = df.loc[buys.index, 'buy_entry'] + rrr * (df.loc[buys.index, 'buy_entry'] - df.loc[buys.index, 'sl'])  
    if not sells.empty:
        df.loc[sells.index, 'pattern'] = '3_bar'  
        df.loc[sells.index, 'sell_entry'] = df.loc[sells.index, 'low']
        df.loc[sells.index, 'sl'] = df.loc[sells.index, 'high']  
        df.loc[sells.index, 'tp'] = df.loc[sells.index, 'sell_entry'] - rrr * (df.loc[sells.index, 'sl'] - df.loc[sells.index, 'sell_entry']) 

def in_htf_zone(df, rrr=3):
    ''' this "entry signal" is just to create a column
    in the df which will track when price is in a htf zone'''

    buys = df[df.bar_type == 'up']
    sells = df[df.bar_type == 'down']

    if not buys.empty:
        df.loc[buys.index, 'pattern'] = 'buy_zone'  
        df.loc[buys.index, 'buy_entry'] = df.loc[buys.index, 'high']  
        df.loc[buys.index, 'sl'] = df.loc[buys.index, 'low']  
        df.loc[buys.index, 'tp'] = df.loc[buys.index, 'buy_entry'] + rrr * (df.loc[buys.index, 'buy_entry'] - df.loc[buys.index, 'sl'])  
    
    if not sells.empty:
        df.loc[sells.index, 'pattern'] = 'sell_zone'  
        df.loc[sells.index, 'sell_entry'] = df.loc[sells.index, 'low']
        df.loc[sells.index, 'sl'] = df.loc[sells.index, 'high']  
        df.loc[sells.index, 'tp'] = df.loc[sells.index, 'sell_entry'] - rrr * (df.loc[sells.index, 'sl'] - df.loc[sells.index, 'sell_entry']) 


'''{'USD': 
        {'M5': 'buy'/'sell'}
    ...
    }'''

def trade_scanner() -> dict:
    ''' Will only return if new trades found '''
    
    timeframes = timeframes_to_request()
    if not timeframes:
        return

    # I need the list of currencies and their directions
    trade_directions = pd.read_csv(Path(p, 'trade_directions.csv'))
    trade_directions.end = pd.to_datetime(trade_directions.end)
    current = trade_directions[datetime.now() < trade_directions.end]
    
    # There aren't supposed to be duplicates but just in case...
    current = current[['currency', 'direction']].drop_duplicates()
    symbols = current.currency.tolist()
    directions = current.direction.tolist()

    ######### TEST SECTION ##########
    # scan on pairs instead of indexes
    # so need to build list of pairs

    opposite = {
        'buy': 'sell',
        'sell': 'buy'
    }

    pairs = []
    p_directions = []
    for ccy, direction in zip(symbols, directions):
        for pair in mt5_symbols['majors']:
            if ccy in pair:
                if ccy == pair[:3]: # base
                    pairs.append(pair)
                    p_directions.append(direction)
                else:  # counter
                    pairs.append(pair)
                    p_directions.append(opposite[direction])

    #####################################

    sr_levels = init_sr_levels(symbols=pairs, timeframes=timeframes)

    entry_signals = {}
    for timeframe in timeframes:
        for symbol, direction in zip(pairs, p_directions):
            df = mt5_ohlc_request(symbol, timeframe, num_candles=250)
            # df = pd.read_parquet(Path(INDEXES_PATH, f'temp_{symbol}.parquet'))
            # df = _resample(df, timeframe).dropna()
            if len(df) < 10:
                continue
            format(df)
            sr_levels = save_levels_data(sr_levels, df, symbol, timeframe)
            flying_buddha(df)
            trade_simple_volume(df)
            trade_reversal(df)      ###
            trade_spring_new(df)    ###
            trade_3_bar(df)         ###
            mike_trade_basic_engulf(df)
            mike_trade_reversal(df)
            filter_level_proximity(sr_levels, df, symbol, timeframe)

            trade = df['pattern'][-2]
            trade_type = 'buy' if not pd.isnull(df['buy_entry'][-2]) else 'sell'
            if trade_type == direction: # manually set direction
                if not pd.isnull(trade):
                    print('\n')
                    print('Trade Scanner:', datetime.now(), symbol, timeframe, trade, direction)
                    bot.send_message(chat_id=446051969, text=f'Trade Scanner: {symbol} {timeframe} {trade} {direction}')

                    ####  THIS IS ALSO PART OF TEST ####
                    # decide whether to send base or counter. kinda.
                    base = symbol[:3]
                    counter =symbol[3:]
                    x = base if base in symbols else counter
                    if symbol not in entry_signals:
                        entry_signals[x] = {}
                        # entry_signals[symbol] = {}
                    entry_signals[x][timeframe] = direction
                    # entry_signals[symbol][timeframe] = direction
    
    if entry_signals:
        
        return entry_signals

if __name__ == '__main__':
        pass
        # bot.send_photo(chat_id=-508710695, photo=open(pathlib.Path(r'C:\Users\ru'+ f'\{symbol}_{timeframe}_{trade}.png'), 'rb'))

        ### CHART PLOTTING SECTION ###

        # df = _add_ending_nans_to_df(df, 3)
        # trade = 'spring'
        # Generate image to send to telegram
        # plots = []
        # if df.sell_entry.any():
        #     plots.append(mpf.make_addplot(df.sell_entry,type='scatter',color='r'))
        # if df.buy_entry.any():
        #     plots.append(mpf.make_addplot(df.buy_entry,type='scatter',color='#00ff00'))
        # if df.buy_engulf.any():
        #     plots.append(mpf.make_addplot(df.buy_engulf,type='scatter',markersize=100,marker='^',color='#ff9900'))
        # if df.sell_engulf.any():
        #     plots.append(mpf.make_addplot(df.sell_engulf,type='scatter',markersize=100,marker='v',color='#ff9900'))
        # if plots:
            # plots.append(mpf.make_addplot(df.sl,type='scatter',color='#ff9900')) 
            # plots.append(mpf.make_addplot(df.tp,type='scatter',color='#0000ff'))
            # plots.append(mpf.make_addplot(df.smma_50,type='line',color='#0066ff',))     
            # mpf.plot(df, type='candle', tight_layout=True, 
            # show_nontrading=False, volume=False,
            # addplot=plots,
            # title=symbol +' '+ timeframe + ' ' + trade,
            # style='classic',   ## black n white
            # savefig=f'{symbol}_{timeframe}_{trade}.png'
            # )
            # time.sleep(1)
            
