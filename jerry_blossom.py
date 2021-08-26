import pandas as pd
import numpy as np
import telegram
import mplfinance as mpf
from datetime import datetime
from symbols_lists import mt5_symbols, indexes, candles_per_day, etf_to_htf, etf_to_int
from ohlc_request import mt5_ohlc_request
import pathlib
import time

INDEXES_PATH = r'C:\Users\ru\forex\db\indexes'
bot = telegram.Bot(token='1777249819:AAGiRaYqHVTCwYZMGkjEv6guQ1g3NN9LOGo')


df = pd.DataFrame()
timeframe = None
symbol = None

##### Format functions #####

def _resample(ohlc_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    ''' Just a simple resampling into higher timeframes for whatever
    OHLCV data set gets passed in. '''

    # Reorder if needed. Number must come first
    if timeframe[0].upper() == 'D' or timeframe[0].upper() == 'H' or timeframe[0].upper() == 'M' or timeframe[0].upper() == 'W':
        if len(timeframe) == 2: 
            tf = timeframe[::-1]
        if len(timeframe) == 3: 
            tf = timeframe[1] + timeframe[2] + timeframe[0]
    else:
        tf = timeframe

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
    
def hlc3(df):
    df['hlc3'] = (df.high + df.low + df.close) / 3

def pinbar(df):
    
    df['pinbar'] = np.nan

    ups_idx = df[
        (df.bar_type == 'up')
        &
        (df.open - df.low > (df.close - df.open) * 2)
        ].index
    df.loc[ups_idx, 'pinbar'] = 'up'

    downs_idx = df[
        (df.bar_type == 'down')
        &
        (df.high - df.open > (df.open - df.close) * 2)
        ].index
    df.loc[downs_idx, 'pinbar'] = 'down'
      
def find_peaks(df, rows):
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

    # Since I'm not using an integer index I gotta get a datetime diff value
    # and apply that to whatever win value I'm looking for

    win = 4  
    for row in temp.itertuples(name=None, index=True):
        i = row[0]
        # Price move from the peak (forwards then backwards)
        if df.loc[i:i+win * rows, 'high'].max() - df.loc[i, 'low'] > fade * df.loc[i, 'atr']:
            if df.loc[i-win * rows:i, 'high'].max() - df.loc[i, 'low'] > fade * df.loc[i, 'atr']:
                df.loc[i, 'peak_lo'] = 'fade'
                continue

        # If not found look for the spring sized trades
        if (df.loc[i:i+win * rows, 'high'].max() - df.loc[i, 'low']) > spring * df.loc[i, 'atr']:
            if (df.loc[i-win * rows:i, 'high'].max() - df.loc[i, 'low']) > spring * df.loc[i, 'atr']:
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
        if (df.loc[i, 'high'] - df.loc[i:i+win * rows, 'low'].min()) > fade * df.loc[i, 'atr']:
            if (df.loc[i, 'high'] - df.loc[i-win * rows:i, 'low'].min()) > fade * df.loc[i, 'atr']:
                df.loc[i, 'peak_hi'] = 'fade'
                continue
            
        # If not found look for the swing sized swings
        if (df.loc[i, 'high'] - df.loc[i:i+win * rows, 'low'].min()) > spring * df.loc[i, 'atr']:
            if (df.loc[i, 'high'] - df.loc[i-win * rows:i, 'low'].min()) > spring * df.loc[i, 'atr']:
                df.loc[i, 'peak_hi'] = 'spring'
    # print(len(df[df.peak.notna()]), 'atr filtered peaks')

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

def set_ema(df):
    df['ema_7'] = df.hlc3.ewm(span=7,adjust=False).mean()
    df['ema_9'] = df.hlc3.ewm(span=9,adjust=False).mean()
    df['ema_20'] = df.hlc3.ewm(span=20,adjust=False).mean()
    df['ema_50'] = df.hlc3.ewm(span=50,adjust=False).mean()
    df['ema_100'] = df.hlc3.ewm(span=100,adjust=False).mean()

def set_atr(df, n=14):
    data = pd.DataFrame()
    data['tr0'] = abs(df.high - df.low)
    data['tr1'] = abs(df.high - df.close.shift())
    data['tr2'] = abs(df.low - df.close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean() 
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
            (df.volume.shift(1) > df.volume)) 
            | 
            ((df.hlc3.shift(2) > df.hlc3.shift(1)) & # down trend
            (df.low == df.low.rolling(10).min()) & 
            (df.volume.shift(1) > df.volume))
            ]
    df.loc[temp.index, 'auction_vol'] = 1 # one bar with less volume

    # twos
    temp = df[((df.hlc3.shift(2) < df.hlc3.shift(1)) &
            (df.high == df.high.rolling(10).max()) & 
            (df.volume.shift(2) > df.volume.shift(1)) & # extra volume bar required
            (df.volume.shift(1) > df.volume)) 
            | 
            ((df.hlc3.shift(2) > df.hlc3.shift(1)) &
            (df.low == df.low.rolling(10).min()) & 
            (df.volume.shift(2) > df.volume.shift(1)) &
            (df.volume.shift(1) > df.volume))
            ]
    df.loc[temp.index, 'auction_vol'] = 2 # two bars with less volume

def find_thrusts(df: pd.DataFrame, roll_window:int = 5) -> None:
    ''' Identify periods where there has been an impulsive price move,
    either up or down, and locate both the start and end of that move.'''

    # I need separate thrust cols so that a single peak can be both a 
    # "end up" and a "start down"
    df['thrust_up'] = np.nan
    df['thrust_down'] = np.nan
    
    # First I need to find where a rolling diff is above/below some threshold
    up_thresh = df.close.diff().rolling(roll_window).mean().describe([.9])['90%']
    down_thresh = df.close.diff().rolling(roll_window).mean().describe([.1])['10%'] 
    up_thrusts = df[df.close.diff().rolling(roll_window).mean() >= up_thresh].index
    down_thrusts = df[df.close.diff().rolling(roll_window).mean() <= down_thresh].index
    df.loc[up_thrusts, 'thrust_up'] = 'up'       
    df.loc[down_thrusts, 'thrust_down'] = 'down'    

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
    df['level_price'] = np.nan
    df['level_thresh'] = np.nan


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
                # df.loc[end_idx, 'brl_idx_loc'] = brl_idx  # I may not end up using this if i start saving: start, end, price and thresh
                df.loc[end_idx, 'level_type'] = 'brl'
                df.loc[end_idx, 'level_direction'] = 'buy'
                df.loc[end_idx, 'level_start'] = end_idx  # trying this out, might make queries later on more clear
                df.loc[end_idx, 'level_price'] = df.loc[brl_idx, 'high']
                df.loc[end_idx, 'level_thresh'] = df.loc[brl_idx, 'high'] + df.atr.mean()

    #   -- SET TERMINATION POINTS  -- 

    # The last thing to do is to find the termination points of a BRL. Once price
    # has pierced a level by some amount I will want that level to become inactive.
    # Price needs to not only pierce by some distance but also for multiple candles
    for i in df[(df.level_type == 'brl') & (df.level_direction == 'buy')].index:
        index_loc = df[
                    (df.close < (df.loc[i, 'level_price'] - df.loc[i, 'atr'] * 1))
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
                df.loc[end_idx, 'level_price'] = df.loc[brl_idx, 'low']
                df.loc[end_idx, 'level_thresh'] = df.loc[brl_idx, 'low'] - df.atr.mean()
    
    #   -- SET TERMINATION POINTS  -- 

    for i in df[(df.level_type == 'brl') & (df.level_direction == 'sell')].index:
        index_loc = df[
                    (df.close > (df.loc[i, 'level_price'] + df.loc[i, 'atr'] * 1))
                    &
                    (df.index > i)
                    ].index
        df.loc[i, 'level_end'] = index_loc[2] if len(index_loc) > 3 else i + (df.index[1] - df.index[0]) * extension
    
def save_trade_info(df, i):
    '''
    Save the trade info to the df
    '''
    cur_atr = df.loc[i, 'atr']
    sl_r = 2.5
    tp1_r = 2.5
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
        df.loc[i, 'buy_entry'] = entry
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
        df.loc[i, 'sell_entry'] = entry
        df.loc[i, 'sl'] = sl
        df.loc[i, 'tp1'] = tp1
        df.loc[i, 'tp2'] = tp2

def find_sd_zones(df):
    ''' 
    supply example 
    (df.peak_high.notna())
    (df.high == df.high.rolling(12).max())
    (df.peak_high.shift(1).rolling(12).notna().any())
    (df.close.shift(-2).diff().rolling(2).mean() > df.close.diff().rolling(20).mean() * 2)
    '''
    pass

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
            new_levels['price'] = df_levels.level_price
            new_levels['thresh'] = df_levels.level_thresh
            new_levels['symbol'] = symbol
            new_levels['timeframe'] = timeframe
            new_levels['direction'] = level_direction
            new_levels['type'] = level_type

            # Append the levels and clear then erase them
            htf_levels = htf_levels.append(new_levels)
            new_levels = new_levels[0:0]

    # Ensure this is sent out with unique index values
    return htf_levels.reset_index(drop=True)


#### TRADE SETUPS ######

def random_test_b(df, vol_roll = 3):
    ''' thrust w vol change'''

    roll_win = 20
    half_win = 10
    down_thresh = df.close.diff().rolling(roll_win).mean().describe([.2])['20%'] 
    #up_thresh = df.close.diff().rolling(roll_win).mean().describe([.6])['60%']
    avg_volume = ((df.high - df.low) / df.volume).rolling(30).mean()
    # These thresh values are meant to target the recent peak
    max_price_thresh = df.low.shift(half_win).rolling(roll_win).min() + (df.atr.rolling(roll_win).mean() * 1)
    min_price_thresh = df.low.shift(half_win).rolling(roll_win).min() - (df.atr.rolling(roll_win).mean() * 2.5)
    # This one is a safety measure in case that peak was breached alot, I dont wanna be too far away from the lowest low
    max_dist_from_lows = df.low.shift(1).rolling(half_win).min() + (df.atr.rolling(roll_win).mean() * 1)
    # I need a lowest to highest peak size filter. to ensure theres been  a decent bounce.
    # Or just like a diff filter based on atr
    
    # Buys
    buy_setups = df[

        # Buyers have already appeared making an up-thrust (this one Im not sure about)
        (df.thrust_up.notna().rolling(10).sum() > 0)   
        &
        # Impulse into the HTF BRL 
        (df.close.diff().rolling(3).mean() < df.atr.rolling(roll_win).mean() * -1.5)    
        &
        # At least one previous peak already exists nearby 
        # (need a shift value to not cheat)
        # should prob increase peak size and increase this window 
        (df.low_peak.shift(1).notna().rolling(roll_win).sum() > 0)
        &
        # This high peak will only exist if enough of a bounce occured off the low peak
        (df.high_peak.shift(1).notna().rolling(roll_win).sum() > 0)
        &
        # Price is low, but not too low
        (df.close < max_price_thresh)
        &
        (df.close > min_price_thresh)
        &
        (df.close < max_dist_from_lows)
        &
        # Some kinda random volume abnormality
        (
            ((df.high - df.low) / df.volume > avg_volume * 1.4)
            |
            ((df.high - df.low) / df.volume < avg_volume * 0.6)
        )
    ].index

    df.loc[buy_setups, 'pattern'] = 'random_b'
    for i in buy_setups:
        save_trade_info(df, i)

def random_test_s(df, vol_roll = 3):
    ''' thrust w vol change'''

    roll_win = 20
    half_win = 10
    up_thresh = df.close.diff().rolling(roll_win).mean().describe([.8])['80%']
    avg_volume = ((df.high - df.low) / df.volume).rolling(30).mean()
    max_price_thresh = df.high.shift(half_win).rolling(roll_win).max() + (df.atr.rolling(roll_win).mean() * 2.5)
    min_price_thresh = df.high.shift(half_win).rolling(roll_win).max() - (df.atr.rolling(roll_win).mean() * 1)
    max_dist_from_highs = df.high.shift(1).rolling(half_win).max() - (df.atr.rolling(roll_win).mean() * 1)

    # Sells
    sell_setups = df[
        (df.thrust_down.notna().rolling(10).sum() > 0)   
        &
        (df.close.diff().rolling(3).mean() > df.atr.rolling(roll_win).mean() * 1.5)    
        &
        (df.high_peak.shift(1).notna().rolling(roll_win).sum() > 0)
        &
        (df.low_peak.shift(1).notna().rolling(roll_win).sum() > 0)
        &
        (df.close > max_dist_from_highs)
        &
        (df.close < max_price_thresh)
        &
        (df.close > min_price_thresh)
        &
        (
            ((df.high - df.low) / df.volume > avg_volume * 1.4)
            |
            ((df.high - df.low) / df.volume < avg_volume * 0.6)
        )
    ].index

    df.loc[sell_setups, 'pattern'] = 'random_s'
    for i in sell_setups:
        save_trade_info(df, i)


#### TRADE FILTERS #####
   
def brl_proximity(df: pd.DataFrame, trade_direction:str) -> bool:
    ''' Check if Close is currently near a BRL level'''
    
    if trade_direction == 'buy':

        # Filter for BRLs which have an 'end' beyond the last row of df
        # and whose price level is within some distance of current price
        buys = df[df.brl_buy_idx_loc.notna()]
        active_buys = buys[
            (buys.brl_buy_idx_end >= df.index[-1])
            &
            (buys.high + df.atr[-1] > df.close[-1])  # Yhe BRL price + 1 atr is > than the current close
            ]

        if len(active_buys) > 0:
            return True

    # Sells
    if trade_direction == 'sell':
        sells = df[df.brl_sell_idx_loc.notna()]
        active_sells = sells[
            (sells.brl_sell_idx_end >= df.index[-1])
            &
            (sells.low - df.atr[-1] < df.close[-1])
            ]

        if len(active_sells) > 0:
            return True

def trade_filter_level_proximity(htf_levels, df, symbol, timeframe):
    '''this func identfies rows that are within a htf levels
    proximity. any trade that is not near a htf level gets deleted
    t. a new column is created in the df to track it because
    another filter function refermces the proximity data '''
    trade_directions = {
        'buy': 'sell',
        'sell': 'buy'
    }

    htf_1 = etf_to_htf[timeframe][0]
    htf_2 = etf_to_htf[timeframe][1]

    df['level_proximity'] = np.nan

    # i will need to filter the htf_levels based on direvtion
    # and timeframe and then iter thru each one to grab rach rows data
    for direction, opposite_direction in trade_directions.items():
        levels = htf_levels[
                (
                    (htf_levels.timeframe == htf_1)
                    |
                    (htf_levels.timeframe == htf_2)
                )
                &
                (
                    (  # Exact symbol or base ccy index
                        ((htf_levels.symbol == symbol)
                        |
                        (htf_levels.symbol == symbol[:3]))
                        &
                        (htf_levels.direction == direction)
                    )
                    |
                    (   # Counter ccy index with flipped direction
                        (htf_levels.symbol == symbol[3:])
                        &   
                        (htf_levels.direction == opposite_direction)
                    )
                )
                ]

        # Erase the trade entries if nothing is found
        if levels.empty:
            if direction == 'buy':
                df.buy_entry = np.nan
            if direction == 'sell':
                df.sell_entry = np.nan
            continue 

        # See if any trades in the df match these params
        for row in levels.itertuples(name=None, index=True):
            start = row[1]
            end = row[2]
            threshold = row[4]

            if direction == 'buy':
                buy_trades = df[
                                (df.index > start)
                                &
                                (df.index < end)
                                &
                                (df.close < threshold)
                            ].index
                df.loc[buy_trades, 'level_proximity'] = 'buy'

            if direction == 'sell':
                sell_trades = df[
                                (df.index > start)
                                &
                                (df.index < end)
                                &
                                (df.close > threshold)
                            ].index
                df.loc[sell_trades, 'level_proximity'] = 'sell'

    # Delete whatever didnt get confirmed
    df.loc[
        (df[
            (df.level_proximity != 'buy') 
            & 
            (df.buy_entry.notna())
            ].index
        ), 'buy_entry'] = np.nan
        
    df.loc[
        (df[
            (df.level_proximity != 'sell') 
            & 
            (df.sell_entry.notna())
            ].index
        ), 'sell_entry'] = np.nan

def trade_filter_time_of_day(df,  start_trading:int=4, stop_trading:int=6):
    ''' linit trqde signals to fire only between certain hours'''

    no_trading = df.index[
        (df.index.hour < start_trading)
        |
        (df.index.hour > stop_trading)
    ]

    df.loc[no_trading, 'buy_entry'] = np.nan
    df.loc[no_trading, 'sell_entry'] = np.nan

def trade_filter_daily_range(df, symbol, timeframe):
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

def trade_filter_flip_direction(df):

    df['buy_to_sell_entry'] = np.nan
    buy = df[(df.level_proximity == 'buy')]
    buy_to_sell = buy[
        # this should be a new column so i can compare + moves and - moves
       # (buy.diff().diff().rolling(3) > buy.atr * 2) # limited number of up thrusts

        # if there has already been 3 bounces on the level
        # (buy.low_peak.notna().rolling(50).sum()> 2)
        # |
        # # if there has already been 1 bounce but now there is sideways PA
        # (
        #     (buy.low_peak.notna().rolling(50).sum()> 0)
        #     &
        #     (abs(buy.high.diff()).rolling(5).mean() < abs(buy.high.diff()).rolling(20).mean())
        #     &
        #     (abs(buy.low.diff()).rolling(5).mean() < abs(buy.low.diff()).rolling(20).mean())
        # )
        # |
        # # if there is a general penant pattern with lower highs but semi equal lows.
        # # the problem with this approach though is i have to get the roll window right
        # (
        #     # can divide by atr to get real values 
        #     (buy.low.diff().rolling(15).mean() > buy.high.diff().rolling(15).mean())
        #     &
        #     (buy.low.diff().rolling(15).mean() / buy.atr.rolling(1).mean() > buy.atr.rolling(1).mean() * -0.3) # hm
        # )
        # |
        # if there are a certain number of bars within range of the zone
        (
            (buy.low_peak.notna().rolling(50).sum() > 4)
            |
            (
                (buy.low_peak.notna().rolling(50).sum() > 3)
                &
                (buy.low.notna().rolling(60).sum() > 45)
            )
        )
        &
        (buy.buy_entry.notna())

    ].index

    # erase those buys and create a new col (just for plottong)
    df.loc[buy_to_sell, 'buy_entry'] = np.nan
    df.loc[buy_to_sell, 'buy_to_sell_entry'] = df.loc[buy_to_sell, 'close']

def trade_filter_excessive_entries(df):
    '''maybe allow 2 entries in a row but then require
    either a certain number of bars for downtime, or a 
    minimum price range'''

    # um….
    # what if i flipped the third or fourth entry if the 
    # price has moved a certain distance (ie big breakout bars)
    # … iono if im really ever gonna find a good reversal filter 
    three_ina_row = df[
        (df.buy_entry.notna().rolling(3).sum() == 3)
    ]

def timeframes_to_request():

    minute = datetime.now().minute
    if minute == 0:
        return etf_to_htf
    
    timeframes = []
    for tf in etf_to_htf:
        if minute % etf_to_int[tf] == 0:
            timeframes.append(tf)
    return timeframes


if __name__ == '__main__':


    while True:
        #if 1:
        if datetime.now().second == 0:
            htf_levels = pd.DataFrame()

            total =0
                # First loop thru the indexes, resampling into HTFs and saving BRL levels
                # ''' I need some other way to track SR proximity since I can't do it with price 
                # like a normal pair'''
            # for index in indexes:
            #     #     # print('checking for BRLs on', symbol)
            #     df = pd.read_parquet(pathlib.Path(INDEXES_PATH + f'\{index}_5_min.parquet'))

            #     #     # Resample into the different timeframes
            #     for timeframe in etf_to_htf:
            #         df = _resample(df, timeframe) if timeframe != 'M5' else df
            #         df = df.dropna()
            #         set_atr(df)
            #         find_simple_peaks(df)
            #         find_thrusts(df)
            #         find_brl_levels(df, 30)
            #         save_levels_data(htf_levels, df, index, timeframe)
            
            for symbol in mt5_symbols['majors']:
            # for symbol in ['DE30', 'NZDCHF','EURUSD']:
                for timeframe in timeframes_to_request():
                #for timeframe in etf_to_htf:

                    df = mt5_ohlc_request(symbol, timeframe, num_candles=450)
                    df['buy_entry'] = np.nan
                    df['sell_entry'] = np.nan    
                    set_atr(df)
                    find_simple_peaks(df)
                    find_thrusts(df)
                    find_brl_levels(df, 30)
                    set_bar_type(df)
                    pinbar(df)

                    htf_levels = save_levels_data(htf_levels, df, symbol, timeframe)

                    if len(htf_levels) == 0:
                        continue    # else error
                    
                    # scan for trades using larger peak size
                    #df.loc[df.index[df.high_peak == 'small'], 'high_peak'] = np.nan
                    #df.loc[df.ind                                                                                                                                              ex[df.low_peak == 'small'], 'low_peak'] = np.nan

                    random_test_b(df)
                    random_test_s(df)

                    # Various filters to remove trade entries
                    # print(symbol, timeframe)
                    #print('trades before filtering: ', df.buy_entry.notna().sum() + df.sell_entry.notna().sum())
                    # trade_filter_level_proximity(htf_levels, df, symbol, timeframe)
                    #print('after BRL filtering: ', df.buy_entry.notna().sum() + df.sell_entry.notna().sum())
                    # trade_filter_time_of_day(df)
                    #print('after TOD filtering: ', df.buy_entry.notna().sum() + df.sell_entry.notna().sum())
                    trade_filter_daily_range(df, symbol, timeframe)
                    #print('after ADR filtering: ', df.buy_entry.notna().sum() + df.sell_entry.notna().sum())
                    # trade_filter_flip_direction(df)
                    
                    # If there is a fresh signal on a new candle...
                    pattern = df.loc[df.index.max(), 'pattern']
                    if not pd.isnull(pattern):
                        #print(df.pattern.tail())
                        print(datetime.now(), symbol, timeframe, pattern)
                        bot.send_message(chat_id=446051969, text=f' {symbol} {timeframe} {pattern}')
                    continue

                    # Create plot objects for entries
                    plots = []
                    if df.sell_entry.any():
                        plots.append(mpf.make_addplot(df.sell_entry,type='scatter',color='r'))
                    if df.buy_entry.any():
                        plots.append(mpf.make_addplot(df.buy_entry,type='scatter',color='#00ff00'))

                    if plots:
                        mpf.plot(df, type='candle', tight_layout=True, 
                        show_nontrading=False, volume=False,
                        addplot=plots,
                        title=symbol +' '+ timeframe,
                        style='classic',   ## black n white
                        savefig=f'{symbol}_{timeframe}.png'
                        )
                    
                    # Count how many trades have occured
                    # df = df[df.index > df.index.max() - pd.Timedelta('2 day')]
                    # if timeframe == 'M3' or timeframe == 'M5':
                    #    total += df.buy_entry.notna().sum() + df.sell_entry.notna().sum()
                    #    print(total)
                    # continue
        # quit()