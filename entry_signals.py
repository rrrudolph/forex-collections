import MetaTrader5 as mt5
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import time
from create_db import econ_db
from symbols_lists import mt5_symbols, mt5_timeframes
from ohlc_request import mt5_ohlc_request
from tokens import bot

econ_con = sqlite3.connect(econ_db)
df = pd.DataFrame()
timeframe = None
symbol = None

def send_trade_alert(symbol, timeframe, pattern):
    bot.send_message(chat_id=446051969, text=f'{symbol} {timeframe} {pattern}')

def _send_telegram_forecast(whatever):
    bot.send_message(chat_id=446051969, text=whatever)

def _lot_size(risk, df, i, symbol):
    r''' Automatically set lot size based on desired risk %.
    Set the risk % as a decimal for the first arg. '''

    symb_info = mt5.symbol_info(symbol)
    acc_info = mt5.account_info()

    # get the pip value (of 0.01 lot)
    pip_val = symb_info.trade_tick_value
    
    # get the distance from entry to sl in pips
    distance = abs(df.loc[i, 'entry'] - df.loc[i, 'sl'])
    distance /= symb_info.point * 10
    
    # min loss
    loss_with_min_lot = distance * pip_val

    # Divide risk per trade by loss_with_min_lot
    risk_per_trade = risk * acc_info.equity
    lot_size = risk_per_trade // loss_with_min_lot
    return lot_size

def _expiration(timeframe, num_candles=4):
    ''' Set the order expiration at n candles, so it will depend on the timeframe. '''

    if timeframe == mt5.TIMEFRAME_D1: # daily
        t_delta = pd.Timedelta('1 day')
    else:
        t_delta = pd.Timedelta(f'{timeframe} min')
    return datetime.now() + num_candles * t_delta

def _increase_volume_if_forecast(df, i, symbol):
    ''' If there are forecasts that contribute to the trade's liklihood 
    of working out, increase the lot size.  There's definitely more to 
    be coded here about how to interpret forecast data. '''


    forecast_df = pd.read_sql('SELECT * FROM outlook', econ_con)

    base_ccy = symbol[:3]
    counter_ccy = symbol[-3:]

    # Check the forecasts
    base_sum = round(sum(forecast_df.forecast[forecast_df.ccy == base_ccy]), 2)
    counter_sum = round(sum(forecast_df.forecast[forecast_df.ccy == counter_ccy]), 2)

    if base_sum or counter_sum:
        # Send the forecast data to telegram
        _send_telegram_forecast(f'{base_ccy}: {base_sum}  {counter_ccy}: {counter_sum}')

    # Check if trade is long or short
    if df.loc[i, 'pattern'][-2:] == '_b':
        trade_type = mt5.ORDER_TYPE_BUY_STOP

    if df.loc[i, 'pattern'][-2:] == '_s':
        trade_type = mt5.ORDER_TYPE_SELL_STOP

    # risk multiplier starts at 1%
    x = 0.01
    if trade_type == mt5.ORDER_TYPE_BUY_STOP:
        if base_sum > 0 and counter_sum < 0:
            x *= 2
        elif base_sum > 0 or counter_sum < 0:
            x *= 1.5
        
        # if totally opposite
        elif base_sum < 0 and counter_sum > 0:
            x *= 0.1
    
    # If it's a short trade reverse the forecast numbers
    if trade_type == mt5.ORDER_TYPE_SELL_STOP:
        if base_sum < 0 and counter_sum > 0:
            x *= 2
        elif base_sum < 0 or counter_sum > 0:
            x *= 1.5
        
        # if totally opposite 
        elif base_sum > 0 and counter_sum < 0:
            x *= 0.1
    
    return x, trade_type

def enter_trade(df, i, symbol, timeframe):
    ''' Read from the df of current setups and set lot size based on 
    the forecast rating for the currencies in the symbol. '''

    risk, trade_type = _increase_volume_if_forecast(df, i, symbol)

    lot_size = _lot_size(risk, df, i, symbol)
    lot_size *= 0.1 ############ too big
    lot_size = round(lot_size, 2)

    expiration = _expiration(timeframe)

    request_1 = {
    "action": mt5.TRADE_ACTION_PENDING,
    "symbol": symbol,
    "volume": lot_size,
    "type": trade_type,
    "price": df.loc[i, 'entry'],
    "sl": df.loc[i, 'sl'],
    "tp": df.loc[i, 'tp1'],  ### TP 1
    "deviation": 20,
    "magic": 234000,
    "comment": f"{df.loc[i, 'pattern']} {timeframe}",
    "type_time": mt5.ORDER_TIME_DAY,
    "type_filling": mt5.ORDER_FILLING_FOK,
    }
                                                                                                                       
    request_2 = {
    "action": mt5.TRADE_ACTION_PENDING,
    "symbol": symbol,
    "volume": lot_size,
    "type": trade_type,
    "price": df.loc[i, 'entry'],
    "sl": df.loc[i, 'sl'],
    "tp": df.loc[i, 'tp2'],  ### TP 2
    "deviation": 20,
    "magic": 234000,
    "comment": f"{df.loc[i, 'pattern']} {timeframe}",
    "type_time": mt5.ORDER_TIME_DAY,
    "type_filling": mt5.ORDER_FILLING_FOK,
    }


    # send a trading request
    trade = mt5.order_send(request_1)
    trade = mt5.order_send(request_2)

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

    # Since I'm not using an integer index I gotta get a datetime diff value
    # and apply that to whatever win value I'm looking for
    row_diff = df.index[1] - df.index[0]

    win = 4  
    for row in temp.itertuples(name=None, index=True):
        i = row[0]
        # Price move from the peak (forwards then backwards)
        if df.loc[i:i+win * row_diff, 'high'].max() - df.loc[i, 'low'] > fade * df.loc[i, 'atr']:
            if df.loc[i-win * row_diff:i, 'high'].max() - df.loc[i, 'low'] > fade * df.loc[i, 'atr']:
                df.loc[i, 'peak_lo'] = 'fade'
                continue

        # If not found look for the spring sized trades
        if (df.loc[i:i+win * row_diff, 'high'].max() - df.loc[i, 'low']) > spring * df.loc[i, 'atr']:
            if (df.loc[i-win * row_diff:i, 'high'].max() - df.loc[i, 'low']) > spring * df.loc[i, 'atr']:
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
        if (df.loc[i, 'high'] - df.loc[i:i+win * row_diff, 'low'].min()) > fade * df.loc[i, 'atr']:
            if (df.loc[i, 'high'] - df.loc[i-win * row_diff:i, 'low'].min()) > fade * df.loc[i, 'atr']:
                df.loc[i, 'peak_hi'] = 'fade'
                continue
            
        # If not found look for the swing sized swings
        if (df.loc[i, 'high'] - df.loc[i:i+win * row_diff, 'low'].min()) > spring * df.loc[i, 'atr']:
            if (df.loc[i, 'high'] - df.loc[i-win * row_diff:i, 'low'].min()) > spring * df.loc[i, 'atr']:
                df.loc[i, 'peak_hi'] = 'spring'
    # print(len(df[df.peak.notna()]), 'atr filtered peaks')

def find_simple_peaks(df):
    ''' This version will not use ATR filtering and is intended to be
    used specifically for thrust start and end locations as well
    as BRLs. The BRLs and start peak size will be smaller than the end peaks. '''

    # Specificy different columns for the down swings and up swings,
    # otherwise they will overwrite each other. An alternative is to not
    # use different peak sizes for start and ends, in that case both
    # swing types could share.

    df['up_peak'] = np.nan
    df['down_peak'] = np.nan

    start_downs = df[
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
            # &
            # (df.high > df.high.shift(5)) 
            # &
            # (df.high >= df.high.shift(-5))
            ].index 
    df.loc[start_downs, 'down_peak']= 'start_down'
    
    start_ups = df[
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
            # &
            # (df.low < df.low.shift(5)) 
            # &
            # (df.low <= df.low.shift(-5))
            ].index
    df.loc[start_ups, 'up_peak'] = 'start_up'

    # Now add some shift values to find bigger swings for the thrust end peaks
    end_ups = df[
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
            # &
            # (df.high > df.high.shift(7)) 
            # &
            # (df.high >= df.high.shift(-8)) 
            ].index
    df.loc[end_ups, 'up_peak'] = 'end_up'
    
    end_downs = df[
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
            # &
            # (df.low < df.low.shift(8)) 
            # &
            # (df.low <= df.low.shift(-8)) 
            ].index
    df.loc[end_downs, 'down_peak'] = 'end_down'


def set_ema(df):
    df['ema_7'] = df.hlc3.ewm(span=7,adjust=False).mean()
    df['ema_9'] = df.hlc3.ewm(span=9,adjust=False).mean()
    df['ema_20'] = df.hlc3.ewm(span=20,adjust=False).mean()
    df['ema_50'] = df.hlc3.ewm(span=50,adjust=False).mean()
    df['ema_100'] = df.hlc3.ewm(span=100,adjust=False).mean()

def atr(df, n=14):
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
    # the thrust period or afterwards. If no low peak exists I'll default to lowest low.

    up_starts = df.thrust_up[
            (df.up_peak == 'start_up')
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

    # Find the end will be essentially the same but I'll be looking for the opposite
    # type of peak, and in the opposite direction time-wise

    up_ends = df[
            (df.up_peak == 'end_up')
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
            (df.down_peak == 'start_down')
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
            (df.down_peak == 'end_down')
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

def find_brl_levels(df: pd.DataFrame, lookback) -> None:
    ''' This will locate SR levels where a fast and fairly large price
    movement has happened well beyond a previous peak. The row which 
    the BRL data will get saved to will be a thrust's "end", this will 
    make backtesting safe as the level will only appear after the 
    thrust which validated it.
    Note: atr() and find_thrusts() needs to be called first.''' 

    # I need to identify each unique thrust by getting start and end points,
    # then get the 50% level between those prices and use that to then find
    # peaks which have occured in the recent past, on a certain side of that 50%

    df['thrust_50'] = np.nan
    df['brl_buy1_idx_loc'] = np.nan
    df['brl_buy1_idx_end'] = np.nan
    df['brl_buy2_idx_loc'] = np.nan
    df['brl_buy2_idx_end'] = np.nan
    df['brl_sell1_idx_loc'] = np.nan
    df['brl_sell1_idx_end'] = np.nan
    df['brl_sell2_idx_loc'] = np.nan
    df['brl_sell2_idx_end'] = np.nan
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
        brl_index = df[
                    ((df.down_peak == 'start_down')               # BRL peak type
                    |              
                    (df.up_peak == 'end_up'))
                    &
                    (df.high < df.loc[end_idx, 'thrust_50'])    # price < than 50 price
                    &
                    (df.high > df.loc[start_idx, 'low'])        # price > than thrust start
                    &
                    (df.index < start_idx)                      # exists before thrust start
                    &
                    (df.index > start_idx - historical_limit)   # exists after historical limit      
                    ].index
                    
        #   -- SET BRLs --

        # Limit the amount found to the most recent 2. If more than 1 is found, 
        # very the older one sticks out abov the newer one
        df.loc[end_idx, 'brl_buy1_idx_loc'] = brl_index[-1] if len(brl_index) > 0 else np.nan
        if len(brl_index) > 1:
            # Compare highs
            if df.loc[brl_index[-2], 'high'] > df.loc[brl_index[-1], 'high']:
                df.loc[end_idx, 'brl_buy2_idx_loc'] = brl_index[-2]

    #   -- SET TERMINATION POINTS  -- 

    # The last thing to do is to find the termination points of a BRL. Once price
    # has pierced a level by some amount I will want that level to become inactive.
    # Price needs to not only pierce by some amount but also for multiple candles
    for i in df[df.brl_buy1_idx_loc.notna()].index:
        brl1_idx = df.loc[i, 'brl_buy1_idx_loc']  # This is the index of the actual brl
        index_loc = df[
                    (df.close < (df.loc[brl1_idx, 'high'] - df.loc[i, 'atr'] * 1.5))
                    &
                    (df.index > i)
                    ].index
        df.loc[i, 'brl_buy1_idx_end'] = index_loc[2] if len(index_loc) > 3 else df.index.max()
        # I set index_loc to 2 so that visual plots will be accurate with what happened live
    
    for i in df[df.brl_buy2_idx_loc.notna()].index:
        brl2_idx = df.loc[i, 'brl_buy2_idx_loc']  
        index_loc = df[
                    (df.close < (df.loc[brl2_idx, 'high'] - df.loc[i, 'atr'] * 1.5))
                    &
                    (df.index > i)
                    ].index
        df.loc[i, 'brl_buy2_idx_end'] = index_loc[2] if len(index_loc) > 3 else df.index.max()
    
    
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
        df.loc[end_idx, 'thrust_50'] = df.loc[start_idx, 'high'] - (df.loc[start_idx, 'high'] - df.loc[end_idx, 'low']) / 2

        #   -- FIND MAXIMIUM LOOK BACK --
        
        atr_high_to_50 = (df.loc[start_idx, 'high'] - df.loc[end_idx, 'thrust_50']) / df.loc[end_idx, 'atr']
        historical_limit = round(abs(atr_high_to_50) * lookback)
        historical_limit *=  df.index[1] - df.index[0]  # historical limit converted to num rows
        
        brl_index = df[
                ((df.down_peak == 'end_down')               # BRL peak type
                |              
                (df.up_peak == 'start_up'))
                &
                (df.low > df.loc[end_idx, 'thrust_50'])    # price > than 50 price
                &
                (df.low < df.loc[start_idx, 'high'])        # price < than thrust start
                &
                (df.index < start_idx)                      # exists before thrust start
                &
                (df.index > start_idx - historical_limit)   # exists after historical limit      
                ].index
                    
        #   -- SET BRLs --

        df.loc[end_idx, 'brl_sell1_idx_loc'] = brl_index[-1] if len(brl_index) > 0 else np.nan
        if len(brl_index) > 1:
            if df.loc[brl_index[-2], 'low'] < df.loc[brl_index[-1], 'low']:
                df.loc[end_idx, 'brl_sell2_idx_loc'] = brl_index[-2]
    
    #   -- SET TERMINATION POINTS  -- 

    for i in df[df.brl_sell1_idx_loc.notna()].index:
        brl1_idx = df.loc[i, 'brl_sell1_idx_loc']  # This is the index of the actual brl
        index_loc = df[
                    (df.close > (df.loc[brl1_idx, 'low'] + df.loc[i, 'atr'] * 1.5))
                    &
                    (df.index > i)
                    ].index
        df.loc[i, 'brl_sell1_idx_end'] = index_loc[2] if len(index_loc) > 3 else df.index.max()
    
    for i in df[df.brl_sell2_idx_loc.notna()].index:
        brl2_idx = df.loc[i, 'brl_sell2_idx_loc']  # This is the index of the actual brl
        index_loc = df[
                    (df.close > (df.loc[brl2_idx, 'low'] + df.loc[i, 'atr'] * 1.5))
                    &
                    (df.index > i)
                    ].index
        df.loc[i, 'brl_sell2_idx_end'] = index_loc[2] if len(index_loc) > 3 else df.index.max()
    
    #   -- HOUSEKEEPING --
    # this shiz no work
    
    # In any BRL 'zone' I want the BRL lines to not pierce sequential BRL peaks, so the 
    # older BRL must be higher than the newer. To keep it simple, I'll just delete any
    # older BRL whose high is lower than the newer
    # rows = df.index[1] - df.index[0]
    # for i in df.brl_buy1_idx_loc[df.brl_buy1_idx_loc.notna()]:
    #     print('runnin 1s at', i)
    #     view = df[i + 1 * rows : i + 12 * rows]
    #     # print(i)
    #     # print(view)
    #     # print('\n')
    #     more_recent_brl1_idx = view.brl_buy1_idx_loc[view.brl_buy1_idx_loc.notna()].values.tolist()
    #     more_recent_brl2_idx = view.brl_buy2_idx_loc[view.brl_buy2_idx_loc.notna()].values.tolist()
    #     # Iter thru any occurences and compare highs
    #     # print('more_recent_brl1_idx')
    #     # print(more_recent_brl1_idx)
    #     # print(more_recent_brl2_idx)
    #     # print('more_recent_brl2_idx')
    #     # print('\n')
    #     for j in more_recent_brl1_idx + more_recent_brl2_idx:
    #         # print('in j')
    #         # print('i', i, 'high', df.loc[i, 'high'])
    #         # print('j', j, 'high', df.loc[j, 'high'])
    #         if df.loc[i, 'high'] < df.loc[j, 'high']:
    #             print('nanning that 1 at', i)
    #             df.loc[i, 'brl_buy1_idx_loc'] = np.nan
    #             df.loc[i, 'brl_buy1_idx_end'] = np.nan
    #             df.loc[i, 'brl_buy2_idx_loc'] = np.nan
    #             df.loc[i, 'brl_buy2_idx_end'] = np.nan

    # for i in df.brl_buy2_idx_loc[df.brl_buy2_idx_loc.notna()]:
    #     print('runnin 2s at', i)
    #     view = df[i + 1 * rows : i + 12 * rows]
    #     more_recent_brl1_idx = view.brl_buy1_idx_loc[view.brl_buy1_idx_loc.notna()].values.tolist()
    #     more_recent_brl2_idx = view.brl_buy2_idx_loc[view.brl_buy2_idx_loc.notna()].values.tolist()
    #     # Iter thru any occurences and compare highs
    #     # print('more_recent_brl1_idx')
    #     # print(len(more_recent_brl1_idx))
    #     # print(len(more_recent_brl2_idx))
    #     # print('more_recent_brl2_idx')
    #     for j in more_recent_brl1_idx + more_recent_brl2_idx:
    #         if df.loc[i, 'high'] < df.loc[j, 'high']:
    #             print('nanning that 2 at', i)
    #             df.loc[i, 'brl_buy1_idx_loc'] = np.nan
    #             df.loc[i, 'brl_buy1_idx_end'] = np.nan
    #             df.loc[i, 'brl_buy2_idx_loc'] = np.nan
    #             df.loc[i, 'brl_buy2_idx_end'] = np.nan
    

    # Now all the same but for the down thrusts
        
    # for end_idx in down_end_idxs:
    #     start_idx = df[
    #                 (df.thrust_down == 'start')
    #                 &
    #                 (df.index < end_idx)
    #                 ].index.max()

    #     df.loc[end_idx, 'thrust_down'] = np.nan if pd.isnull(start_idx) else 'end'
    #     df.loc[end_idx, 'thrust_50'] = df.loc[start_idx, 'high'] - (df.loc[start_idx, 'high'] - df.loc[end_idx, 'low']) / 2

    # for start in down_starts:

    #     price_diff = (df.loc[start, 'high'] - df.loc[start, 'thrust_50']) / df.loc[start, 'atr']
    #     historical_limit = round(abs(price_diff) * lookback)  # <<< this nearly always rounds to either 0, 1 or 2
    #     rows = df.index[1] - df.index[0]
    #     historical_limit *= rows
        
    #     index_loc = df[
    #             (df.up_peak == 'start_low')
    #             &
    #             (df.low >= df.loc[start, 'thrust_50'])
    #             &
    #             (df.low < df.loc[start, 'high'])
    #             &
    #             (df.index < start)
    #             &
    #             (df.index > start - historical_limit)
    #             ] = df.index.max()
    #     df.loc[start, 'brl_sell'] = index_loc

    # # K now reset the end points of each BRL to the row in which price has
    # # pierced them by some amount
    # for i in df[df.brl_buy.notna()].index:
    #     index_loc = df[
    #                 (df.close < (df.loc[i, 'high'] - df.loc[i, 'atr'] * 1.5))
    #                 &
    #                 (df.index > df[ # find the first "end" occuring after the BRL
    #                                 (df.thrust_up == 'end')
    #                                 &
    #                                 (df.index > i)
    #                                 ].index.min())
    #                 ].index.min()
    #     df.loc[i, 'brl_buy'] = index_loc

    # for i in df[df.brl_sell.notna()].index:
    #     index_loc = df[
    #                 (df.close > (df.loc[i, 'low'] + df.loc[i, 'atr'] * 1.5))
    #                 &
    #                 (df.index > df[
    #                                 (df.thrust_down == 'end')
    #                                 &
    #                                 (df.index > i)
    #                                 ].index.min())
    #                 ].index.min()
    #     df.loc[i, 'brl_sell'] = index_loc
    ''' i need to have info about the thrust, but since i set the brl's on their own I dont know
    which rows thurst end I need to look for. I may need to create a df copy to keep all this data
    in different columns on the same row as start that way I can access and params easily'''


# test
# df = mt5_ohlc_request('audcad','M15', num_candles=650)  # << broke on this one suddenly
df = mt5_ohlc_request('eurusd','M5', num_candles=650)
atr(df)
find_simple_peaks(df)
find_thrusts(df)
find_brl_levels(df, 15)

print(df[(df.brl_buy2_idx_loc.notna()) | (df.brl_sell2_idx_loc.notna())])

import mplfinance as mpf

buy_lines = []
sell_lines = []
# print(len(df[df.brl_buy.notna()])) | (df.brl_buy2_idx.notna())])
for i in df[df.brl_buy1_idx_loc.notna()].index:
    # The brl data is at the index of a thrust "end." However, the data which those 
    # rows hold are the index locations pointing to where the actual BRL happened
    # so to get that plot price I need to....
    brl_loc = df.loc[i, 'brl_buy1_idx_loc']
    price = df.loc[brl_loc, 'high']

    # buy_lines is the line that will plot
    buy_lines.append([(i, price), (df.loc[i, 'brl_buy1_idx_end'], price)]) 
    # this is the actual BRL peak
    df.loc[brl_loc, 'brls_buy_peak'] = df.loc[brl_loc, 'high']

for i in df[df.brl_buy2_idx_loc.notna()].index:
    brl_loc = df.loc[i, 'brl_buy2_idx_loc']
    price = df.loc[brl_loc, 'high']
    buy_lines.append([(i, price), (df.loc[i, 'brl_buy2_idx_end'], price)]) 
    df.loc[brl_loc, 'brls_buy_peak'] = df.loc[brl_loc, 'high']

for i in df[df.brl_sell1_idx_loc.notna()].index:
    brl_loc = df.loc[i, 'brl_sell1_idx_loc']
    price = df.loc[brl_loc, 'low']
    sell_lines.append([(i, price), (df.loc[i, 'brl_sell1_idx_end'], price)]) 
    df.loc[brl_loc, 'brls_sell_peak'] = df.loc[brl_loc, 'low']
for i in df[df.brl_sell2_idx_loc.notna()].index:
    brl_loc = df.loc[i, 'brl_sell2_idx_loc']
    price = df.loc[brl_loc, 'low']
    sell_lines.append([(i, price), (df.loc[i, 'brl_sell2_idx_end'], price)]) 
    df.loc[brl_loc, 'brls_sell_peak'] = df.loc[brl_loc, 'low']

# for i in df[df.brl_sell.notna()].index:
#     sell_lines.append([(i, df.loc[i, 'low']), (df.loc[i, 'brl_sell'], df.loc[i, 'low'])])  
#     buy_lines.append([(i, df.loc[i, 'low']), (df.loc[i, 'brl_sell'], df.loc[i, 'low'])])  

#  
# df['brls_s'] = df.low[df.brl_sell.notna()]

df['up_start'] = df.low[df.thrust_up == 'start']
df['up_end'] = df.high[df.thrust_up == 'end']
df['up'] = df.close[df.thrust_up == 'up']

df['down_start'] = df.high[df.thrust_down == 'start']
df['down_end'] = df.low[df.thrust_down == 'end']
df['down'] = df.close[df.thrust_down == 'down']

# df['start_low'] = df.high[df.up_peak == 'start_low']
# df['start_low'] = df.high[df.up_peak == 'start_low']
plots = [
        mpf.make_addplot(df['brls_buy_peak'],type='scatter',color='g'),
        mpf.make_addplot(df['brls_sell_peak'],type='scatter',color='r'),
        # mpf.make_addplot(df['brls_s'],type='scatter',color='r'),
        # mpf.make_addplot(df['up_start'],type='scatter',color='b', marker='^'),
        # mpf.make_addplot(df['up_end'],type='scatter',color='b', marker='v'),
        # mpf.make_addplot(df['up'],type='scatter',color='b'),

        # mpf.make_addplot(df['down_start'],type='scatter',color='r', marker='v'),
        # mpf.make_addplot(df['down_end'],type='scatter',color='r', marker='^'),
        # mpf.make_addplot(df['down'],type='scatter',color='r'),
]

# make colors
c = ['g' for _ in range(len(buy_lines))]
sellc = ['r' for _ in range(len(sell_lines))]
c += sellc
mpf.plot(df, type='candle', tight_layout=True, 
        show_nontrading=False, volume=False,
        addplot=plots,
        alines=buy_lines + sell_lines,
        # alines=dict(buy_lines, colors=c)
)
        # savefig=f'{symbol}_{timeframe}.png')
# print(df.head(40))
quit()




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


def _trade_scanner(timeframe, bot=bot):
    ''' Read the database for the current timeframe. This function gets
    called within a loop so only a single timeframe will be passed. '''

    forecast_df = pd.read_sql('SELECT * FROM outlook', econ_con)

    
    symbols = mt5_symbols['majors'] + mt5_symbols['others']
    for symbol in symbols:

        # Verify there's an upcoming forecast and see what the score is
        base = symbol[:3]
        counter = symbol[-3:]
        if base or counter in forecast_df.ccy:

            df = mt5_ohlc_request(symbol, timeframe)

            df['pattern'] = np.nan

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

                # Make sure forecasts agree
                base_sum = forecast_df.forecast[forecast_df.ccy == base].sum()
                counter_sum = forecast_df.forecast[forecast_df.ccy == counter].sum()
                long_ok = base_sum > 0 or counter_sum < 0
                short_ok = base_sum < 0 or counter_sum > 0

                ok_to_trade = (pattern[-1] == 'b' and long_ok) or (pattern[-1] == 's' and short_ok)
                if ok_to_trade:
                    print(symbol, timeframe, pattern)
                    # rather than enter the trade here, send that df row back to 'main'
                    # ... but i can't just return the row andbreak the loop
                    send_trade_alert(symbol, timeframe, pattern)
                    bot.send_message(chat_id=446051969, text=f'{symbol} {timeframe} {pattern}')

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