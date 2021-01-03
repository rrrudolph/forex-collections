# import pymt5adapter as mt5
import MetaTrader5 as mt5

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import time
import telegram
# import sys
# sys.path.append('C:/Program Files (x86)/Python38-32/code')
# import format_functions
# import trades

bot = telegram.Bot(token='1371103620:AAG_6iRGzmeGDTd0V-W2gPTIavI-gVSPolA')
df = pd.DataFrame()

# acct_info = ''
# symb_info = ''

def open_trade(df, i, symb_info, acct_info):
    if df.loc[i, 'pattern'][-2:] == '_b':
        trade_type = mt5.ORDER_TYPE_BUY_STOP
    if df.loc[i, 'pattern'][-2:] == '_s':
        trade_type = mt5.ORDER_TYPE_SELL_STOP
    # lot = lot_size(0.02, acct_info, symb_info)
    lot = 0.1
    deviation = 20
    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lot,
        "type": trade_type,
        "price": df.loc[i, 'entry'],
        "sl": df.loc[i, 'sl'],
        "tp": df.loc[i, 'tp1'],
        "deviation": deviation,
        "magic": 234000,
        "comment": "",
        "type_time": mt5.ORDER_TIME_DAY,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
 
    # send a trading request
    result = mt5.order_send(request)

    # now make the second entry
    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": lot,
        "type": trade_type,
        "price": df.loc[i, 'entry'],
        "sl": df.loc[i, 'sl'],
        "tp": df.loc[i, 'tp2'],
        "deviation": deviation,
        "magic": 234000,
        "comment": "",
        "type_time": mt5.ORDER_TIME_DAY,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)

def spread_is_ok(sym, df, symb_info):
    i = df.index.max() - 1 
    spread = symb_info['spread'] * symb_info['point']
    if spread < df.loc[i, 'atr'] * 1/3:
        return True
    else:
        return False

def lot_size(risk, acct_info, symb_info):
    # get the point value
    tick_val = symb_info['trade_tick_value']
    # get the distance from entry to sl
    distance = df.loc[i, 'entry'] - df.loc[i, 'sl']
    # min loss
    loss_with_min_lot = distance * tick_val
    # Divide risk per trade by loss_with_min_lot
    risk_per_trade = risk * acct_info['equity']
    lot_size = round(risk_per_trade / loss_with_min_lot, 2)
    
    return lot_size


##### Format functions ######
# formatting within a function doesn't work for some reason
def format_data(df):
    df = df[:-1] # sometimes calculating the current bar throws errors
    df = df.rename(columns={'time': 'dt', 'tick_volume': 'volume'})
    df.dt = pd.to_datetime(df.dt, unit='s')
    df['pattern'] = np.nan
    df['peak'] = np.nan

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
    dfdf = df.loc[-3:].copy()
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

#### SYMBOLS #####

symbols = [
    'EURUSD',
    'GBPUSD',
    'USDJPY',
    'NZDUSD',
    'USDCAD',
    'AUDUSD',
    'EURJPY',
    'GBPJPY',
    'CADJPY',
    'AUDJPY',
    'NZDJPY',
    'EURCAD',
    'EURAUD',
    'GBPCAD',
    'EURGBP',
    'AUDCAD',
    'NZDCAD',
    'AUDNZD',
    'GBPAUD',
    'GBPNZD',
    'EURNZD',
    'USDCHF',
    'GBPCHF',
    'CADCHF',
    'USDPLN',
    'USDSEK',
    'USDMXN',
    'USDZAR',
    'GBPSGD',
    'USDZAR',
    'XAUUSD',
    'XAGUSD',
    'XTIUSD',
    'DE30',
    'US30',
    'US500',
    'USTEC'
]

# all_symbols = ['EURUSD', 'GBPUSD', 'USDCHF', 'USDJPY', 'USDCAD', 'AUDUSD', 'AUDNZD', 'AUDCAD', 'AUDCHF', 'AUDJPY', 'CHFJPY', 'EURGBP', 'EURAUD', 'EURCHF', 'EURJPY', 'EURNZD', 'EURCAD', 'GBPCHF', 'GBPJPY', 'CADCHF', 'CADJPY', 'GBPAUD', 'GBPCAD', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD', 'USDSGD', 'AUDSGD', 'CHFSGD', 'EURDKK', 'EURHKD', 'EURNOK', 'EURPLN', 'EURSEK', 'EURSGD', 'EURTRY', 'EURZAR', 'GBPDKK', 'GBPNOK', 'GBPSEK', 'GBPSGD', 'GBPTRY', 'NOKJPY', 'NOKSEK', 'SEKJPY', 'SGDJPY', 'USDCNH', 'USDCZK', 'USDDKK', 'USDHKD', 'USDHUF', 'USDMXN', 'USDNOK', 'USDPLN', 'USDRUB', 'USDSEK', 'USDTHB', 'USDTRY', 'USDZAR', 'AUS200', 'CHINA50', 
#                 'DE30', 'UK100', 'US2000', 'US30', 'US500', 'USTEC', 'XAGEUR', 'XAGUSD', 'XAUEUR', 'XAUUSD', 'XPDUSD', 'XPTUSD', 'XBRUSD', 'XNGUSD', 'XTIUSD', 'BTCUSD', 'BCHUSD', 'DSHUSD', 'ETHUSD', 'LTCUSD', 'AAPL.NAS', 'ADBE.NAS', 'ADI.NAS', 'ADP.NAS', 'AMAT.NAS', 'AMGN.NAS', 'AMZN.NAS', 'ATVI.NAS', 'AVGO.NAS', 'BKNG.NAS', 'BIIB.NAS', 'CELG.NAS', 'CHTR.NAS', 'CMCSA.NAS', 'CME.NAS', 'COST.NAS', 'CRON.NAS', 'CSCO.NAS', 'CSX.NAS', 'CTSH.NAS', 'DISH.NAS', 'EA.NAS', 'EBAY.NAS', 'EQIX.NAS', 'FB.NAS', 'FOX.NAS', 'FOXA.NAS', 'GILD.NAS', 'GOOG.NAS', 'INTC.NAS', 'INTU.NAS', 'ISRG.NAS', 'KHC.NAS', 'MAR.NAS', 'MDLZ.NAS', 'MSFT.NAS', 'MU.NAS', 
#                 'NFLX.NAS', 'NVDA.NAS', 'PEP.NAS', 'PYPL.NAS', 'QCOM.NAS', 'REGN.NAS', 'SBUX.NAS', 'SHPG.NAS', 'TFCF.NAS', 'TLRY.NAS', 'TMUS.NAS', 'TSLA.NAS', 'TXN.NAS', 'VRTX.NAS', 'WBA.NAS', 'ABBV.NYSE', 'ABT.NYSE', 'ACB.NYSE', 'AGN.NYSE', 'AXP.NYSE', 'BA.NYSE', 'BAC.NYSE', 'BLK.NYSE', 'BMY.NYSE', 'C.NYSE', 'CGC.NYSE', 'CVS.NYSE', 'CVX.NYSE', 'DD.NYSE', 'DIS.NYSE', 'DWDP.NYSE', 'GE.NYSE', 'GS.NYSE', 'HD.NYSE', 'HON.NYSE', 'IBM.NYSE', 'JNJ.NYSE', 'JPM.NYSE', 'KO.NYSE', 'LLY.NYSE', 'LMT.NYSE', 'MA.NYSE', 'MCD.NYSE', 'MMM.NYSE', 'MO.NYSE', 'MRK.NYSE', 'MS.NYSE', 'NKE.NYSE', 'ORCL.NYSE', 'PFE.NYSE', 'PG.NYSE', 'PM.NYSE', 
#                 'SLB.NYSE', 'SPOT.NYSE', 'T.NYSE', 'TWX.NYSE', 'UNH.NYSE', 'UNP.NYSE', 'UPS.NYSE', 'USB.NYSE', 'UTX.NYSE', 'V.NYSE', 'VZ.NYSE', 'WFC.NYSE', 'WMT.NYSE', 'XOM.NYSE', 'XRPUSD', 'EOSUSD', 'EMCUSD', 'NMCUSD', 'PPCUSD', 'DBA.NYSE', 'EEM.NYSE', 'EWH.NYSE', 'EWW.NYSE', 'EWZ.NYSE', 'FXI.NYSE', 'GDX.NYSE', 'GDXJ.NYSE', 'IEMG.NYSE', 'LQD.NYSE', 'MJ.NYSE', 'MOO.NYSE', 'RSX.NYSE', 'SIL.NYSE', 'UNG.NYSE', 'URA.NYSE', 'USO.NYSE', 'VXX.NYSE', 'VXXB.NYSE', 'VYM.NYSE', 'XLE.NYSE', 'XLF.NYSE', 'XLI.NYSE', 'XLP.NYSE', 'XLU.NYSE', 'XOP.NYSE', 'QQQ.NAS', 'TLT.NAS', 'XAUAUD', 'LIN.NYSE', 'DEO.NYSE', 'NEE.NYSE', 
#                 'AMT.NYSE', 'BABA.NYSE', 'BRK-B.NYSE', 'TMO.NYSE', 'LYFT.NAS', 'TM.NYSE', 'UBER.NYSE', 'PINS.NYSE', 'WTI_X0', 'BRENT_Z0', 'VIX_V0', 'DXY_U0', 'DXY_Z0']

#### TIMEFRAMES ####
times = {
    0: [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, 
          mt5.TIMEFRAME_M30, mt5.TIMEFRAME_H1],
    30: [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M30],
    # '20': [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M10, mt5.TIMEFRAME_M20],
    15: [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15],
    # '10': [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M10],
    5: [mt5.TIMEFRAME_M5],
    # '1': [mt5.TIMEFRAME_M1]
}

#### MT5 CONNECTION ####
if not mt5.initialize(login=50341259, server="ICMarkets-Demo",password="ZhPcw6MG"):
    print("initialize() failed, error code =", mt5.last_error())
    quit()
logger = mt5.get_logger(path_to_logfile='my_mt5_log.log', loglevel=logging.DEBUG, time_utc=True)
# mt5_connected = mt5.connected(
#     path=r'C:\Program Files\ICMarkets - MetaTrader 5\terminal64.exe',
#     portable=True,
#     server='ICMarkets-Demo',
#     login=50341259,
#     password='ZhPcw6MG',
#     timeout=5000,
#     logger=logger, # default is None
#     ensure_trade_enabled=True,  # default is False
#     enable_real_trading=True,  # default is False
#     raise_on_errors=True,  # default is False
#     return_as_dict=False, # default is False
#     return_as_native_python_objects=False, # default is False
# )
acct_info = mt5.account_info()._asdict()

try:
    # with mt5_connected as conn:
    while True:
        timeframes = None
        minute = int(str(datetime.now()).split(':')[1])
        if minute in times:
            timeframes = times[minute] # this is to stop a div by 0 error
        else:
            # '15' should happen 4 times per hour so need a modulo function
            for t in times:
                # t > 0 is just to skip the 0 key cuz I couldnt slice times (times[1:] is a no go)
                if t > 0 and minute % t == 0:
                    #print(f'divided {minute} by times[{t}] without error')
                    timeframes: times[t]

        if timeframes:    
            # pause for a sec to ensure the new current candle has printed
            time.sleep(1)
            for timeframe in timeframes:
                for symbol in symbols:
                    print(symbol, timeframe)
                    symb_info = mt5.symbol_info(symbol)._asdict()

                    # If request fails, re-attempt once
                    try:
                        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 70)
                    except:
                        time.sleep(0.03)
                        try:
                            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 70)
                        except:
                            print(f'failed to retrieve {symbol} {timeframe} data')
                            continue
                    df = pd.DataFrame(rates)
                    # format_data(df)    Why doesnt this work here?
                    df = df.rename(columns={'time': 'dt', 'tick_volume': 'volume'})
                    df.dt = pd.to_datetime(df.dt, unit='s')
                    df['pattern'] = np.nan
                    atr(df)
                    if spread_is_ok(symbol, df, symb_info):
                        hlc3(df)
                        set_ema(df)
                        auction_vol(df)
                        avg_vol(df)
                        pinbar(df)
                        avg_bar_size(df)
                        find_peaks(df)
                        fade_b(df)
                        fade_s(df)
                        spring_b(df) # spring seems to be the one causing problems 
                        spring_s(df)
                        stoprun_b(df)
                        stoprun_s(df)

                        # If trades are found set the proper order type
                        i = df.index.max() - 1 
                        if not pd.isnull(df.loc[i, 'pattern']):
                            print("{} {} {}".format(symbol, timeframe, df.loc[i, 'pattern']), datetime.now()) # + pd.to_timedelta('06:00:00')
                            bot.send_message(chat_id=446051969, text='{} {} {}'.format(symbol, timeframe, df.loc[i, 'pattern']))
                            acct_info = mt5.account_info()._asdict()
                            open_trade(df, i, symb_info, acct_info)
                        # else:
                        #     print('{} {} spread is too high'.format(symbol, timeframe))

            # Ensure that same data isn't requested again
            time.sleep(59)
except Exception as e: 
    print('error:', e)
    bot.send_message(chat_id=446051969, text='error')
    
                                