import concurrent.futures
import sqlite3
import pandas as pd
import time
from datetime import datetime
from create_db import ohlc_db, setups_db, econ_db
from entry_signals import scan_trades
from tokens import bot

# These 2 functions are end points for their modules
from ohlc_request import ohlc_request_handler
from ff_calendar_request import forecast_handler
from entry_signals import trade_scan_handler

''' Central controller.  I'll set up each function to run on its own process for speed
    because each of the imported functions have a ton of other function calls they 
    make behind the scenes.  No two processes will be writing to the same database at any point. 
    The forecasts and trade signals each write to the central ratings_df which is then queried 
    for potentil trade entries.'''

ohlc_con = sqlite3.connect(ohlc_db)
setups_con = sqlite3.connect(setups_db)
econ_con = sqlite3.connect(econ_db)




# this function was giving me errors
# def spread_is_ok(df=df, symbol_info=symbol_info, symbol=symbol):
#     ''' Return True if the spread is less than 1/3 of the current ATR. '''
    
#     print('spread')
#     print(symbol_info.spread[symbol_info.currency == symbol])
#     spread = symbol_info.spread[symbol_info.currency == symbol].values[0]
#     if spread < df.atr.tail(1).values[0] * 1/3:
#         return True
#     else:
#         return False

def _send_telegram_message(*args):

    bot.send_message(chat_id=446051969, text=[arg for arg in args])
    # bot.send_message(chat_id=446051969, text=f'{symbol} {timeframe} {pattern}')

    
def _lot_size(risk, df, i, symbol_info=symbol_info, account_info=account_info):
    r''' Set lot size. 1 = 1k (aka 0.01 lots). 
    Enter the risk % as a decimal for the first arg. '''

    symbol = df.loc[i, 'symbol']

    # get the pip value
    tick_val = symbol_info.pipCost[symbol_info.currency == symbol].values[0]
    
    # get the distance from entry to sl
    distance = abs(df.loc[i, 'entry'] - df.loc[i, 'sl'])
    
    # min loss
    loss_with_min_lot = distance * tick_val

    # Divide risk per trade by loss_with_min_lot
    risk_per_trade = risk * account_info.equity.values[0]

    lot_size = risk_per_trade // loss_with_min_lot
    
    return lot_size

def _expiration(df, i, num_candles=4):
    ''' Set the order expiration at n candles, so it will depend on the timeframe. '''

    timeframe = df.loc[i, 'timeframe']

    if timeframe == 'D':
        t_delta = pd.Timedelta('1 day')
    else:
        t_delta = pd.Timedelta(f'{timeframe} min')

    e = datetime.now() + num_candles * t_delta

    return e

def _buy_or_sell(df, i):
    ''' Return True if trade type is buy, False otherwise. '''

    entry = df.loc[i, 'entry']
    stop_loss = df.loc[i, 'sl']
    
    if entry > stop_loss:
        return True
    else:
        return False

def _enter_trade(entry_df, forecast_df):
    ''' Read from the df of current setups and set lot size based on 
    the forecast rating for the currencies in the symbol. '''

    for i in entry_df.index:

        # Get base and counter currencies
        symbol = entry_df.loc[i, 'symbol']
        base_ccy = symbol[:3]
        counter_ccy = symbol[-3:]


        # Check the forecasts
        base_sum = sum(forecast_df.weekly[forecast_df.ccy == base_ccy])
        counter_sum = sum(forecast_df.weekly[forecast_df.ccy == counter_ccy])

        # Check if trade is long or short
        long = _buy_or_sell(entry_df, i)

        # lot size  multiplier
        x = 0.01
        if long:
            if base_sum > 0 and counter_sum < 0:
                x *= 2
            elif base_sum > 0 or counter_sum < 0:
                x *= 1.5
            
            # if totally opposite forecast pass on the trade
            elif base_sum < 0 and counter_sum > 0:
                continue
        
        # If it's a short trade reverse the forecast numbers
        if not long:
            if base_sum < 0 and counter_sum > 0:
                x *= 2
            elif base_sum < 0 or counter_sum > 0:
                x *= 1.5
            
            # if totally opposite forecast pass on the trade
            elif base_sum > 0 and counter_sum < 0:
                continue

    # TP1
    fxcm_con.create_entry_order(symbol=symbol,
                                is_buy=_buy_or_sell(entry_df, i),
                                amount=_lot_size(x, entry_df, i), 
                                is_in_pips = False,
                                time_in_force='GTD', 
                                expiration=_expiration(entry_df, i),
                                rate=entry_df.loc[i, 'entry'], 
                                limit=entry_df.loc[i, 'tp1'],
                                stop=entry_df.loc[i, 'entry'],
                                )
    # TP2
    fxcm_con.create_entry_order(symbol=symbol,
                                is_buy=_buy_or_sell(entry_df, i),
                                amount=_lot_size(x, entry_df, i), 
                                is_in_pips = False,
                                time_in_force='GTD', 
                                expiration=_expiration(entry_df, i),
                                rate=entry_df.loc[i, 'entry'], 
                                limit=entry_df.loc[i, 'tp2'],
                                stop=entry_df.loc[i, 'entry'],
                                )

    tf = entry_df.loc[i, 'timeframe']
    pattern = entry_df.loc[i, 'pattern']
    _send_telegram_message(symbol, tf, pattern)

def send_orders_and_record_setups():
    ''' Reads the current trade setups and forecasts.  Handles the placing of trades
    and also saves pertinent info to a database. '''

    while True:

        # Open current setups df
        entry_df = trade_scan_handler()

        if len(entry_df) > 0:

            # Open forecasts df (this is current data only)
            forecast_df = pd.read_sql('outlook', econ_con)

            _enter_trade(entry_df, forecast_df)

            # Now for the saving to the database....
            for i in forecast_df.index:

                ccy = forecast_df.loc[i, 'ccy']

                # Go thru each row of entry df and see if that forecast matches the base or
                # counter ccy of any entry setups, then save
                for ii in entry_df.index:

                    # First 3 letters or last 3
                    base = entry_df.loc[ii, 'symbol'][:3]
                    counter = entry_df.loc[ii, 'symbol'][-3:]

                    if ccy in base:
                        entry_df.loc[ii, 'base_ccy_weekly'] = forecast_df.loc[i, 'weekly']
                        entry_df.loc[ii, 'base_ccy_monthly'] = forecast_df.loc[i, 'monthly']

                    if ccy in counter:
                        entry_df.loc[ii, 'counter_ccy_weekly'] = forecast_df.loc[i, 'weekly']
                        entry_df.loc[ii, 'counter_ccy_monthly'] = forecast_df.loc[i, 'monthly']

            entry_df.to_sql('historical', setups_con, if_exists='append', index=False)


# Create separate threads for each server
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     t1 = executor.submit(_fxcm(timeframes))
#     t2 = executor.submit(_finnhub(timeframes))

# This function is self contained with its timer and databse building functions
