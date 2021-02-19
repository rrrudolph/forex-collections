import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd
import sqlite3
from tokens import bot
from create_db import econ_db


econ_con = sqlite3.connect(econ_db)

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

    e = datetime.now() + num_candles * t_delta

    return e


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
        _send_telegram_forecast(f'{base_ccy}:{base_sum}  {counter_ccy}:{counter_sum}')

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
    "comment": f"{df.loc[i, 'pattern']}",
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
    "comment": f"{df.loc[i, 'pattern']}",
    "type_time": mt5.ORDER_TIME_DAY,
    "type_filling": mt5.ORDER_FILLING_FOK,
    }


    # send a trading request
    trade = mt5.order_send(request_1)
    trade = mt5.order_send(request_2)

