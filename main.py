import concurrent.futures
import sqlite3
import pandas as pd
import time
from datetime import datetime
from ohlc_request import save_raw_ohlc, _timeframes_to_request
from ff_calendar_request import forecast_handler
from create_db import ohlc_db, setups_db, econ_db
from entry_signals import scan_trades
from tokens import bot, fxcm_con


ohlc_con = sqlite3.connect(ohlc_db)
setups_con = sqlite3.connect(setups_db)
econ_con = sqlite3.connect(econ_db)

def bot_message(*args):

    bot.send_message(chat_id=446051969, text=[arg for arg in args])
    # bot.send_message(chat_id=446051969, text=f'{symbol} {timeframe} {pattern}')


# Get info like spread and pip value
symbol_info = fxcm_con.get_offers(kind='dataframe')
account_info = fxcm_con.get_accounts_summary()


def spread_is_ok(df=df, symbol_info=symbol_info, symbol=symbol):
    ''' Return True if the spread is less than 1/3 of the current ATR. '''
    
    print('spread')
    print(symbol_info.spread[symbol_info.currency == symbol])
    spread = symbol_info.spread[symbol_info.currency == symbol].values[0]
    if spread < df.atr.tail(1).values[0] * 1/3:
        return True
    else:
        return False

def _lot_size(risk, df=df, symbol_info=symbol_info, account_info=account_info, symbol=symbol):
    r''' Set lot size. 1 = 1k (aka 0.01 lots). 
    Enter the risk % as a decimal for the first arg. '''

    # get the pip value
    tick_val = symbol_info.pipCost[symbol_info.currency == symbol].values[0]
    
    # get the distance from entry to sl
    distance = abs(df.entry.tail(1).values[0] - df.sl.tail(1).values[0])
    
    # min loss
    loss_with_min_lot = distance * tick_val

    # Divide risk per trade by loss_with_min_lot
    risk_per_trade = risk * account_info.equity.values[0]

    lot_size = risk_per_trade // loss_with_min_lot
    
    return lot_size

def _expiration(df, num_candles=4):
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

def enter_trade(df, tp):
    ''' Read from the historical setups db and set lot size based on 
    the forecast rating for the symbol. '''

    for i in df.index:

        # Check how many forecasts exists

    if tp == 'tp1':
        tp = df.loc[i, 'tp1']

    if tp == 'tp2':
        tp = df.loc[i, 'tp1']


    fxcm_con.create_entry_order(symbol=df.loc[i, 'symbol'],
                                is_buy=_buy_or_sell(),
                                amount=_lot_size(0.01), 
                                is_in_pips = False,
                                time_in_force='GTD', 
                                expiration=_expiration(),
                                rate=df.entry.tail(1).values[0], 
                                limit=tp,
                                stop=df.sl.tail(1).values[0]
                                )


''' Central controller.  I'll set up each function to run on its own process for speed
    because each of the imported functions have a ton of other function calls they 
    make behind the scenes.  No two processes will be writing to the same database at any point. 
    The forecasts and trade signals each write to the central ratings_df which is then queried 
    for potentil trade entries.'''

# Update the economic forecasts every hour 
# (the database will get updated on Saturdays)
def update_forecasts():
    ''' Just calls the forecast handler from the cal request module '''

    # Run every hour
    time.sleep(60*60)

    # A df with cols: ccy, ccy_event, event_datetime, forecast, monthly
    forecasts = forecast_handler()



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
        db = pd.read_sql(r"SELECT datetime, timeframe FROM 'EUR/USD'", conn, parse_dates=['datetime'])

        for tf in timeframes:
            datetime = db.datetime[db.timeframe == tf]
            latest_datetime = datetime.values[-1]
            
            # See if any are newer than what's in the 'times' dict
            if latest_datetime > latest_timestamps[tf]:

                # Update dict
                latest_timestamps[tf] = latest_datetime

                # Scan for trade setups on that timeframe
                return scan_trades(tf)

def save_setups():
    ''' Reads the setup_db for trade setups and checks the forecast for
    currencies which currently have setups listed.  The forecast data will
    be added to the setup_db and then that combined data will be saved to the
    historical table. '''


    currencies = [
        'USD',
        'EUR',
        'GBP',
        'JPY',
        'CAD',
        'AUD',
        'NZD',
        'CHF',
        'CNY'
    ]

    # Open current setups df
    entry_df = trade_scan_handler()

    # Open forecasts df (this is current data only)
    forecast_df = pd.read_sql('outlook', econ_con)

    # Loop thru the forecasts
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
