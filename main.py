import concurrent.futures
import sqlite3
import pandas as pd
import time
from datetime import datetime
from create_db import ohlc_db, econ_db
from tokens import bot

# These 2 functions are end points for their modules
# from ohlc_request import ohlc_request_handler
# from ff_calendar_request import forecast_handler  # importing this will call it into an infinite loop
from entry_signals import trade_scanner
from ohlc_request import timeframes_to_request

''' Central controller.  I'll set up each function to run on its own process for speed
    because each of the imported functions have a ton of other function calls they 
    make behind the scenes.  No two processes will be writing to the same database at any point. 
    The forecasts and trade signals each write to the central ratings_df which is then queried 
    for potentil trade entries.'''

ohlc_con = sqlite3.connect(ohlc_db)
# setups_con = sqlite3.connect(setups_db)
econ_con = sqlite3.connect(econ_db)


def test():  

    while True:
        
        timeframes = timeframes_to_request()
        # times = 
        #     'test': [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M2, mt5.TIMEFRAME_M3, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M10, mt5.TIMEFRAME_M15, 
        #         mt5.TIMEFRAME_M30, mt5.TIMEFRAME_H1],
        # }

        if timeframes:
        # Open current setups df
            for t in timeframes:
                trade_scanner(t)
                # Make sure to not rerun the same timeframe while its still active
                time.sleep(59)
                

test()


def send_orders_and_record_setups():
    ''' Reads the current trade setups and forecasts.  Handles the placing of trades
    and also saves pertinent info to a database. '''

    while True:
        
        timeframe = timeframes_to_request()

        # Open current setups df
        entry_df = trade_scanner(timeframe)

        if len(entry_df) > 0:

            # Open forecasts df (this is current data only)
            forecast_df = pd.read_sql('SELECT * FROM outlook', econ_con)

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



# # # Create separate threads for each server
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     t1 = executor.submit(forecast_handler)
#     t2 = executor.submit(test)
