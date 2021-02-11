import concurrent.futures
import pandas as pd
from ohlc_request import save_raw_ohlc, _timeframes_to_request
from ff_calendar_request import forecast_handler
from create_db import ohlc_path
from entry_signals import scan_trades



''' Central controller.  I'll set up each module to run on its own process for speed
    because each of the imported functions have a ton of other function calls they 
    make behind the scenes. '''

# Update the economic forecasts every hour 
# (the database will get updated on Saturdays)
def 



def scan_for_trades():
    ''' A continuous loop that checks the database for the last
    timestamp of each timeframe.  If a newer timestamp appears
    closer to now() than what's been saved, the database has
    been updated and it's time to scan for trades on that timeframe. '''

    # Instantiate the dict which will hold the latest timestamp of each timeframe
    times = {}

    while True:

        # Read the entire database into memory :/
        db = pd.read_sql('ohlc', ohlc_path)
        db.datetime = pd.to_datetime(db.datetime)
        timeframes = db.timeframes.unique()

        if times:
            # Check the last timestamp for each timeframe in the database
            for tf in timeframes:
                datetime = db.datetime[db.timeframe == tf]
                latest_datetime = datetime.values[-1]
                
                # See if any are newer than what's in the 'times' dict
                if latest_datetime > times[tf]:

                    # Scan for trade setups on that timeframe
                    scan_trades(tf)

                    # Update dict
                    times[tf] = latest_datetime

        # Fill 'times' dict
        else:
            for tf in timeframes:
                datetime = db.datetime[db.timeframe == tf]
                last_datetime = datetime.values[-1]
                times[tf] = last_datetime
                




            # Create separate threads for each server
            # with concurrent.futures.ThreadPoolExecutor() as executor:
            #     t1 = executor.submit(_fxcm(timeframes))
            #     t2 = executor.submit(_finnhub(timeframes))

# This function is self contained with its timer and databse building functions
