import sqlite3
import os
import pathlib
from create_db import ohlc_db, econ_db
from tokens import bot
from corr_value_scanner import find_correlations
from indexes import build_in_memory_database, run_calculation_processes
from ff_calendar_request import verify_db_tables_exist,  forecast_handler
from symbols_lists import indexes

''' Central controller.  I'll set up each function to run on its own process for speed
    because each of the imported functions have a ton of other function calls they 
    make behind the scenes.  No two processes will be writing to the same database at any point. 
    The forecasts and trade signals each write to the central ratings_df which is then queried 
    for potentil trade entries.'''



# ^^ iono
if __name__ == '__main__':

    # Calendar
    verify_db_tables_exist()

    while True:

        # Calendar
        forecast_handler()

        # Correlation
        find_correlations(historical_fill=True)

        # Indexes
        database = build_in_memory_database(mt5_symbols['majors'], days=30, _from=None) 
        run_calculation_processes(database, indexes, final_period='5 min')

        # Value
        calculate_value_line_from_correlations()
        value_line_scores = value_line_rating_scanner()
        top_pairs = [k for (k,v) in value_line_scores.items() if v > 0.8 and v < .95]
        plot_charts(top_pairs, 'H4', include_forecasts=True, num_candles=100)

        # Fire ze photoz!  (chat id correct?)
        pics = []
        for filename in os.listdir(r'C:\Users\ru'):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                pics.append(os.path.join(directory, filename))
        
        # Send indexes first and pairs second
        pairs = []
        for pic in pics:
            name = pic.splt('.')[0]
            if any(index in name for index in indexes):
                bot.send_photo(chat_id=446051969, photo=open(pic, 'rb'))
            else:
                pairs.append(pic)
        for pic in pairs:
            bot.send_photo(chat_id=446051969, photo=open(pic, 'rb'))
                
