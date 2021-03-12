import pandas as pd
import sqlite3
from datetime import datetime
import time
from create_db import value_db, correlation_db
from symbols_lists import mt5_symbols, indexes
from indexes import _invert


CORR_CON = sqlite3.connect(correlation_db)
VAL_CON = sqlite3.connect(value_db)


''' 
The goal of this module is to read the raw correlation data and derive buy/sell values from it.
I'll do this by pair and also by index.  So once I have the data for each pair, I'll then
combine it similar to how I made the indexes, inverting when I need to and adding them all together.
There are up to 3 sets of data for each pair and I'll want to keep them separated throughout.
'''

timeframes = [
    '_LTF',
    '_MTF',
    '_HTF',
]

def set_pair_value():
    # I don't know how to get a list of table names from the db so..
    # Iter each pair and try to open a table from each timeframe
    for pair in mt5_symbols['majors']:

        # This will end up holding the LTF, MTF and HTF value columns
        df = pd.DataFrame()

        for tf in timeframes:

            name = pair + tf
            print(name)

            try:
                cor_df = pd.read_sql(f'SELECT * FROM {name}', CORR_CON)
            except:
                continue
            
            if len(cor_df) < 10:
                continue

            cor_df.index = pd.to_datetime(cor_df['index'])

            # Parse the column names because I need to multiply each symbols column by its corr column
            # I left myself some keys so I could easily grab what I need
            things_to_omit = [r'*', 'corr', 'index']
            cols = cor_df.columns.tolist()

            new_cols = []
            for col in cols:
                if all(x not in col for x in things_to_omit):
                    new_cols.append(col)
            
            # Multiply the values
            temp_df = pd.DataFrame()
            for col in new_cols:
                temp_df[f'{col}'] = cor_df[f'{col}'] * cor_df[f'{col}_corr']
            
            temp_df = temp_df.fillna(0)
            cor_sum = temp_df.sum(axis=1)

            # This is what will be saved to the db
            df[f'{tf[1:]}'] = round(cor_sum - cor_df[fr'*{pair}*'], 4)
            
        df.to_sql(pair, VAL_CON, if_exists='replace', index=True)

            # time.sleep(99)

def set_index_value():
    ''' Using the data that was calculated in the pair function, combine
    now into index value scores. Some of the same steps from making the OHLC 
    indexes will exist here. '''

    # This will store the combined data of each timeframe
    indexes = {
        'USD': {},
        'EUR': {},
        'GBP': {},
        'JPY': {},
        'AUD': {},
        'NZD': {},
        'CAD': {},
        'CHF': {},
    }

    # These will store the data for any column found in each pair's data.
    # This data will get overwritten with each pair
    tf_data = {
        'LTF': pd.Series(dtype=float),
        'MTF': pd.Series(dtype=float),
        'HTF': pd.Series(dtype=float)
        }

    # Add the keys to the indexes now to avoid errors later
    for ccy in indexes:
        for tf in tf_data:
            indexes[ccy][tf] = tf_data[tf] 

    print(indexes)

    # Iter each index and add applicable pair values to it
    for pair in mt5_symbols['majors']:
        
        # The table might not exist
        try:
            df = pd.read_sql(f'SELECT * FROM {pair}', VAL_CON)
        except:
            continue

        df = df.set_index(df['index'], drop=True)
        df.index = pd.to_datetime(df.index)
        df = df.drop(columns='index')
        
        # Split up the L/M/HTF columns, assuming they exist
        cols = df.columns.tolist()

        for i in range(0, len(cols)):
            tf_data[cols[i]] = df[cols[i]]



        # Add each tf's data to its proper dict currency, inverting if necessary
        for ccy in indexes:
            for tf in tf_data:

                # Base currency
                if ccy == pair[:3]:
                    if len(indexes[ccy][tf]) == 0:
                        indexes[ccy][tf] = tf_data[tf]
                    else:
                        indexes[ccy][tf] += tf_data[tf]
                    continue
                
                # Counter currency
                elif ccy == pair[-3:]:
                    if len(indexes[ccy][tf]) == 0:
                        indexes[ccy][tf] = tf_data[tf] * -1
                    else:
                        indexes[ccy][tf] += tf_data[tf] * -1
    print(indexes)
set_index_value()
