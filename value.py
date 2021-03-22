import pandas as pd
import sqlite3
from datetime import datetime
import time
from create_db import value_db, correlation_db
from symbols_lists import mt5_symbols


CORR_CON = sqlite3.connect(correlation_db)
VAL_CON = sqlite3.connect(value_db)


''' 
The goal of this module is to read the raw correlation data and derive buy/sell values from it.
I'll do this by pair and also by index.  So once I have the data for each pair, I'll then
combine it similar to how I made the indexes, inverting when I need to and adding them all together.
There are up to 3 sets of data for each pair and I'll want to keep them separated throughout.
The data I have saved from the corr scanner is the normalized price values along with their corr rating.
I should be able to just multiply a symbol's normd values by its corr rating and then get the average of
all those values row by row.
'''

timeframes = [
    '_LTF',
    '_MTF',
    '_HTF',
]

def _add_data_to_existing(existing, new_data):
    ''' When a tf_data[tf] of a different length gets added to what already exists,
    nans will occur on any new index locations.  Those nans need to be overwritten by
    the new values. However some of the tf_data[tf] has nans as well'''

    # # Add new data to existing
    # existing += new_data

    # # If nans appear fill with new data
    # if len(existing[existing.isnull()]) > 0:
    #     existing[existing.isnull()] = new_data[existing.isnull()]

    # return existing

    # i donno why that gives me unalignable index ^
    col = existing.columns[0]
    for row in new_data.itertuples(name=None, index=True):
        i = row[0]
        value = row[1]
        if i in existing.index:
            existing.loc[i, col] += value
        else:
            existing.loc[i,col] = value

    return existing


def set_pair_value():

    # keep the same format as the correlation scanner by itering timeframes and then pairs
    for tf in timeframes:
    
    # I don't know how to get a list of table names from the db so..
    # Iter each pair and try to open a table from each timeframe
        for pair in mt5_symbols['majors']:
            
            name = pair + tf

            # This will be the final, single column df that is saved
            df = pd.DataFrame()

            try:
                cor_df = pd.read_sql(f'SELECT * FROM {name}', CORR_CON)
            except:
                print(f'couldnt open {name}')
                continue
            
            if len(cor_df) < 10:
                print(f'cor_df didnt have any data for {name}')
                continue

            cor_df.index = pd.to_datetime(cor_df['index'])

            # Parse the column names because I need to multiply each symbols column by its corr column
            # I left myself some keys so I could easily grab what I need
            things_to_omit = [r'*', 'corr', 'index', 'shift']
            cols = cor_df.columns.tolist()

            new_cols = []
            for col in cols:
                if all(x not in col for x in things_to_omit):
                    new_cols.append(col)
            
            # Multiply the diff values by the current correlation score
            temp_df = pd.DataFrame()
            for col in new_cols:
                temp_df[f'{col}'] = cor_df[f'{col}'] * cor_df[f'{col}_corr']
            
            # Now I have df of just the symbol names but tons of nans. Before I deal with those
            # I want to save the number of non nan symbols in each row so I can get an average at the end
            non_nans = temp_df.notna().sum(axis=1)

            temp_df = temp_df.fillna(0)
            cor_sum = temp_df.sum(axis=1)

            cor_sum /= non_nans

            # This is what will be saved to the db
            # (I used to subtract the key_symbol values from cor_sum but I didn't end
            # up liking the oscillator type look)
            df[f'{tf[1:]}'] = round(cor_sum, 4)
            
            df.to_sql(name, VAL_CON, if_exists='replace', index=True)

    CORR_CON.close()

def set_index_value():
    ''' Using the data that was calculated in the pair function, combine
    now into index value scores. Some of the same steps from making the OHLC 
    indexes will exist here. '''

  
    # Just like how the corr was scanned I will set the outer loop
    # as the timeframes and scan each pair within a certain timeframe. 
    # then just like the index ohlc data was made I will aggregate
    # the value for each timeframe as I iter thru the pairs

    for tf in timeframes:

         # This will store the combined data of each timeframe
        template = pd.DataFrame(columns=[tf[1:]])
        indexes = {
            'USD': template.copy(),
            'EUR': template.copy(),
            'GBP': template.copy(),
            'JPY': template.copy(),
            'AUD': template.copy(),
            'NZD': template.copy(),
            'CAD': template.copy(),
            'CHF': template.copy(),
        }

        # Iter each index and add applicable pair values to it
        for pair in mt5_symbols['majors']:

            name = pair + tf

            # The table might not exist
            try:
                df = pd.read_sql(f'SELECT * FROM {name}', VAL_CON)
            except:
                continue

            df = df.set_index(df['index'], drop=True)
            df.index = pd.to_datetime(df.index)
            df = df.drop(columns='index')

            # Add each tf's data to its proper dict currency, inverting if necessary
            for ccy in indexes:

                # Ensure the Series is not empty
                if len(df) == 0:
                    continue

                # Base currency
                if ccy == pair[:3]:

                    if len(indexes[ccy]) > 0:
                        _add_data_to_existing(indexes[ccy], df)
                    else:
                        indexes[ccy] = df
                    continue
            
                # Counter currency
                elif ccy == pair[-3:]:

                    inverted = df * -1  
                    if len(indexes[ccy]) > 0:
                        _add_data_to_existing(indexes[ccy], inverted)
                    else:
                        indexes[ccy] = inverted

        # Now write indexes to the db
        for ccy in indexes:
            name = ccy + tf
            indexes[ccy].to_sql(name, VAL_CON, if_exists='replace', index=True)
            
    VAL_CON.close()
    

set_pair_value()
set_index_value()
