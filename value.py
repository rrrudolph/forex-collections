import pandas as pd
import sqlite3
import pathlib
from datetime import datetime
import time
from create_db import value_db, correlation_db
from correlations import _normalize
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
                print(f"Couldn't open {name}")
                continue
            
            if len(cor_df) < 10:
                print(f'cor_df didnt have any data for {name}')
                continue

            cor_df.index = pd.to_datetime(cor_df['index'])

            # Parse the column names because I need to multiply each symbols column by its corr column
            # I left myself some keys so I could easily grab what I need
            things_to_omit = [r'*', 'corr', 'index', 'shift']
            cols = cor_df.columns.tolist()

            parsed_cols = []
            for col in cols:
                if all(x not in col for x in things_to_omit):
                    parsed_cols.append(col)
            
            # Multiply the symbol values by the current correlation score
            temp_df = pd.DataFrame()
            for col in parsed_cols:
                temp_df[col] = cor_df[col] * cor_df[f'{col}_corr']
            
            # Now I have df of just the symbol values but tons of nans. Before I deal with those
            # I want to save the number of non nan symbols in each row so I can get an average value
            non_nans = temp_df.notna().sum(axis=1)

            temp_df = temp_df.fillna(0)
            cor_sum = temp_df.sum(axis=1)

            # Average by the count of non nans
            cor_sum /= non_nans

            # Save the normalized values of the pairs close prices and the derived value
            df[pair] = cor_df[f'*{pair}*']
            df[f'{tf[1:]}'] = round(cor_sum, 4)

            df = _normalize(df, pair)
            df = _normalize(df, tf[1:])
            
            df.to_sql(name, VAL_CON, if_exists='replace', index=True)


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
            
    
def value_corr_scanner():
    ''' Iter over each symbol and save the correlation of its value line against price.
    Then I can sort by the best ratings to find which charts are probably nicest to trade '''

    # This will save the symbol names and their value line's correlation score
    values = pd.Series(dtype=object)

    for tf in timeframes:
        for pair in mt5_symbols['majors']:
            
            name = pair + tf
            
            try:
                df = pd.read_sql(f'SELECT * FROM {name}', VAL_CON)
            except:
                print(f"Couldn't open {name}")
                continue

            # Get correlation value
            corr_value = df[pair].corr(df[tf[1:]])

            # Save the data
            values[name] = corr_value
    
    values = values.sort_values(ascending=False)

    p = r'C:\Users\ru'
    values.to_csv(pathlib.Path(p, 'corr-scores.csv'))




# set_pair_value()
# set_index_value()
value_corr_scanner()
CORR_CON.close()
VAL_CON.close()