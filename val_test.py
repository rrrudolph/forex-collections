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
'''

timeframes = [
    '_LTF',
    '_MTF',
    '_HTF',
]

def _add_data(existing, new, col):
    ''' When a column of a different length gets added to what already exists,
    nans will occur on any new index locations.  Those nans need to be overwritten by
    the new values. '''

    # existing += new_data
    # if len(existing[existing.isnull()]) > 0:
    #     existing[existing.isnull()] = new_data[existing.isnull()]

    # i donno why that gives me unalignable index error
    for i, value in new.items():

        if i in existing.index:
            existing.loc[i, col] += value
        else:
            existing.loc[i, col] = value

    return existing


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

    CORR_CON.close()
    VAL_CON.close()

def set_index_value():
    ''' Using the data that was calculated in the pair function, combine
    now into index value scores. Some of the same steps from making the OHLC 
    indexes will exist here. '''

    # This will store the combined data of each timeframe
    template = pd.DataFrame(columns=['LTF', 'MTF', 'HTF'])
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

        # The table might not exist
        try:
            df = pd.read_sql(f'SELECT * FROM {pair}', VAL_CON)
        except:
            continue

        df = df.set_index(df['index'], drop=True)
        df.index = pd.to_datetime(df.index)
        df = df.drop(columns='index')
        
        # Go one by one thru the columns and put that data in the ccy index df
        for col in df.columns.tolist():
            # drop any nans
            temp = df[col].dropna()

            for ccy in indexes:
                
                # Base currency
                if ccy == pair[:3]:
                    indexes[ccy] = _add_data(indexes[ccy], temp, col)

                # Counter currency
                elif ccy == pair[3:]:
                    inverted = temp * -1
                    indexes[ccy] = _add_data(indexes[ccy], inverted, col)

    return indexes

def save_indexes_to_db(indexes):
    ''' Each index in the indexes dict is a dict containing 1 to 3 keys.
    I need to combine that data into a single df and save to a database table. '''
    
    # "index" is the currency name
    for index in indexes:
        df = pd.DataFrame()


        # "key" will be the timeframe name and 'value' will be the series
        for key, value in indexes[index].items():
            df[key] = value

        df.to_sql(index, VAL_CON, if_exists='replace', index=True)

        # "index" is a dict object containing 1 to 3 keys
        # print('\nnumber of nulls')
        # print(len(indexes[index][indexes[index].isnull()]))
    # print(indexes)

    VAL_CON.close()

indexes = set_index_value()
save_indexes_to_db(indexes)
