import pandas as pd
import numpy as np
import sqlite3
import pathlib
from datetime import datetime
import mplfinance as mpf
import time
from create_db import value_db, correlation_db, econ_db, ohlc_db
from corr_value_scanner import _normalize
from symbols_lists import mt5_symbols, mt5_timeframes, indexes
from ohlc_request import mt5_ohlc_request
from indexes import _resample, _reorder_timeframe_if_needed

VAL_CON = sqlite3.connect(value_db)
OHLC_CON = sqlite3.connect(ohlc_db)
ECON_CON = sqlite3.connect(r'C:\Users\ru\forex\db\economic_data.db')
# ECON_CON = sqlite3.connect('\\192.168.1.223\Documents\econ.db') wtf
CORR_CON = sqlite3.connect(correlation_db)
INDEXES_PATH = r'C:\Users\ru\forex\db\indexes'


''' 
The goal of this module is to read the raw correlation data and derive buy/sell values from it.
I'll do this by pair and also by index.  So once I have the data for each pair, I'll then
combine it similar to how I made the indexes, inverting when I need to and adding them all together.
There are up to 3 sets of data for each pair and I'll want to keep them separated throughout.
The data I have saved from the corr scanner is the normalized price values along with their corr rating.
I should be able to just multiply a symbol's normd values by its corr rating and then get the average of
all those values row by row.
'''

periods = [
    # '_LTF',
    '_MTF',
    '_HTF',
]


def _add_to_df(df, pair, period, cor_df, cor_sum):
    
    df[pair] = cor_df[f'*{pair}*']
    df[f'{period[1:]}'] = round(cor_sum, 4)

    df = _normalize(df, pair)
    df = _normalize(df, period[1:])

    return df

def calculate_value_line_from_correlations() -> None:
    ''' Read the correlation data for each symbol and derive an average.
    Essentially convert an ugly, chopped n screwed dataframe into a series that can be plotted '''

    # keep the same format as the correlation scanner by itering periods and then pairs
    for period in periods:
    
    # I don't know how to get a list of table names from the db so 
    # iter each pair and try to open a table from each timeframe
        for pair in mt5_symbols['majors']:
            
            name = pair + period
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
            
            # print(cor_df)
            cor_df.index = pd.to_datetime(cor_df.datetime)
            cor_df = cor_df.dropna(subset=[f'*{pair}*'])
            # print(cor_df)

            # Parse the column names because I need to multiply each symbols column by its corr column
            # I left myself some keys so I could easily grab what I need
            things_to_omit = [r'*', 'corr', 'shift', 'datetime']
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
            cor_sum = cor_sum[cor_sum.notna()]

            # Save the normalized values of the pairs close prices and the derived value
            df = _add_to_df(df, pair, period, cor_df, cor_sum)
            df.to_sql(name, VAL_CON, if_exists='replace', index=True)


def value_line_rating_scanner() -> pd.DataFrame:
    ''' Check the correlation of price to a symbol's value line. That way
    good value lines can be identified and focused on for trading signals. 
    Remember to call calculate_value... to get the latest data first.'''

    # This will save the symbol names and their value line's correlation score
    corr_values = pd.DataFrame()
    for symbol in mt5_symbols['majors']:  
        for period in periods:
            name = symbol + period
            try:
                df = pd.read_sql(f'SELECT * FROM {name}', VAL_CON)
                corr_value = df[symbol].corr(df[period[1:]])
                corr_values.loc[symbol, period] = corr_value
            except:
                print(f"Couldn't open {name} from the value DB.")
                continue
            
    # Create an average corr column based on whatever period is notna(). row wise.
    non_nans = corr_values.notna().sum(axis=1)
    temp_df = corr_values.fillna(0)
    corr_values['value'] = temp_df.sum(axis=1) / non_nans

    # find some details about the corr values by currency
    corr_values = corr_values.sort_values(by=['value'], ascending=False)
    highscores = corr_values[corr_values > 0.8]
    print('\n ~~~ Correlation values ~~~ \n', highscores)

    return corr_values


def _make_single_value_line(value_data: dict) -> pd.Series:
    ''' Make an average line of the various periods value '''
    avg = pd.DataFrame()
    for period in value_data:
        avg[period] = value_data[period][period[-3:]]

    non_nans = avg.notna().sum(axis=1)
    temp_df = avg.fillna(0)
    avg['value'] = temp_df.sum(axis=1) / non_nans
    return avg['value']

def _make_value_line(symbol:str, period:list) -> pd.Series:
    ''' Read the value DB for all available periods of the given symbol and return them as a df.
    If return_single_line is True, create an average of the two lines and return a Series. '''

    value_data = {}   # type: dict[symbol_name: pd.DataFrame]
    for period in periods:
        # If the table exists open it
        table_name = symbol + period
        try:
            df = pd.read_sql(f'SELECT * FROM {table_name}', VAL_CON)
            # structure is index, datetime, symbol normalized close, normalized value line
            df.index = pd.to_datetime(df.datetime)
            df = df.drop(columns=['datetime'])
            value_data[table_name] = df
        except Exception:
            continue

    # In the future of I want to return all the value lines I can just omit this last function call
    return _make_single_value_line(value_data)



def _set_forecasts_as_volume(ohlc_df, index, timeframe) -> pd.DataFrame:
    ''' Get the forecasts for a given currency and overwrite
    the volume column of an OHLC data set with them. '''

    calendar_df = pd.read_sql(f'''SELECT * FROM forecast''', ECON_CON) 
    calendar_df = calendar_df[calendar_df.ccy == index]
    calendar_df.index = pd.to_datetime(calendar_df.datetime)
    calendar_df.short_term = calendar_df.short_term.replace('', np.nan).astype(float)
    calendar_df.index = calendar_df.index.floor('H')

    tf = _reorder_timeframe_if_needed(timeframe)
    forecasts = calendar_df.short_term.resample(tf).sum().copy()
    forecasts = forecasts[forecasts.index.isin(ohlc_df.index)]
    forecasts = forecasts.dropna()

    # Set the forecast as volume for plotting, and set to 0 in case theres no data
    ohlc_df.volume = np.nan
    ohlc_df.volume = forecasts
    ohlc_df.volume = 0 if len(ohlc_df.volume.notna()) == 0 else ohlc_df.volume
   
    return ohlc_df

def _save_chart_pic(ohlc_df: pd.DataFrame, symbol: str, timeframe: str, overlay) -> None:
    ''' Save the mpf pic of the chart. '''

    # First drop any rows without data (I think resampling might create them or something)
    ohlc_df = ohlc_df.dropna(subset=['open'])

    # attemp to add some space at right edge
    num_rows = 10
    
    if overlay is not None: 
        mpf.plot(ohlc_df, type='candle', tight_layout=True, 
            show_nontrading=False, volume=True, title=f'{symbol}', 
            addplot=overlay, savefig=f'{symbol}_{timeframe}.png')
    else:
        mpf.plot(ohlc_df, type='candle', tight_layout=True, 
            show_nontrading=False, volume=True, title=f'{symbol}', 
            savefig=f'{symbol}_{timeframe}.png')


def _add_data_to_existing(existing, new_data):
    ''' When a period_data[period] of a different length gets added to what already exists,
    nans will occur on any new index locations.  Those nans need to be overwritten by
    the new values. However some of the period_data[period] has nans as well'''

    col = existing.columns[0]
    for row in new_data.itertuples(name=None, index=True):
        i = row[0]
        value = row[1]
        if i in existing.index:
            existing.loc[i, col] += value
        else:
            existing.loc[i,col] = value

    return existing

def _set_index_value() -> dict:
    ''' Using the data that was found in the calculate_value_line function, combine
    now into index value scores. Some of the same steps from making the OHLC 
    indexes will exist here. '''


    # This will store the combined value line from each symbol
    template = pd.DataFrame(columns=[period for period in periods])
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
    for symbol in mt5_symbols['majors']:
        
        value_line = _make_value_line(symbol, periods)

        # Add each period's data to its proper dict currency, inverting if necessary
        for index in indexes:
            if value_line.empty:
                continue
            # Base currency
            if index == symbol[:3]:
                if indexes[index].empty:
                    indexes[index] = value_line
                else:
                    _add_data_to_existing(indexes[index], value_line)
            # Counter currency
            elif index == symbol[-3:]:
                inverted = value_line * -1  
                if indexes[index].empty:
                    indexes[index] = inverted
                else:
                    _add_data_to_existing(indexes[index], inverted)
        
        return indexes

def _create_index_value_line(ohlc_df: pd.DataFrame, value_line: pd.Series):
    ''' Returns an mpf.addplot object '''
    
    # Add the value line series to the ohlc_df to ensure matching lengths
    ohlc_df = ohlc_df.join(value_line, how='outer')
    print(ohlc_df)
    ohlc_df.value = ohlc_df.value.fillna(method='ffill').fillna(method='bfill')
    ohlc_df = ohlc_df.dropna(subset=['open'])
    return mpf.make_addplot(ohlc_df.value, color='g', panel=0, secondary_y=True, width=1)
    
    # # Create temp df to ensure index alignment of value line and dataframe
    # temp = ohlc_df.copy()
    # temp['value'] = value_line
    # temp.value.fillna(method='ffill').fillna(method='bfill')
    # # temp['value'] = value_line[value_line.index.isin(ohlc_df.index)]
    # # temp.value = temp.value.fillna(method='ffill').fillna(method='bfill')
    # return mpf.make_addplot(temp.value, color='g', panel=0, secondary_y=True)

def plot_charts(symbols:list, timeframe:str, num_candles:int=200, num_charts:int=10, include_forecasts:bool=True):
    ''' Get ohlc data from MT5 or indexes, overlay the value data and plot '''

    value_lines = _set_index_value()
    for symbol in symbols:
        if symbol in indexes:
            # ohlc = pd.read_parquet(INDEXES_PATH,f'{index}_M5.parquet')
            ohlc = pd.read_parquet(pathlib.Path(INDEXES_PATH + f'\{symbol}_M5.parquet'))
            ohlc.index = pd.to_datetime(ohlc.index)
            ohlc = _resample(ohlc, timeframe)
            if include_forecasts:
                ohlc = _set_forecasts_as_volume(ohlc, symbol, timeframe)
            overlay = _create_index_value_line(ohlc, value_lines[symbol])
            try:
                _save_chart_pic(ohlc, symbol, timeframe, overlay)
            except Exception as e:
                print(e)
                print('\n misaligned overlay with df probably. saving pics without value line.')
            _save_chart_pic(ohlc, symbol, timeframe, [])
        
        else:
            ohlc_df = mt5_ohlc_request(symbol, mt5_timeframes[timeframe], num_candles=num_candles)
            value_line = _make_value_line(symbol, periods)   
            ohlc_df = _resample(ohlc_df, timeframe)
            
            # Make plot overlay
            data = value_line[value_line.index.isin(ohlc_df.index)]
            overlay = mpf.make_addplot(data, color='g', panel=0, secondary_y=True)
            _save_chart_pic(ohlc_df, symbol, timeframe, overlay)


plot_charts(indexes, 'H8', include_forecasts=True, num_candles=400)