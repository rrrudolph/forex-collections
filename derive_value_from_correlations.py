import pandas as pd
import numpy as np
import sqlite3
import pathlib
import mplfinance as mpf
import time
import tqdm
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
    # 'LTF',
    'MTF',
    'HTF',
]


def _add_pairs_corr_values_to_aggregate_df(df, pair, period, cor_df, cor_sum) -> pd.DataFrame:
    ''' The final utility function used at the end of calculating value
    from the correlations data. '''

    df[pair] = cor_df[f'*{pair}*']
    df[f'{period}'] = round(cor_sum, 4)

    df = _normalize(df, pair)
    df = _normalize(df, period)

    return df

def calculate_value_line_from_correlations() -> None:
    ''' Read the correlation data for each symbol and derive an average.
    Essentially convert an ugly, chopped n screwed dataframe into a series that can be plotted '''

    # keep the same format as the correlation scanner by itering periods and then pairs
    for period in periods:
        for pair in mt5_symbols['majors']:
            name = pair + '_' + period
            df = pd.DataFrame() # final df that is saved
            try:
                cor_df = pd.read_sql(f'SELECT * FROM {name}', CORR_CON)
            except:
                print(f"Couldn't open {name}")
                continue
            
            if len(cor_df) < 10:
                print(f"There's only {len(cor_df)} rows of corr data for {name}. Skipping")
                continue
            
            cor_df.index = pd.to_datetime(cor_df.datetime)
            cor_df = cor_df.dropna(subset=[f'*{pair}*'])

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
            df = _add_pairs_corr_values_to_aggregate_df(df, pair, period, cor_df, cor_sum)
            df.to_sql(name, VAL_CON, if_exists='replace', index=True)


def value_line_rating_scanner() -> pd.DataFrame:
    ''' Check the correlation of price to a symbol's value line. That way
    good value lines can be identified and focused on for trading signals.'''

    # This will save the symbol names and their value line's correlation score
    corr_values = pd.DataFrame()
    for symbol in mt5_symbols['majors']:  
        for period in periods:
            name = symbol + '_' + period
            try:
                df = pd.read_sql(f'SELECT * FROM {name}', VAL_CON)
                print(df.head())
                corr_value = df[symbol].corr(df[period])
                corr_values.loc[symbol, period] = corr_value
            except:
                print(f"Couldn't open {name} from the value DB.")
                continue
            
    # Create an average corr column based on whatever period is notna(). row wise.
    non_nans = corr_values.notna().sum(axis=1)
    temp_df = corr_values.fillna(0)
    corr_values['avg_value'] = temp_df.sum(axis=1) / non_nans

    # Iter thru the correlation scores and save to a dict
    corr_values = corr_values.sort_values(by=['HTF'], ascending=False)
    # actually screw the averages, since HTF is always highest just roll with that. 
    # it looks too high (0.95) for AUD&GBPJPY but whatever https://prnt.sc/13aj2f7
    corr_dict = {}
    for i in corr_values.index:
        corr_dict[f'{i}_HTF'] = corr_values.loc[i, 'HTF']

    return corr_dict


def _make_average_value_line(value_data) -> pd.Series:
    ''' Make an average line of the various periods value. This 
    is called by two separate functions, one passes a dict and 
    the other a series. I need the same result in both cases 
    but need to handle the data types passed accordingly. '''

    avg = pd.DataFrame() # used by both

    if isinstance(value_data, dict):
        for table_name, value in value_data.items():
            avg[table_name] = value[table_name[-3:]]
        # avg has columns "EURUSD_MTF", "EURUSD_HTF" with datetime index
        non_nans = avg.notna().sum(axis=1)
        temp_df = avg.fillna(0)

    if isinstance(value_data, pd.DataFrame):
        non_nans = value_data.notna().sum(axis=1)
        temp_df = value_data.fillna(0)
    
    try:
        avg['value'] = temp_df.sum(axis=1) / non_nans
    except Exception as e:
        print(e)
        print('Tried averaging value lines but probably had nans in all columns.')
        quit()

    return avg['value']

def _make_value_line(symbol: str, average_line: bool=True) -> pd.Series:
    ''' Read the value DB for all available periods of the given symbol and return them as a df.
    If return_average_line is True, create an average of the two lines and return a Series. '''

    internal_symbol = symbol
    value_data = {}   # type: dict[symbol: pd.DataFrame]
    for period in periods:
        # If a symbol is passed with its period in the name just use that
        if ('_LTF' or '_MTF' or '_HTF') in symbol:
            print(111111111111)
            table_name = symbol
            # period = symbol[-3:]
            internal_symbol = symbol[:6]
            average_line = False
        else:
            table_name = symbol + '_' + period
            print(222222222222222222)

        try:
            df = pd.read_sql(f'SELECT * FROM {table_name}', VAL_CON)
            # structure is index, datetime, symbols normalized close (name=symbol), value lines normalized close (name=period)
            df.index = pd.to_datetime(df.datetime)
            df = df.drop(columns=['datetime'])
            df = df.dropna()

        except Exception as e:
            print('\n', table_name, 'got an error inside _make_value_line():')
            print(e)
            continue

        # Check for minimum correlation score
        value_corr = df[internal_symbol].corr(df[period])
        if value_corr < 0.7:
            print(f"{table_name}'s value line had a correlation of {round(value_corr, 2)}. Skipping.")
            continue

        value_data[table_name] = df

    if average_line:
        return _make_average_value_line(value_data)
    
    return df[period] # the value column only


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

def _add_index_data_to_existing_df(existing: pd.Series, new_data: pd.Series) -> pd.Series:
    ''' Two Series need to be combined into a df and their values averaged.
    Nans are just overwritten by the single value that exists. If both are nans
    then they are ffilled.'''

    temp = existing.to_frame(name='existing').join(new_data, how='outer')
    temp = _make_average_value_line(temp)

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
        # Add each period's data to its currency dict, inverting if necessary
        for index in indexes:
            if value_line.empty:
                print("no value found for", symbol, '(_set_index_value)')
                continue
            # Base currency
            if index == symbol[:3]:
                if indexes[index].empty:
                    indexes[index] = value_line
                else:
                    _add_index_data_to_existing_df(indexes[index], value_line)
            # Counter currency
            elif index == symbol[-3:]:
                inverted = value_line * -1  
                if indexes[index].empty:
                    indexes[index] = inverted
                else:
                    _add_index_data_to_existing_df(indexes[index], inverted)
        
    return indexes

def _make_mpf_addplot_overlay(ohlc_df: pd.DataFrame, value_line: pd.Series):
    ''' Returns an mpf.make_addplot object '''
    
    # Add the value line series to the ohlc_df to ensure matching lengths
    ohlc_df = ohlc_df.join(value_line, how='outer')

    # Get name of last column (varies based on the caller)
    value_col = ohlc_df.columns[-1]
    ohlc_df[value_col] = ohlc_df[value_col].fillna(method='ffill').fillna(method='bfill')
    ohlc_df = ohlc_df.dropna(subset=['open'])
    return mpf.make_addplot(ohlc_df[value_col], color='g', panel=0, secondary_y=True,)
    
def plot_charts(symbols:list, timeframe:str, num_candles:int=200, include_forecasts:bool=True):
    ''' Get ohlc data from MT5 or indexes, overlay the value data and plot '''

    if symbols[0] in indexes:
        value_lines = _set_index_value()

    for symbol in symbols:
        if symbol in indexes:
            # ohlc = pd.read_parquet(INDEXES_PATH,f'{index}_M5.parquet')
            ohlc = pd.read_parquet(pathlib.Path(INDEXES_PATH + f'\{symbol}_M5.parquet'))
            ohlc.index = pd.to_datetime(ohlc.index)
            ohlc = _resample(ohlc, timeframe)
            if include_forecasts:
                ohlc = _set_forecasts_as_volume(ohlc, symbol, timeframe)
            overlay = _make_mpf_addplot_overlay(ohlc, value_lines[symbol])
            try:
                _save_chart_pic(ohlc, symbol, timeframe, overlay)
            except Exception as e:
                print(e)
                print('\n misaligned overlay with df probably. saving pics without value line.')
                _save_chart_pic(ohlc, symbol, timeframe, [])
        
        # Handle single symbols passed with or without their period tag 
        if symbol[:6] or symbol in mt5_symbols['majors']:
            if '_LTF' or '_MTF' or '_HTF' in symbol:
                ohlc = mt5_ohlc_request(symbol[:6], mt5_timeframes[timeframe], num_candles=num_candles)
            else:
                ohlc = mt5_ohlc_request(symbol, mt5_timeframes[timeframe], num_candles=num_candles)
            value_line = _make_value_line(symbol)   
            ohlc = _resample(ohlc, timeframe)
            overlay = _make_mpf_addplot_overlay(ohlc, value_line)
            _save_chart_pic(ohlc, symbol, timeframe, overlay)


if __name__ == '__main__':

    # Call this function to update the value database first
    #calculate_value_line_from_correlations()
    plot_charts(indexes, 'H8', include_forecasts=True, num_candles=400)

    # Call this function to get the n pairs with the best value lines
    # value_line_scores = value_line_rating_scanner()
    # top_pairs = [k for (k,v) in value_line_scores.items() if v > 0.8]
    # plot_charts(top_pairs, 'H4', include_forecasts=True, num_candles=100)
    # print(top_n_pairs)

# I need value_line_rating_scanner to return a dict with pair_peroid
# as the key and the correlation value as the value. Then I can pass
# the key in as the symbol name to plot charts and it will just give 
# me that single line 