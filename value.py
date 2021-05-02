import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import sqlite3
import pathlib
from datetime import datetime
import time
from create_db import value_db, correlation_db, econ_db, ohlc_db
from correlation_scanner import _normalize
from symbols_lists import mt5_symbols
from ohlc_request import mt5_ohlc_request
from tokens import fin_token, mt5_login, mt5_pass
import mplfinance as mpf


VAL_CON = sqlite3.connect(value_db)
OHLC_CON = sqlite3.connect(ohlc_db)
ECON_CON = sqlite3.connect(econ_db)
# CORR_CON = sqlite3.connect(correlation_db)
CORR_CON = sqlite3.connect(r'C:\Users\ru\forex\db\correlation3tfs.db')


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

def _add_to_df(df, pair, tf, cor_df, cor_sum):
    
    df[pair] = cor_df[f'*{pair}*']
    df[f'{tf[1:]}'] = round(cor_sum, 4)

    df = _normalize(df, pair)
    df = _normalize(df, tf[1:])

    return df

def set_pair_value():
    ''' Read the correlation data for each symbol and derive an average.
    Essentially convert an ugly, chopped n screwed dataframe into a series that can be plotted '''

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
            df = _add_to_df(df, pair, tf, cor_df, cor_sum)

            
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
            except Exception:
                continue

            # Verify the mean correlation from this table is above n
            if df[pair].corr(df[tf[1:]]) < 0.8:
                continue

            df = df.set_index(df['datetime'], drop=True)
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

    indexes = [
    'USD',
    'EUR',
    'GBP',
    'JPY',
    'AUD',
    'NZD',
    'CAD',
    'CHF'
    ]

    indexes.extend(mt5_symbols['majors'])

    # This will save the symbol names and their value line's correlation score
    corr_values = pd.DataFrame()
    for symbol in mt5_symbols['majors']:  #### edited for crypto scan
        
        for tf in timeframes:
            
            name = symbol + tf
            
            try:
                df = pd.read_sql(f'SELECT * FROM {name}', VAL_CON)
                corr_value = df[symbol].corr(df[tf[1:]])
                corr_values.loc[symbol, tf] = corr_value
                
            except:
                print(f"Couldn't open {name}")
                continue
            
    # Create an average corr column based on whatever tf is notna(). row wise.
    non_nans = corr_values.notna().sum(axis=1)
    temp_df = corr_values.fillna(0)
    corr_values['cor_avg'] = temp_df.sum(axis=1) / non_nans
     


    # find some details about the corr values by currency
    corr_values = corr_values.sort_values(by=['cor_avg'], ascending=False)
    highscores = corr_values[corr_values > 0.8]
    print(highscores)
    # print('')
    # print(corr_values.index.tolist())
    
    # for ccy in indexes:
    #     for symbol in corr_values.index:
    #         avg = corr_values[corr_values.loc[symbol, :] > 0.8].mean()
    #         num = corr_values.loc[symbol][corr_values > 0.8].notna().sum(axis=1)
    p = r'C:\Users\ru'
    # corr_values.to_csv(pathlib.Path(p, 'corr-values.csv'))

    return corr_values

def _set_forecasts(symbol, ohlc):
    ''' Grab the outlook numbers from the forecasts database and plug those into
    the ohlc data which will be plotted '''

    symbol_1 = symbol[:3]
    symbol_2 = symbol[-3:]

    cal = pd.read_sql(f'''SELECT * FROM ff_cal
                          ''', ECON_CON)  
                        #   WHERE outlook.ccy = {symbol_1} OR outlook.ccy = {symbol_2}''', ECON_CON)

    cal = cal[(cal.ccy == symbol_1)
                |
                (cal.ccy == symbol_2)
    ]

    cal.index = pd.to_datetime(cal.datetime)
    cal.index = cal.index.floor('H')
    
    # cal = cal.reindex(ohlc.index).notna()
    cal = cal[cal.index.isin(ohlc.index)]
    cal = cal.dropna(subset=['outlook'])
    
    # Anything existing on the counter side of the pair will act as an opposite force
    cal.loc[cal.ccy == symbol_2] *= -1
    ohlc.loc[cal.index, 'forecasts'] = cal.outlook

    # Set the forecast as volume for plotting, and set to 0 in case theres no data
    ohlc.volume = np.nan
    ohlc.volume = ohlc.forecasts
    ohlc.volume = 0 if len(ohlc.volume.notna()) == 0 else ohlc.volume
    return ohlc

def _resample_with_value_line(timeframe:str, tf_values:dict, ohlc:pd.DataFrame) -> pd.DataFrame:

    open = ohlc.open.resample(timeframe).first()
    high = ohlc.high.resample(timeframe).max()
    low = ohlc.low.resample(timeframe).min()
    close = ohlc.close.resample(timeframe).last()
    volume = ohlc.volume.resample(timeframe).sum()
    forecasts = ohlc.forecasts.resample(timeframe).sum()
    
    ohlc = pd.DataFrame({'open': open,
                        'high': high,
                        'low': low,
                        'close': close,
                        'volume': volume,
                        'forecasts': forecasts,
                        })

    # Add in the value lines to the ohlc df
    for tf in tf_values:
        df = tf_values[tf]
        df = df.reindex(ohlc.index)
        ohlc.loc[df.index, tf] = df.loc[df.index, tf].values # add tf col
        ohlc = _normalize(ohlc, tf)

    # Also add an average value line in case it's requested
    avg = _make_average_value_line(tf_values)
    avg = avg.reindex(ohlc.index)
    ohlc.loc[avg.index, 'avg'] = avg.values 
    ohlc = _normalize(ohlc, 'avg')
    return ohlc

def _make_average_value_line(tf_values: dict) -> pd.Series: 
    ''' Make an average line between the tfs '''
    avg = pd.DataFrame()
    for tf in tf_values:
        avg[tf] = tf_values[tf][tf]  # accessing column with same name within dataframe

    non_nans = avg.notna().sum(axis=1)
    temp_df = avg.fillna(0)
    avg['cor_avg'] = temp_df.sum(axis=1) / non_nans
    return avg['cor_avg']
        
def _make_plot_overlays(ohlc, single_value_line):  

    value_lines = []
    if single_value_line:
        plot = mpf.make_addplot(ohlc.avg, color='b', panel=0, secondary_y=True, )
        value_lines.append(plot)
    # else:
    #     for tf, c in zip(tf_values, colors):
    #         x = mpf.make_addplot(ohlc[tf], color=c, panel=0, secondary_y=True, title=tf,)
    #         value_lines.append(x)
    
    return value_lines

def plot_value(symbols:list, timeframe:str, num_candles:int, num_charts:int, using_currency_index_data:bool=False, single_value_line=True):
    ''' Get ohlc data from MT5, overlay the value data and plot '''
    
    # This will be used as a tag in the saved images
    the_time = str(datetime.now()).split('.')[0].replace(' ','_')
    the_time = the_time.split(':')[0]

    colors = [
        # 'w',
        'b',
        'g']

    tfs = [
        # 'LTF',
        'MTF',
        'HTF',
    ]

    tf_values = {}          # type: dict[pd.Dataframe]
    for symbol in symbols[:num_charts]:
    # for symbol, score in zip(symbols, scores):
        for tf in tfs:
            # If the table exists open it
            table_name = symbol + '_' + tf
            try:
                value = pd.read_sql(f'SELECT * FROM {table_name}', VAL_CON)
                # structure is index, datetime, symbol, tf
                value.index = pd.to_datetime(value.datetime)
                # Resample
                val = value[tf].resample(timeframe).last()
                val_resampled = pd.DataFrame({
                    tf: val,
                })
                tf_values[tf] = val_resampled
            except:
                continue
        
        if not tf_values: 
            continue

        # This block was used for currency index ohlc data
        if using_currency_index_data:
            try:
                ohlc = pd.read_sql(f'SELECT * FROM {symbol}', OHLC_CON)
                ohlc = ohlc.set_index(ohlc['datetime'], drop=True)
                ohlc.index = pd.to_datetime(ohlc.index)
                ohlc = ohlc.drop(columns='datetime')
            except:
                continue

        if not mt5.initialize(login=mt5_login, server="ICMarkets-Demo",password=mt5_pass):
            print("initialize() failed, error code =", mt5.last_error())
            quit()

        ohlc = mt5_ohlc_request(symbol, mt5.TIMEFRAME_H1, num_candles=num_candles)
        
        # If current symbol is a currency, request the calendar forecast data for plotting
        if symbol in mt5_symbols['majors']:
            ohlc = _set_forecasts(symbol, ohlc)
            ohlc = _resample_with_value_line(timeframe, tf_values, ohlc)
            ohlc = ohlc.dropna(subset=['open', 'high', 'low'])
            
        value_lines = _make_plot_overlays(ohlc, single_value_line)

        # control plot windowing (zoom in on historical data)
        # ohlc['idx'] = range(0, len(ohlc))
        # ohlc = ohlc[(ohlc.idx > len(ohlc) * .00)
        #             &
        #             (ohlc.idx < len(ohlc) * .999)
        #             ]
        p = rf'Desktop'
        p = pathlib.Path(p, f'{the_time}_score_{symbol}.png')
        mpf.plot(ohlc,type='candle',tight_layout=True, 
                show_nontrading=False, volume=True, title=f'{symbol}', 
                addplot=value_lines, savefig=f'{the_time}_{symbol}_{timeframe}.png')


set_pair_value()
# set_index_value()
# value_corr_scanner()
# corr_scores = pd.read_csv('corr-values.csv')
# symbols = corr_scores.iloc[:, 0].values

symbols = mt5_symbols['majors']
plot_value(symbols, '480 min', 1000, 15)
CORR_CON.close()
VAL_CON.close()