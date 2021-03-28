import MetaTrader5 as mt5
from matplotlib.pyplot import savefig
import pandas as pd
import numpy as np
from datetime import date, datetime
import time
import sqlite3
from symbols_lists import mt5_symbols
from ohlc_request import mt5_ohlc_request
from create_db import econ_db , value_db, ohlc_db
# from tokens import ff_cal_sheet, forecast_sheet, bot, indexes_db
# from ff_calendar_request import _set_dtypes
# from indexes import make_ccy_indexes
import mplfinance as mpf
import pathlib
from tokens import fin_token, mt5_login, mt5_pass

ECON_CON = sqlite3.connect(econ_db)
OHLC_CON = sqlite3.connect(ohlc_db)
VAL_CON = sqlite3.connect(value_db)


def _set_forecasts(symbol, ohlc):
    ''' Grab the outlook numbers from the forecasts database and plug those into
    the ohlc data which will be plotted '''


    symbol_1 = symbol[:3]
    symbol_2 = symbol[-3:]

    cal = pd.read_sql(f'''SELECT * FROM ff_cal
                          ''', ECON_CON)   # WTF ?
                        #   WHERE outlook.ccy == {symbol_1} OR outlook.ccy == {symbol_2}''', ECON_CON)

    cal = cal[(cal.ccy == symbol_1)
                |
                (cal.ccy == symbol_2)
    ]

    cal.index = pd.to_datetime(cal.datetime)
    cal.index = cal.index.floor('H')
    
    # cal = cal.reindex(ohlc.index).notna()
    cal = cal[cal.index.isin(ohlc.index)]
    cal = cal.dropna(subset=['outlook'])


    if len(cal) == 0:
        return ohlc
    
    # Anythin existing on the counter side of the pair will act as an opposite force
    cal.loc[cal.ccy == symbol_2] *= -1

    ohlc.loc[cal.index, 'forecasts'] = cal.outlook

    # try to set as vol again
    ohlc.volume = np.nan
    ohlc.volume = ohlc.forecasts

    return ohlc


def plot_value(timeframe, num_candles, num_charts):
    ''' get ohlc data, resample the value data and plot '''
    
    # This will be used as a tag in the saved images
    the_time = str(datetime.now()).split('.')[0].replace(' ','_')
    the_time = the_time.split(':')[0]
    
    corr_scores = pd.read_csv('corr-values.csv')

    symbols = corr_scores.iloc[:, 0].values

    # scores = corr_scores.iloc[:, 1].values + 

    colors = [
        # 'w',
        'b',
        'g']

    tfs = [
        # 'LTF',
        'MTF',
        'HTF',
    ]

    tf_values = {}
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
        
        # If nothing found, skip this symbol
        if not tf_values:
            continue

        # Make an average line between the tfs
        avg = pd.DataFrame()
        for tf in tf_values:
            avg[tf] = tf_values[tf][tf]

        non_nans = avg.notna().sum(axis=1)
        temp_df = avg.fillna(0)
        avg['cor_avg'] = temp_df.sum(axis=1) / non_nans

        # This block was used for currency index ohlc data. maybe need to engage this with a kwarg in function call
        if False:
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
            #ohlc.to_sql(f'{ccy}', OHLC_CON, if_exists='replace', index=True)

        # Exchange the volume data for the forecast data
        ohlc = _set_forecasts(symbol, ohlc)

        # Resample 
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

            # Reset the value line index to match the ohlc index
            # print('before reindex:', len(df))
            df = df.reindex(ohlc.index)
            # print('after reindex:', len(df))

            # Add the tf column to the ohlc df
            ohlc.loc[df.index, tf] = df.loc[df.index, tf].values

        ohlc = ohlc.dropna(subset=['open', 'high', 'low'])

        # control plot windowing (zoom in on historical data)
        ohlc['idx'] = range(0, len(ohlc))
        ohlc = ohlc[(ohlc.idx > len(ohlc) * .05)
                    &
                    (ohlc.idx < len(ohlc) * .99)
                    ]


        # COMMENTED THIS OUT TO PLOT JUST THE SINGLE AVERAGE OF MTF AND HTF
        # Create the value line plot
        value_lines = []
        # for tf, c in zip(tf_values, colors):
        #     # Normalize value lines for window
        #     ohlc[tf] = (ohlc[tf] - min(ohlc[tf])) / (max(ohlc[tf]) - min(ohlc[tf]))
        #     tf_values[tf] = mpf.make_addplot(ohlc[tf], color=c, panel=0, secondary_y=True, title=tf,)
        #     value_lines.append(tf_values[tf])
    

        # avg = avg.reindex(ohlc.index)
        ohlc['avg'] = (avg.cor_avg - min(avg.cor_avg)) / (max(avg.cor_avg) - min(avg.cor_avg))
        plot = mpf.make_addplot(ohlc.avg, color='b', panel=0, secondary_y=True, title=tf,)
        value_lines.append(plot)

        # ohlc.volume = (ohlc.volume - min(ohlc.volume)) / (max(ohlc.volume) - min(ohlc.volume))
        # plot = mpf.make_addplot(ohlc.volume, color='black', panel=1, secondary_y=True, title=tf,)
        # value_lines.append(plot)

        #score = str(round(score, 2))
        # quit()
        # save plot
        p = rf'Desktop'
        p = pathlib.Path(p, f'{the_time}_score_{symbol}.png')
        mpf.plot(ohlc,type='candle',tight_layout=True, 
                show_nontrading=False, volume=True, title=f'{symbol}', 
                addplot=value_lines, savefig=f'{the_time}_{symbol}_{timeframe}.png')
                # addplot=value_lines)
                # addplot=plot)  # this ones for plotting just a single line

plot_value('240 min', 600, 3)

# need to push candles back to make room for up coming forecasts
