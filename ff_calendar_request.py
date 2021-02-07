# import gspread
# from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import sqlite3
import time
from datetime import datetime
from create_db import setup_conn, econ_db  # local path to the database 
from tokens import ff_cal_sheet

conn, c = setup_conn(econ_db)


forecast_weights = {
    'Unemployment': 3,
    'Employment': 3,
    'CPI': 3,
    'Consumer': 3,
    'Housing': 3,
    'House': 3,
    'Non-Farm': 3,
    'Retail': 3,
    'Confidence': 3,
    'Flash': 3,
    'NHPI': 3,
    'HPI': 3,
    'PMI': 3,
    'Prelim': 2,
    'Final': 0.5
}



def _update_gsheet_cell(*args, historical_fill=False, sheet=ff_cal_sheet):
    ''' Handles content insertion into spreadsheet. 
    If doing a historical fill, pass month as first arg and year as second. '''
    
    month = args[0]
    year = args[1]

    if historical_fill == True:
        cell = fr"""=importhtml("https://www.forexfactory.com/calendar?month={month}.{year}", "table", 4)"""
    else:
        cell = fr"""=importhtml("https://www.forexfactory.com/calendar?week=this", "table", 4)"""
        
    sheet.update_cell(1,1, cell)

    time.sleep(1)

    data = sheet.get_all_values() # list of lists
    df = pd.DataFrame(data)

    return df


def build_historical_db(year_start=2012, year_end=2022):
    
    months = [
        'jan', 'feb', 'mar', 'apr', 'may', 'jun', 
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ]
    years = range(year_start, year_end)

    for year in years:
        for month in months:
            
            # Request data (there's a slight chance the program will go faster than gsheets will load)
            df = _update_gsheet_cell(month, year, historical_fill=True)
            time.sleep(1)

            # Clean and format a little
            df = clean_data(df, year, remove_non_numeric=False)

            # There won't be any blank 'actual' cells at this point, 
            # so if one is found, stop saving data cuz the current month is reached
            if len(df[df.actual.isnull()]) > 0:
                break

            # Update database
            save_ff_cal_to_db(df)

    #  :/
    conn.close()


def weekly_ff_cal_request():

    # Make request
    df = _update_gsheet_cell()

    year = datetime.today().year
    df = clean_data(df, year)

    return df


def run_regex(df):
    ''' Remove the non-numeric values '''

    reg = r'([<>KMBT%])'
    df.actual = df.actual.replace(reg,'',regex=True)
    df.forecast = df.forecast.replace(reg,'',regex=True)
    df.previous = df.previous.replace(reg,'',regex=True)

    return df


def clean_data(df, year, remove_non_numeric=True):

    # Clean data cuz it's really quite bad

    df = df.rename(columns = {
            df.columns[0]: 'date', 
            df.columns[1]: 'time', 
            df.columns[2]: 'ccy', 
            df.columns[3]: 'impact', 
            df.columns[4]: 'event', 
            df.columns[5]: 'detail', 
            df.columns[6]: 'actual', 
            df.columns[7]: 'forecast', 
            df.columns[8]: 'previous', 
            df.columns[9]: 'graph', 
            })
    df = df.drop(axis=1, columns = ['graph', 'impact', 'detail'])
            
    # Convert blanks to nans and then fill               
    df.date[df.date == ''] = np.nan
    df.date = df.date.fillna(method='ffill')         
    df.time[df.time == ''] = np.nan
    df.time = df.time.fillna(method='ffill')

    # Remove some unwanted rows
    df = df.loc[1:, :]
    df = df[df.time.str.contains(':')]
    df = df[df.previous != '']
    error_causers = 'Spanish|Italian|French|Bond|Data|Vote|MPC'
    df = df[~df.event.str.contains(error_causers)]

    # Fix the date and add the current year
    df['date'] = df.apply(lambda x: x['date'][3:] + ' ' + str(year), axis=1) 

    # Organize some columns and change time from EST to CST
    df['datetime'] = pd.to_datetime(df.date + ' ' + df.time)
    df.datetime = df.datetime - pd.Timedelta('1 hour')
    df = df.drop(columns = ['date', 'time'])
    df['ccy_event'] = df.ccy + ' ' + df.event


    # Remove all the non-numeric values
    if remove_non_numeric == True:
        df = run_regex(df)

    # Re-order the columns
    df = df[['datetime', 'ccy', 'ccy_event', 'actual', 'forecast', 'previous']]

    return df


def rate_upcoming_forecasts(weights=forecast_weights):
    ''' Calculate the current weeks data by normalizing
    against the database. '''

    # While my database is still small it makes more sense to
    # read the whole thing into memory rather than make queries, so...
    db = pd.read_sql('ff_cal', conn)

    df = weekly_ff_cal_request()
    
    # Filter for events with forecasts which are set to drop within the next 48 hours
    # datetime.now() is actually 6 hours ahead 
    time_horizon = datetime.now() + pd.Timedelta('42 hours') 
    df = df[
        (df.forecast.notna()) 
        & 
        (df.datetime < time_horizon)
    ]
    
    # Combine the df's to normalize and stuff
    combined = pd.concat([db, df])

    # Now go through the upcoming events and calculate stuff
    forecasts = {}
    for i in df.index:
        event = df.loc[i, 'ccy_event']

        temp = combined[combined.ccy_event == event]

        # Normalize forecast and previous 
        temp.forecast = (temp.forecast - min(temp.forecast)) / (max(temp.forecast) - min(temp.forecast))
        temp.previous = (temp.previous - min(temp.previous)) / (max(temp.previous) - min(temp.previous))

        # Get the initial forecast change value
        forecast_rating = temp.loc[-1, 'forecast'] - temp.loc[-1, 'previous']

        # For the next group of calculations the last row will always be nan,
        # so grab the second to last value

        # Check the trend of the n previous releases (actuals)
        trend = db.loc[-2, 'trend'][db.ccy_event == event]

        both_positive = forecast_rating > 0 and trend > 0
        both_negative = forecast_rating < 0 and trend < 0
        if both_positive or both_negative:
            forecast_rating *= 1.25

        # Check the currency's overall data trend
        ccy_trend = db.loc[-2, 'ccy_trend'][db.ccy_event == event]

        both_positive = forecast_rating > 0 and ccy_trend > 0
        both_negative = forecast_rating < 0 and ccy_trend < 0
        if both_positive or both_negative:
            forecast_rating *= 1.25

        # Multiply by the forecasts average accuracy (%)
        accuracy = db.loc[-2, 'accuracy'][db.ccy_event == event]
        forecast_rating *= accuracy

        # Finally, add that currency and its data to the dict to send off to the global ratings controller
        ccy = df.loc[i, 'ccy']
        if forecasts[ccy]:
           forecasts[ccy] += forecast_rating
        else:
           forecasts[ccy] = forecast_rating

    # dict
    return forecasts


def calculate_raw_db():
    ''' Add accuracy, trend, and weight columns to the raw db,
    then save as a new formatted file. '''

    # Open file and prepare for calculations
    df = pd.read_sql('ff_cal_raw', conn)
    df = run_regex(df)

    # Find unique events (forecasts)
    df['ccy_event'] = df.ccy + ' ' + df.event
    unique_events = df.ccy_events.unique()


    # Run some calculations on each unique event
    for unique_event in unique_events:
        temp = df[df.ccy_event == unique_event]

        # Apply weights (certain forecast events are more important than others)
        # randnote: if an event is 'Final...PMI', it will assign a weight of 0.5
        weight = None
        for event in forecast_weights:     
            
            if event in unique_event:
                weight = forecast_weights[event]
           
           # If nothing is found set to 1
            if weight is None:
                weight = 1
        
        df.loc[temp.index, 'weight'] = weight

        # Get the recent accuracy of the forecast as a percentage
        accuracy = (100 - temp.forecast / abs(temp.forecast - temp.actual)) / 100
        df.loc[temp.index, 'abs_accuracy'] = accuracy.rolling(6).mean()

        # Get the recent trend of the actuals (per unique forecast)
        trend = temp.actual.diff().rolling(4).mean()
        df.loc[temp.index, 'trend'] = trend

        # Get the recent trend of the actuals (overall by currency)
        # Make a new temp df
        temp = df[df.ccy == temp.ccy[1]]
        ccy_trend = temp.trend.rolling(7).mean()
        df.loc[temp.index, 'ccy_trend'] = ccy_trend

    # Overwrite current file
    df.to_sql('ff_cal', conn, if_exists='replace')


def save_ff_cal_to_db(df):
    ''' Write the data to the database. '''

    df.to_sql('ff_cal_raw', conn, if_exists='append', index=False)


build_historical_db()