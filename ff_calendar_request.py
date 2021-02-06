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

''' 
    This module will make a request for the current week's data every hour.
    It will then query the db in order to perform its normalizations etc, 
    but will only update the db with the weekly data on the weekend
    once all the 'actuals' are set. 
    
    Functions:
        0. Build historical db
        1. Request the current weeks data
        2. Format that data
        3. Query the db
        4. Combine current week with db and calculate
'''

event_weights = {
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


def build_historical_db(year_start=2012, year_end=2021):
    
    months = [
        'jan', 'feb', 'mar', 'apr', 'may', 'jun', 
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ]
    years = range(year_start, year_end)

    with conn:
        for year in years:
            for month in months:

                # Request data
                df = _update_gsheet_cell(month, year, historical_fill=True)

                # Clean and format a little
                df = clean_data(df, year, remove_non_numeric=False)

                # Update database
                save_ff_cal_to_db(df)


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

    # Remove all the non-numeric values
    if remove_non_numeric == True:
        df = run_regex(df)

    # Re-order the columns
    df = df[['datetime', 'ccy', 'event', 'actual', 'forecast', 'previous']]

    return df


def evaluate_forecast_rating(weights=event_weights):
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
    forecasts = pd.DataFrame()
    for i in df.index:
        ccy = df.loc[i, 'ccy']
        event = df.loc[i, 'event']

        temp = combined[(combined.ccy == ccy) & (combined.event == event)]

        # Normalize forecast and previous 
        temp.forecast = (temp.forecast - min(temp.forecast)) / (max(temp.forecast) - min(temp.forecast))
        temp.previous = (temp.previous - min(temp.previous)) / (max(temp.previous) - min(temp.previous))

        # Get the initial forecast change value
        forecast_rating = temp.loc[-1, 'forecast'] = temp.loc[-1, 'previous']

        # Apply weighting
        # Note: if an event is Final...PMI, it will assign a weight of 0.5
        weight = None
        for item in weights:     
            
            if item in event:
                weight = weights[item]
           
           # If nothing is found set to 1
            if weight is None:
                weight = 1
        
        forecast_rating *= weight

        # Check the accuracy of the forecast  -------------------------- to do
        # this should be a rollng mean or sometin, maybe like period 10. and make sure its abs()!!
        accuracy = db.loc[-1, 'accuracy'][(db.ccy == ccy) & (db.event == event)]
        forecast_rating *= accuracy

        # Check the trend of the n previous releases
        trend = db.loc[-1, 'trend'][(db.ccy == ccy) & (db.event == event)]
        forecast_rating *= trend

def calculate_raw_db():
    ''' Add accuracy, trend, and weight columns to the raw db,
    then save as a new formatted file. '''

    # Open file and prepare for calculations
    df = pd.read_sql('ff_cal_raw', conn)
    df = run_regex(df)

    # Find unique events
    df['ccy_event'] = df.ccy + df.event
    unique_events = df.ccy_events.unique()



    def _get_recent_accuracy(df, unique_events):
        ''' The accuracy of the forecast will affect it's tradability. '''

        # Make it a % 
        accuracy = (100 - df.forecast / abs(df.forecast - df.actual)) / 100
        df['abs_accuracy'] = accuracy.rolling(6).mean


    def _get_recent_trend(df, unique_events):

        



    ratings = {}
    for event in upcoming_events:

        # Query the database for all equivalent events
        db_df = read_database(event)

        # Database values are saved as str so convert to proper dtype
        db_df.datetime = pd.to_datetime(db_df.datetime)
        db_df.actual = db_df.actual.astype(float)
        db_df.forecast = db_df.forecast.astype(float)
        db_df.previous = db_df.previous.astype(float)

        # Combine the current event with equivalent db events and sort
        df = pd.concat(weekly_df[weekly_df.event == event], db_df)
        df = df.sort_values(by=['datetime'])

        # Normalize the data
        df.actual = (df.actual - min(df.actual)) / (max(df.actual) - min(df.actual))
        df.forecast = (df.forecast - min(df.forecast)) / (max(df.forecast) - min(df.forecast))
        df.previous = (df.previous - min(df.previous)) / (max(df.previous) - min(df.previous))

        # Check the forecast rating and apply weighting
        forecast_rating = df.loc[-1, 'forecast'] - df.loc[-1, 'previous']


        # Get the accuracy of the forecasts in predicting the actuals.
        # A forecast of 12 for an actual of 10 will result in 80% (note: prev used on purpose).
        accuracy = 100 - (abs(df.forecast - df.previous) / df.previous * 100)
        avg_accuracy = str(round(accuracy.tail(5).mean())) + '%'

        # Now add those values to the ratings dict
        ccy_name = event[:3]
        if ccy_name in ratings:
            ratings[ccy_name][0] += forecast_rating
            ratings[ccy_name][1] += avg_accuracy
        else:
            ratings[ccy_name][0] = forecast_rating
            ratings[ccy_name][1] = avg_accuracy

        return ratings

        '''
        what I need to do is add some logic to weight the forecast rating based on other
        factors before combining.  In essence to try to get a "relevance" score to apply
        to the rating.
        Considerations:
        1. the recent trend of 'previous' (have they been consistently positive or negative)
        2. the accuracy of the forecast (an absolute value)
        3. the accuracy of the forecast in terms of being consistently high or low

        I'll create a formula to calculate the weight of each of these and then derive the final
        forecast rating from that, which then when combined with other ratings won't allow a forecast
        to be skewed.  

        However this is low priority as it's only going to really matter when there are multiple 
        forecasts falling within the same 48hr window.
        '''

def save_ff_cal_to_db(df):
    ''' Write the data to the database. '''

    df.to_sql('ff_cal_raw', conn, if_exists='append', index=False)

build_historical_db()