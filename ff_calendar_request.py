# import gspread
# from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import sys
import time
import sqlite3
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



def update_gsheet_cell(*args, historical_fill=False, sheet=ff_cal_sheet):
    ''' Handles content insertion into spreadsheet. 
    If doing a historical fill, pass month as first arg and year as second. '''
    
    if args:
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


def run_regex(df):
    ''' Remove the non-numeric values '''

    reg = r'([<>KMBT%])'
    df.actual = df.actual.replace(reg,'',regex=True)
    df.forecast = df.forecast.replace(reg,'',regex=True)
    df.previous = df.previous.replace(reg,'',regex=True)

    return df


def _set_dtypes(df):

    # Fill blanks with nan so I can convert to the right dtype
    df = df.replace('', np.nan)

    df.forecast = df.forecast.astype(float)
    df.actual = df.actual.astype(float)
    df.previous = df.previous.astype(float)

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

    # Re-order the columns
    df = df[['datetime', 'ccy', 'ccy_event', 'forecast', 'actual', 'previous']]

    # Remove all the non-numeric values
    if remove_non_numeric == True:
        df = run_regex(df)


    return df


def build_historical_db(year_start=2012):
    ''' Get historical data from 2012 through the last completed month. 
    E.g., if current date is Feb 2021, it will get everything thru Jan 2021. '''
    
    months = {
        1:'jan', 2:'feb', 3:'mar', 4:'apr', 5:'may', 6:'jun', 
        7:'jul', 8:'aug', 9:'sep', 10:'oct', 11:'nov', 12:'dec'
    }

    current_month = months[datetime.today().month]
    current_year = datetime.today().year

    years = range(year_start, current_year + 1)

    for year in years:
        for month in months.values():

            if month == current_month and year == current_year:
                sys.exit()

            # Request data
            df = update_gsheet_cell(month, year, historical_fill=True)
            time.sleep(1)

            # Clean and format a little
            # It's possible that the gsheet didn't load in time, if that happens pause and retry
            try:
                df = clean_data(df, year, remove_non_numeric=False)
            except IndexError:
                time.sleep(3)
                df = clean_data(df, year, remove_non_numeric=False)

            # Update database
            save_ff_cal_to_db(df)

    #  :/
    conn.close()


def weekly_ff_cal_request():
    ''' Get the current week's data. If today is Saturday append
    that data to the raw database and sleep for 24hrs. '''  

    # Make request
    df = update_gsheet_cell()

    # If today is Saturday, save the data and sleep the program for 24hrs
    # Monday is 0, Sunday is 6
    year = datetime.today().year
    weekday = datetime.today().weekday()
    if weekday == 5:

        df = clean_data(df, year, remove_non_numeric=False)
        save_ff_cal_to_db(df)
        calculate_raw_db(df)
        time.sleep(60*1440)

    # Otherwise just a regular operation
    else:
        df = clean_data(df, year)

    return df


def rate_weekly_forecasts(weights=forecast_weights):
    ''' Calculate the current weeks data by normalizing
    against the database. Returns a dict. '''

    # While my database is still small it makes more sense to
    # read the whole thing into memory rather than make queries, so...
    db = pd.read_sql('SELECT * FROM ff_cal', conn)
    db = _set_dtypes(db)

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


def rate_monthly_outlook():
    ''' Calculate the longer term, directional outlook for each currency
    using a few select events.  Returns a dict'''

    # Load data and combine
    db = pd.read_sql('SELECT * FROM ff_cal', conn)
    df = weekly_ff_cal_request()
    df = pd.concat([db, df])

    df = _set_dtypes(df)

    # Filter for only the events I want
    events = 'PMI|Confidence|Sentiment'
    monthly_uniques = df[df.ccy_event.str.contains(events)]
    ccys = df.ccy.unique()

    # Iter through each currency
    ccy_outlook = {}
    for ccy in ccys:

        # Filter for only the events belonging to that currency
        ccy_uniques = monthly_uniques[monthly_uniques.ccy == ccy]
        # print('ccy event_uniques:',event_uniques)

        # Get the latest data (forecast or actual) for each unique event, within each unique ccy
        total = []
        event_uniques = ccy_uniques.ccy_event.unique()
        for event in event_uniques:
            # print('event:', event)

            # Get last row of matching event where there is a forecast listed
            temp = df[(df.ccy_event == event) 
                      & 
                      ((df.forecast.notna()) 
                      | 
                      (df.actual.notna()))
                    ].copy()
            
            # In the rare case of a new type of event where there's
            # no history to calculate, just skip it
            if len(temp) < 2:
                print('rate_monthly_outlook: new event!')
                print(ccy, event)
                continue

            # Create normalized columns
            temp['norm_actual'] = (temp.actual - min(temp.actual)) / (max(temp.actual) - min(temp.actual))
            temp['norm_forecast'] = (temp.forecast - min(temp.forecast)) / (max(temp.forecast) - min(temp.forecast))
            temp['norm_previous'] = (temp.previous - min(temp.previous)) / (max(temp.previous) - min(temp.previous))

            temp = temp.tail(2).reset_index()

            # If an actual is available use that, otherwise use forecast
            if temp.actual is not np.nan:
                latest = temp.norm_actual[1] - temp.norm_actual[0]
                total.append(latest)

            else: 
                latest = temp.norm_forecast[1] - temp.norm_forecast[0]
                total.append(latest)
            
        # Save result in the dict
        avg = sum(total) / len(total)
        ccy_outlook[ccy] = avg

    return ccy_outlook
        

def calculate_raw_db():
    ''' Add accuracy, trend, and weight columns to the raw db,
    then save as a new formatted file. This only needs to be done weekly
    when new data is added, and only needs to look at the previous 7 months. '''

    # Open file and prepare for calculations
    df = pd.read_sql('SELECT * FROM ff_cal_raw', conn)
    df = run_regex(df)
    df = _set_dtypes(df)

    # Find unique events (forecasts)
    unique_events = df.ccy_event.unique()

    # Run some calculations on each unique event
    for unique_event in unique_events:

        # Normalize the actuals
        df['norm_actual'] = (df.actual - min(df.actual)) / (max(df.actual) - min(df.actual))

        # Apply weights (certain forecast events are more important than others)
        # randnote: if an event is 'Final...PMI', it will assign a weight of 0.5
        weight = None
        for event in forecast_weights:     
            
            if event in unique_event:
                weight = forecast_weights[event]
           
           # If nothing is found set to 1
            if weight is None:
                weight = 1
        
        temp = df[df.ccy_event == unique_event]
        df.loc[temp.index, 'weight'] = weight


        # Get the recent accuracy of the forecast as a percentage (returns lots of 'inf's)
        accuracy = temp.actual / temp.forecast 
        accuracy = accuracy.replace([-np.Inf, np.Inf], 0)
        df.loc[temp.index, 'abs_accuracy'] = round(accuracy.rolling(6).mean(), 2)

        # Get the recent trend of the actuals (per unique forecast)
        trend = temp.norm_actual.diff().rolling(4).mean()
        df.loc[temp.index, 'trend'] = trend

        # Get the recent trend of the actuals (overall by currency)
        # Make a new temp df
        temp = df[df.ccy == temp.ccy.values[0]]
        ccy_trend = temp.trend.rolling(7).mean()
        df.loc[temp.index, 'ccy_trend'] = ccy_trend


    # Overwrite current database file
    df.to_sql("ff_cal", conn, if_exists='replace', index=True)


def save_ff_cal_to_db(df):
    ''' Write the data to the database. '''

    df.to_sql('ff_cal_raw', conn, if_exists='append', index=False)


def rating_handler():
    ''' Both of these return dicts.  Data will be automatically
    added to the database on Saturday through the weekly_request function
    which gets called inside each of these functions. So essentially, everything
    is getting handled behind the scenes. '''

    outlook = {}
    outlook['weekly'] = rate_weekly_forecasts()
    outlook['monthly'] = rate_monthly_outlook()

    return outlook