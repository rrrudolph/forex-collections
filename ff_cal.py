import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import datetime as dt
import sqlite3
from datetime import datetime

path = r'C:\Users\Rudy\Desktop\codez\forex.db'

''' 
    This module will make a request for current, weekly data every hour.
    It will then query the db in order to perform it's calculations, 
    but will only update the db with the weekly data on the weekend
    once all the 'actuals' are set. 
    
    Functions:
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

def weekly_ff_cal_request():
    # use creds to create a client to interact with the Google Drive API
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(r'C:\Users\Rudy\Desktop\codez\client_secret.json', scope)
    client = gspread.authorize(creds)

    # Find a workbook by name and open the first sheet
    sheet = client.open("ffcal").sheet1

    # Extract and print all of the values
    data = sheet.get_all_records()

    return data

def format_weekly_data():

    data = weekly_ff_cal_request()

    # Clean data cuz its really really bad
    df = pd.DataFrame(data)
    # df = df[1:]
    df = df.rename(str.lower, axis='columns')
    df = df.drop(axis=1, columns = ['graph', 'impact', 'detail'])
    df = df.rename(columns = {
            df.columns[0]: 'date', 
            df.columns[1]: 'time', 
            df.columns[2]: 'ccy', 
            df.columns[3]: 'event', 
            df.columns[4]: 'actual', 
            df.columns[5]: 'forecast', 
            df.columns[6]: 'previous', 
            })
    # df = df.dropna(axis=1)
    # Convert blanks to nans and then fill               
    df.date[df.date == ''] = np.nan
    df.date = df.date.fillna(method='ffill')

    # Remove some unwanted rows
    df = df[df.time != '[TABLE]']
    df = df[df.previous != '']
    error_causers = ['Spanish', 'Italian', 'French', 'Bond', 'Data', 'Vote', 'MPC']
    df = df[~df.event.str.contains(error_causers)]

    # Fix the date and add the current year
    year = dt.date.today().year -1
    df['date'] = df.apply(lambda x: x['date'][3:] + ' ' + str(year), axis=1) 

    # Organize some columns and change time from EST to CST
    df['datetime'] = pd.to_datetime(df.date + ' ' + df.time)
    df.datetime = df.datetime - pd.Timedelta('1 hour')
    df.event = df.ccy + ' ' + df.event
    df = df.drop(columns = ['date', 'time', 'ccy'])

    # Remove all the non-numeric values
    reg = r'([<>KMBT%])'
    df.previous = df.previous.replace(reg,'',regex=True)
    df.forecast = df.forecast.replace(reg,'',regex=True)
    df.forecast = df.forecast.replace(reg,'',regex=True)

    print(df)
    return df

def read_database(event, path=path):
    
    # Grab dat db
    conn = sqlite3.connect(path)
    c = conn.cursor()

    query = f'''(SELECT * 
                FROM ff_cal
                WHERE event = {event}; 
                )'''

    return c.execute(query).fetchall()


def evaluate_forecast_rating(weights=event_weights):

    weekly_df = format_weekly_data()
    
    # Loop through events with forecasts which are set to drop within the next 48 hours
    time_horizon = datetime.now() + pd.Timedelta('2 days')
    filter_ = ((weekly_df.forecast.notna()) & (weekly_df.datetime < time_horizon))
    
    for event in weekly_df.event[filter_]:
        db_df = read_database(event)

        # Database values are saved as string so convert to proper dtype
        db_df.datetime = pd.to_datetime(db_df.datetime)
        db_df.actual = db_df.actual.astype(float)
        db_df.forecast = db_df.forecast.astype(float)
        db_df.previous = db_df.previous.astype(float)

        # Combine the weekly row with the db and sort
        df = pd.concat(weekly_df[weekly_df.event == event], db_df)
        df = df.sort_values(by=['datetime'])

        # Normalize the data
        df.actual = (df.actual - min(df.actual)) / (max(df.actual) - min(df.actual))
        df.forecast = (df.forecast - min(df.forecast)) / (max(df.forecast) - min(df.forecast))
        df.previous = (df.previous - min(df.previous)) / (max(df.previous) - min(df.previous))

        # Check the forecast rating and apply weighting
        forecast_rating = df.loc[-1, 'forecast'] - df.loc[-1, 'previous']

        # Note, if an event is Final...PMI, it will assign a weight of 0.5
        for item in weights:     
            if item in event:
                weight = weights[item]
        
        forecast_rating *= weight


        ## Okkk done for the night. at this point I need to accumulate all
        # of the forecasts and send them off to the central ratings.csv