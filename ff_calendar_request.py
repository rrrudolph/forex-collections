# import gspread
# from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import sys
import time
import sqlite3
from datetime import datetime
from create_db import econ_db  # local path to the database 
from tokens import ff_cal_sheet, forecast_sheet, bot


ECON_CON = sqlite3.connect(econ_db)

# month = 0
# year = 0


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
    'Final': 0.5,
    'Revised': 0.5,
}

inverted_weights = [
    'Borrowing',
    'Foriegn Currency Reserves',
    'Inventories',
    'Unemployment',
    'Delinquencies',
    'Budget',
]


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

    # Don't proceed until you have the data
    while len(df) < 10:
        time.sleep(2) 
        df = pd.DataFrame(sheet.get_all_values())

        print('Google is being slow or throttling requests. Sleeping for 1min.')
        time.sleep(60) 
        df = pd.DataFrame(sheet.get_all_values())

    return df

def _run_regex(df):
    ''' Remove the non-numeric values '''

    reg = r'([<>KkMmBbTt%])'
    df.actual = df.actual.replace(reg,'',regex=True)
    df.forecast = df.forecast.replace(reg,'',regex=True)
    df.previous = df.previous.replace(reg,'',regex=True)

    return df

def _set_dtypes(df):

    # Fill blanks with nan so I can convert to the right dtype
    df = df.replace('', np.nan)
    df.datetime = pd.to_datetime(df.datetime)
    df.forecast = df.forecast.astype(float)
    df.actual = df.actual.astype(float)
    df.previous = df.previous.astype(float)

    return df

def _clean_data(df, year, remove_non_numeric=True):

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
    
    df.time[(df.time == '') | (df.time == '[TABLE]')] = np.nan
    df.time = df.time.fillna(method='ffill')
    df.time = df.time.replace('Tentative', '00:00')

    df.date[df.date == ''] = np.nan
    df.date = df.date.fillna(method='ffill')       

    df = df[df.event != '']
    df = df[df.previous != '']
    
    df = df.drop(axis=1, columns = ['graph', 'impact', 'detail'])

    # Remove some unwanted rows
    error_causers = 'Spanish|Italian|French|Bond|Data|Vote|MPC'
    df = df[~df.event.str.contains(error_causers)]
    df = df[df.time.str.contains(':')]

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
        df = _run_regex(df)

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
        print(f'{year}...')
        for month in months.values():

            if month == current_month and year == current_year:
                break

            print(f' -{month.title()}')

            # Request data
            df = update_gsheet_cell(month, year, historical_fill=True)
            df = _clean_data(df, year, remove_non_numeric=False)

            # Update database
            df.to_sql('ff_cal_raw', ECON_CON, if_exists='append', index=False)

    ECON_CON.close()  # :/

def _set_accuracy(df, temp, normd_actual, normd_forecast):
    ''' Set the forecast accuracy for a given event '''

    accuracy = (normd_forecast - normd_actual).rolling(6).mean()
    df.loc[temp.index, 'accuracy'] = round(accuracy, 3)

    return df         

def _set_event_trend(df, normd_actual, unique_event):
    ''' Set the rolling trend of the data for a given event '''

    temp = df[(df.ccy_event == unique_event)
              &
              (df.actual.notna())
             ]

    # Get the recent trend of the actuals (normalized)
    trend = normd_actual.diff().rolling(6).mean()
    df.loc[temp.index, 'trend'] = round(trend, 3)

    return df

def _set_ccy_trend(df, unique_ccy):
    ''' Called within 'calculate_raw_db to set each currency's ccy_trend value '''

    # Refilter the df to calculate the trend of the ccy overall
    temp = df[(df.ccy == unique_ccy) 
                &
                (df.trend.notna())
                ]

    # Get the recent trend of the actuals (overall by currency)
    ccy_trend = temp.trend.rolling(7).mean()
    df.loc[temp.index, 'ccy_trend'] = round(ccy_trend, 3)

    return df

def _set_importance_weight(df, unique_event, temp):
    ''' Set weights (certain forecast events are more important than others) '''

    # randnote: if an event is 'Final...PMI', it will assign a weight of 0.5
    weight = 1
    for event in forecast_weights:     
        if event in unique_event:
            weight = forecast_weights[event]
        
    for event in inverted_weights:     
        if event in unique_event:
            weight *= -1    

    # Set the value
    df.loc[df.ccy_event == unique_event, 'weight'] = weight

    return df

def calculate_raw_db():
    ''' Add accuracy, trend, and weight columns to the raw db,
    then save as a new formatted file. This only needs to be done weekly
    when new data is added, and only needs to look at the previous 7 months. '''

    # Open file and prepare for calculations
    df = pd.read_sql('SELECT * FROM ff_cal_raw', ECON_CON)

    # In case duplicates were made at some point delete them and re-save
    df = df.drop_duplicates(subset=['datetime', 'ccy_event'])
    df.to_sql("ff_cal_raw", ECON_CON, if_exists='replace', index=False)

    df = _run_regex(df)
    df = _set_dtypes(df)

    # Run some calculations on each unique event
    unique_events = df.ccy_event.unique()
    for unique_event in unique_events:

        # Filter for just that event
        temp = df[(df.ccy_event == unique_event)
                  &
                  (df.forecast.notna())
                  &
                  (df.actual.notna())
                 ]

        if len(temp) < 2:
            continue

        # Get the recent accuracy of the forecast (a positive number means forecast missed above)
        normd_actual = (temp.actual - min(temp.actual)) / (max(temp.actual) - min(temp.actual))
        normd_forecast = (temp.forecast - min(temp.actual)) / (max(temp.actual) - min(temp.actual))

        df = _set_accuracy(df, temp, normd_actual, normd_forecast)
        df = _set_event_trend(df, normd_actual, unique_event)
        df = _set_importance_weight(df, unique_event, temp)


    # Set the rolling ccy trend
    unique_ccys = df.ccy.unique()
    for unique_ccy in unique_ccys:

        df = _set_ccy_trend(df, unique_ccy)

    # Create a col for the final outlook, used for filtering for updates
    df['outlook'] = np.nan

    # Overwrite current database file
    df.to_sql("ff_cal", ECON_CON, if_exists='replace', index=False)

def current_week_cal_request():
    ''' Get the current week's data. If today is Saturday append
    that data to the raw database and sleep for 24hrs. '''  

    # Make request
    year = datetime.today().year
    month = datetime.today().month
    df = update_gsheet_cell(month, year)

    # If today is Saturday, save the data and sleep the program for 24hrs+
    # Monday is 0, Sunday is 6
    weekday = datetime.today().weekday()
    year = datetime.today().year
    if weekday == 5:

        df = _clean_data(df, year, remove_non_numeric=False)
        df.to_sql('ff_cal_raw', ECON_CON, if_exists='append', index=False)
        df = pd.read_sql('SELECT * FROM ff_cal_raw', ECON_CON)
        df = df.drop_duplicates()
        df.to_sql('ff_cal_raw', ECON_CON, if_exists='replace', index=False)
        
        calculate_raw_db()
        time.sleep(60*2040)

    # Otherwise just a regular operation
    else:
        df = _clean_data(df, year)

    return df

def _evaluate_trend_scores(temp, forecast_rating, combined, unique_event):

    # Get the trend of the actuals
    trend = combined.trend[(combined.ccy_event == unique_event)
                            &
                            (combined.trend.shift(1).notna()) # trend lags by 1 until actual is released
    ]

    # Multiply forecast rating by trend and accuracy scores                      
    if len(trend) > 0:

        both_positive = temp[(forecast_rating > 0)
                                &
                                (trend > 0)
        ]
        both_negative = temp[(forecast_rating < 0)
                                &
                                (trend < 0)
        ]
        forecast_rating[both_positive.index] *= 1.25
        forecast_rating[both_negative.index] *= 1.25

        # Check the currency's overall data trend
        ccy_trend = combined.ccy_trend[(combined.ccy_event == unique_event)
                                        &
                                        (combined.ccy_trend.shift(1).notna()) # lags by 1 until actual is released
        ]
        both_positive = temp[(forecast_rating > 0)
                                &
                                (ccy_trend > 0)
        ]
        both_negative = temp[(forecast_rating < 0)
                                &
                                (ccy_trend < 0)
        ]
        forecast_rating[both_positive.index] *= 1.25
        forecast_rating[both_negative.index] *= 1.25

def _evaluate_accuracy_scores(forecast_rating, combined, unique_event):

        # Get the accuracy range
        accuracies = combined.ccy_trend[(combined.ccy_event == unique_event)
                                        &
                                        (combined.accuracy.shift(1).notna()) # lags by 1 until actual is released
        ]

        # For newish events there won't yet be an accuracy value                        
        if len(accuracies) > 0:  

            # Create some bins (remember, perfect accuracy is 0)
            bins = accuracies.describe()

            good = accuracies[(bins['25%'] < accuracies) 
                              &
                              (accuracies < bins['75%'])
            ]
            bad = accuracies[(bins['25%'] > accuracies) 
                             |
                             (accuracies > bins['75%'])
            ]

            forecast_rating.loc[good.index] *= 1.25
            forecast_rating.loc[bad.index] *= 0.75

def calculate_outlook(bot=bot, save_to_db=False):
    ''' Calculate the current weeks data by normalizing
    against the database. Returns a dict. '''

    # While my database is still small it makes more sense to
    # read the whole thing into memory rather than make queries, so...
    db = pd.read_sql('SELECT * FROM ff_cal', ECON_CON)
    db = _set_dtypes(db)

    df = current_week_cal_request()
    df = _set_dtypes(df)
    
    # Filter for events with forecasts
    df = df[(df.forecast.notna())]

    combined = pd.concat([db, df])
    combined = combined.drop_duplicates().reset_index(drop=True)
    
    # Run some calculations on each unique event
    unique_events = combined.ccy_event[combined.outlook.isna()].unique()
    for unique_event in unique_events:

        # Filter for just that event
        temp = combined[(combined.ccy_event == unique_event)
                        &
                        (combined.forecast.notna())
                        &
                        (combined.previous.notna())
                        ]

        if len(temp) < 2:
            continue

        # Normalize forecast and previous 
        forecast = (temp.forecast - min(temp.forecast)) / (max(temp.forecast) - min(temp.forecast))
        previous = (temp.previous - min(temp.previous)) / (max(temp.previous) - min(temp.previous))

        # Get the initial forecast change value  ~~~~~~~~~~~~~~~~~ C H A N G E D ~~~~~~~~~~~~~~~~~~~~~
        forecast_rating = forecast - previous

        # Mltiply the forecast ratings by the trend and accuracy scores
        _evaluate_trend_scores(temp, forecast_rating, combined, unique_event)
        _evaluate_accuracy_scores(forecast_rating, combined, unique_event)
    
        # Mltiply the forecast ratings by the event's weight
        forecast_rating *= db.weight[db.ccy_event == unique_event].values[0]

        # Finally, save that data
        combined.loc[forecast_rating.index, 'outlook'] = round(forecast_rating, 2)

    # Add the outlook cols data back into the same db table
    combined.to_sql('ff_cal', ECON_CON, if_exists='replace', index=False)


calculate_outlook()
def rate_monthly_outlook():
    ''' Calculate the longer term, directional outlook for each currency
    using a few select events.  Returns a dict'''

    # Load data and combine
    db = pd.read_sql('SELECT * FROM ff_cal', ECON_CON)
    df = current_week_cal_request()
    df = pd.concat([db, df])

    df = _set_dtypes(df)

    # Filter for only the events I want
    events = 'PMI|Confidence|Sentiment'
    monthly_uniques = df[df.ccy_event.str.contains(events)]
    ccys = df.ccy.unique()

    # Iter through each currency
    monthly_df = pd.DataFrame()
    for ccy in ccys:

        # Filter for only the events belonging to that currency
        ccy_uniques = monthly_uniques[monthly_uniques.ccy == ccy]

        # Get the latest data (forecast or actual) for each unique event, within each unique ccy
        total = []
        event_uniques = ccy_uniques.ccy_event.unique()
        for event in event_uniques:

            # Filter for matching event
            temp = df[(df.ccy_event == event) 
                      & 
                      ((df.forecast.notna()) 
                      &
                      (df.actual.notna()))
                    ]

            # In the rare case of a new type of event where there's
            # no history to calculate, just skip it
            if len(temp) < 2:
                continue

            # Create normalized columns
            norm_actual = (temp.actual - min(temp.actual)) / (max(temp.actual) - min(temp.actual))
            norm_forecast = (temp.forecast - min(temp.forecast)) / (max(temp.forecast) - min(temp.forecast))

            norm_actual = norm_actual.tail(2).values
            norm_forecast = norm_forecast.tail(2).values

            # If an actual is available use that, otherwise use forecast
            if norm_actual[1] is not np.nan:
                latest = norm_actual[1] - norm_actual[0]
                total.append(latest)

            else: 
                latest = norm_forecast[1] - norm_actual[0]
                total.append(latest)
            
        # Save result in the dict
        if len(total) < 1:  
            continue
        avg = sum(total) / len(total)

        temp_df={}
        temp_df['ccy'] = ccy
        temp_df['monthly'] = round(avg, 2)
        monthly_df = monthly_df.append(temp_df, ignore_index=True)

    return monthly_df


def upload_to_gsheets(df, sheet=forecast_sheet):

    # First clear the current data
    sheet.clear()

    # Ensure no nans cuz gsheets doesn't like them
    df = df.replace(np.nan, '')

    # Convert the df into series
    ccys = df.ccy.unique()

    # Insert data
    column = 0
    for ccy in ccys:

        # Move over one column
        column += 1
        date = df.event_datetime[df.ccy == ccy]
        for num, d in enumerate(date):
            sheet.update_cell(num+1, column, d.values[0])

        column += 1
        forecasts = df.forecast[df.ccy == ccy]
        for num, f in enumerate(forecasts):
            sheet.update_cell(num+1, column, f.values[0])

      
def forecast_handler(sheet=forecast_sheet, ECON_CON=ECON_CON):
    ''' Both of these return dfs.  Data will be automatically
    added to the database on Saturday through the weekly_request function
    which gets called inside each of these functions. So essentially, everything
    is getting handled behind the scenes. '''

    while True:
        week = rate_weekly_forecasts('48 hours')
        month = rate_monthly_outlook()


        # This is to combine the dfs and save the data 
        # (although it will only list ccy's with forecasts)
        combined = week.copy()
        ccys = week.ccy.unique()
        for ccy in ccys:
            index = week[week.ccy == ccy].index
            monthly = month.monthly[month.ccy == ccy].values
           
            if len(monthly) > 0:
                combined.loc[index, 'monthly'] = monthly[0]
        
        # rearrange
        combined = combined[['event_datetime', 'ccy', 'ccy_event', 'forecast', 'monthly']]

        # in case any monthlys come up nan set to 0
        combined = combined.replace(np.nan, '')

        # upload this weekly data to a gsheet (must be json serializable)
        gsheet = combined.drop(columns=['ccy'])
        gsheet.event_datetime = gsheet.event_datetime.astype(str)

        sheet.clear()
        sheet.update([gsheet.columns.values.tolist()] + gsheet.values.tolist())

        # One random change is that USD Oil Inventories really affects CAD more,
        # so if that one exists, make it into a CAD forecast
        idx = combined[combined.ccy_event.str.contains('Crude')].index
        if len(idx) > 0:
            latest_cad_monthly_value = combined.monthly[combined.ccy_event.str.contains('CAD')]
            combined.loc[idx, 'ccy'] = 'CAD'
            combined.loc[idx, 'monthly'] = latest_cad_monthly_value.tail(1)

        # this always errors first time and sometimes more
        # if len(combined) > len(historical):
        #     combined.to_sql('outlook', ECON_CON, if_exists='replace', index=False)

        # Update 
        
        # Scan again in 1 hour
        time.sleep(60*60)


# When this module gets imported, have it run this to verify that the 
# database and tables exists. If they don't, run the needed functions

def verify_db_tables_exist():
    try: 
        first = pd.read_sql('SELECT * FROM ff_cal_raw LIMIT 1', ECON_CON)
    except:
        print(f"\nCouldn't locate the db table 'ff_cal_raw' at {econ_db}.")
        print('Going to build the historical db. This will take a few mins.')
        try:
            build_historical_db()
        except Exception as e:
            print(e)
            return None
    try:
        second = pd.read_sql('SELECT * FROM ff_cal LIMIT 1', ECON_CON)
    except:
        print('\nNow just a few calculations...')
        calculate_raw_db()

    try:
        third = pd.read_sql('SELECT * FROM outlook LIMIT 1', ECON_CON)
    except:

        # This is essentially a simplified forecast_handler() 
        
        week = rate_weekly_forecasts()
        month = rate_monthly_outlook()

        # This is to combine the dfs and save the data 
        # (although it will only list ccy's with forecasts)
        combined = week.copy()
        ccys = week.ccy.unique()
        for ccy in ccys:
            index = week[week.ccy == ccy].index
            monthly = month.monthly[month.ccy == ccy].values[0]
            combined.loc[index, 'monthly'] = monthly
        

        combined.to_sql('outlook', ECON_CON, if_exists='replace', index=False)
        print('Done!')


# verify_db_tables_exist()


# if __name__ == "__main__":

    # forecast_handler()
    
    # df = update_gsheet_cell('mar', '2021', historical_fill=True)
    # df = _clean_data(df, '2021', remove_non_numeric=False)

    # # Update database
    # df.to_sql('ff_cal_raw', ECON_CON, if_exists='append', index=False)



