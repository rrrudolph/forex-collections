import pandas as pd
import numpy as np
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import sqlite3
from datetime import datetime

''' This module reads from the forex factory calendar via google sheets to create
a sqlite database. It then runs numerous calculations on the data to derive 
synthetic forecast values in an attempt to see the real importance that a forecast
might have to institutions. There's a few extra functions to make it a bit less error 
prone.  For example, if the database doesn't exist it will automatically get created, if 
its near the end of the month it will request the next month as well, and if
the module isn't run for a while any missing data will be auto populated.'''

econ_db = r'C:\Users\ru\forex\db\economic_data.db'
ECON_CON = sqlite3.connect(econ_db)

# This section has to go here rather than in the tokens file because my 
# multiprocessing modules end up causing an error by calling it too much
SCOPE = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
CREDS = ServiceAccountCredentials.from_json_keyfile_name(r'C:\Users\ru\forex\gsheets_token.json', SCOPE)
CLIENT = gspread.authorize(CREDS)

#DASHBOARD sheet(0)
forecast_sheet = CLIENT.open('data').get_worksheet(1) 
ohlc_sheet = CLIENT.open('data').get_worksheet(2) 
ff_cal_sheet = CLIENT.open('data').get_worksheet(4)
bonds_sheet = CLIENT.open('bonds').get_worksheet(1)

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

# Map datetime month values to what forex factory needs in their url
months = {
    1:'jan', 2:'feb', 3:'mar', 4:'apr', 5:'may', 6:'jun', 
    7:'jul', 8:'aug', 9:'sep', 10:'oct', 11:'nov', 12:'dec'
}


def _update_gsheet_cell(month, year, sheet=ff_cal_sheet):
    ''' Handles content insertion into spreadsheet. 
    If doing a historical fill, pass month as first arg and year as second. '''

    # Convert to a 3 letter month value
    month = months[month]
    cell = fr"""=importhtml("https://www.forexfactory.com/calendar?month={month}.{year}", "table", 4)"""

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

def get_historical_calendar_data(year_start=2007):
    ''' Build the historical database by requesting one month at a
    time from forexfactory.com.  It often can take quite a while but
    the "_update_gsheet_cell" function will handle any delays. '''

    current_year = datetime.today().year
    current_month = datetime.today().month
    for year in range(year_start, current_year + 1):
        print(f'{year}...')
        for month in months:

            if month == current_month and year == current_year:
                break

            print(f' -{months[month].title()}')

            # Request data
            df = _update_gsheet_cell(month, year)
            df = _clean_data(df, year, remove_non_numeric=False)

            # Update database
            df.to_sql('ff_cal_raw', ECON_CON, if_exists='append', index=False)
    
def _fill_any_missing_data():
    ''' Compare the current month to the last month in the database.
    If there is a gap, ie if its March but the last month in db is Jan, 
    fill any missing months and bring the db up to a current state. '''

    last_date = pd.read_sql('''SELECT datetime 
                        FROM ff_cal_raw
                        ORDER BY datetime DESC
                        LIMIT 1''', ECON_CON)
    last_db_month = int(str(last_date).split()[2].split('-')[1])

    # See if the difference between the current month and that month is > 1
    current_month = datetime.today().month
    missing_months = abs(current_month - last_db_month) # abs used in case of year roll over 
    for x in range(1, missing_months):
        missing_month = months[current_month - x]
        df = _update_gsheet_cell(missing_month, datetime.today().year)
        df = _clean_data(df, datetime.today().year, remove_non_numeric=False)
        df.to_sql('ff_cal_raw', ECON_CON, if_exists='append', index=False)
    
    # Now get the ff_cal table filled in
    calculate_raw_db()

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

    # Overwrite current database file
    df.to_sql("ff_cal", ECON_CON, if_exists='replace', index=False)

def current_month_cal_request():
    ''' Get the current month's data. If current time is equal
    or greater than the last datetime of the data, append to
    the database and sleep for 24hrs. '''  

    _fill_any_missing_data()

    # Make request
    year = datetime.today().year
    month = datetime.today().month
    df = _update_gsheet_cell(month, year)
    df = _clean_data(df, year)

    # If its getting close to the end of the month, grab next months data too
    if df.datetime.tail(1).values[0] - pd.to_datetime(datetime.now()) < pd.Timedelta('4 days'):
        month += 1 
        if month == 13:
            month = 1
            year += 1
        df2 = _update_gsheet_cell(month, year)
        df2 = _clean_data(df2, year)
        df = pd.concat(df, df2)
    
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

def calculate_short_term_outlook() -> pd.DataFrame:
    ''' Calculate the current weeks data by normalizing
    against the database.'''

    # While my database is still small it makes more sense to
    # read the whole thing into memory rather than make queries, so...
    db = pd.read_sql('SELECT * FROM ff_cal', ECON_CON)
    db = _set_dtypes(db)

    df = current_month_cal_request()
    df = _set_dtypes(df)
    
    # Filter for events with forecasts
    df = df[(df.forecast.notna())]

    combined = pd.concat([db, df])
    combined = combined.drop_duplicates(subset=['datetime', 'ccy_event']).reset_index(drop=True)

    # Run some calculations on each unique event
    unique_events = combined.ccy_event.unique()
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

        # Get the initial forecast change value
        forecast_rating = forecast - previous

        # Mltiply the forecast ratings by the trend and accuracy scores
        _evaluate_trend_scores(temp, forecast_rating, combined, unique_event)
        _evaluate_accuracy_scores(forecast_rating, combined, unique_event)
    
        # Mltiply the forecast ratings by the event's weight
        forecast_rating *= db.weight[db.ccy_event == unique_event].values[0]

        # Add the outlook cols data back into the same db table
        combined.loc[forecast_rating.index, 'short_term'] = round(forecast_rating, 2)

    return combined

def calculate_long_term_outlook() -> pd.DataFrame:
    ''' Calculate the longer term, directional outlook for each currency
    using a few select events.'''

    # Load data and combine
    db = pd.read_sql('SELECT * FROM ff_cal', ECON_CON)
    df = current_month_cal_request()
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

        temp_df = {}
        temp_df['ccy'] = ccy
        temp_df['long_term'] = round(avg, 2)
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
     
def forecast_handler(sheet=forecast_sheet):
    ''' Both of these return dfs.  Data will be automatically
    added to the database on Saturday through the weekly_request function
    which gets called inside each of these functions. So essentially, everything
    is getting handled behind the scenes. '''

    week = calculate_short_term_outlook()
    month = calculate_long_term_outlook()

    # This is to combine the dfs and save the data 
    # (although it will only list ccy's with forecasts)
    combined = week.copy()
    ccys = week.ccy.unique()
    for ccy in ccys:
        index = week[week.ccy == ccy].index
        monthly = month.long_term[month.ccy == ccy].values
        
        if len(monthly) > 0:
            combined.loc[index, 'long_term'] = monthly[0]
    
    # rearrange
    combined = combined[['datetime', 'ccy', 'ccy_event', 'short_term', 'long_term']]

    # in case any monthlys come up nan set to 0
    combined = combined.replace(np.nan, '')

    # upload this weekly data to a gsheet (must be json serializable)
    gsheet = combined.drop(columns=['ccy'])
    gsheet.datetime = gsheet.datetime.astype(str)

    sheet.clear()
    sheet.update([gsheet.columns.values.tolist()] + gsheet.values.tolist())

    # One random change is that USD Oil Inventories really affects CAD more,
    # so if that one exists, make it into a CAD forecast
    idx = combined[combined.ccy_event.str.contains('Crude')].index
    if len(idx) > 0:
        latest_cad_monthly_value = combined.long_term[combined.ccy_event.str.contains('CAD')]
        combined.loc[idx, 'ccy'] = 'CAD'
        combined.loc[idx, 'long_term'] = latest_cad_monthly_value.tail(1)

    # Save
    combined.to_sql('forecast', ECON_CON, if_exists='replace', index=False)

def verify_db_tables_exist():
    ''' When this module gets ran (even just imported) the first time,
    I need to ensure the database tables exist.  This will set everything
    up in case they don't. '''

    try: 
        pd.read_sql('SELECT * FROM ff_cal_raw LIMIT 1', ECON_CON)
    except:
        print(f"\nCouldn't locate the db table 'ff_cal_raw' at {econ_db}.")
        print('Going to build the historical db. This will take a while.')
        try:
            get_historical_calendar_data()
        except Exception as e:
            print('Got an error:', e)
            print('Exiting "get_historical_calendar_data"')
            return None

    try:
        pd.read_sql('SELECT * FROM ff_cal LIMIT 1', ECON_CON)
    except:
        print('\nNow just a few calculations...')
        calculate_raw_db()

    try:
        pd.read_sql('SELECT * FROM forecast LIMIT 1', ECON_CON)
    except:
        print('Creating the forecast database table.')

        # This is essentially a simplified forecast_handler()
        week = calculate_short_term_outlook()
        month = calculate_long_term_outlook()

        # This is to combine the dfs and save the data 
        # (although it will only list ccy's with forecasts)
        combined = week.copy()
        ccys = week.ccy.unique()
        for ccy in ccys:
            index = week[week.ccy == ccy].index
            if not month.long_term[month.ccy == ccy].empty:
                combined.loc[index, 'long_term'] = month.long_term[month.ccy == ccy].values[0]
        
        # Save
        combined = combined[['datetime', 'ccy', 'ccy_event', 'short_term', 'long_term']]
        combined.to_sql('forecast', ECON_CON, if_exists='replace', index=False)
        print('Done!')


if __name__ == "__main__":
    
    verify_db_tables_exist()
    while True:
        forecast_handler()
        time.sleep(60*60)
    



