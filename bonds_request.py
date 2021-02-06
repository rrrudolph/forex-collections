import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import time

from create_db import setup_conn, path  # local path to the database 
from tokens import bonds_gsheet
from symbols_lists import bonds_table_nums

conn, c = setup_conn(path)


def _update_cell(key, sheet=bonds_gsheet):
    ''' Handles content insertion into spreadsheet. '''
    
    cell = fr"""=importhtml("https://www.investing.com/rates-bonds/world-government-bonds", "table", {key})"""
    sheet.update_cell(1,1, cell)

    time.sleep(1)

    data = sheet.get_all_values() # list of lists
    df = pd.DataFrame(data)

    return df


def _unpack_name(df):
    ''' Simple parser '''

    name = df[1].tail(1).values[0]
    country = name.split()[:-1]

    # Unpack from list (New Zealand is the only 2 word name)
    country = country[0] if len(country) == 1 else country[0] + ' ' + country[1]

    return country


def update_bonds_table_nums(countries=bonds_table_nums, sheet=bonds_gsheet):
    ''' Check each table on the web page to see if it is a country
    who's data I want. This is so if the website changes anything I can
    update my code easily. '''

    for num in range(1, 99):

        # Extract the country info
        df = _update_cell(num)
        
        country = _unpack_name(df)

        # Make a dict with the country name and 
        if country in countries:
            countries[country] = num
            print(countries)

            if country == 'U.S.':
                return countries

    return countries

# update_bonds_table_nums()

def bonds_data_request(countries=bonds_table_nums, sheet=bonds_gsheet):
    
    for k,v in countries.items():
        print(k, v)

        df = _update_cell(v)
        # col 1 is the name, 2 is the price, 8 is the time
        # times are EST and can be formatted 0:23:45 or 03/02
        # if its a price from a previous day
        print(df['Name'])  # nope
        print(df)
        
        # Find the desired rows (2y and 10y)
        
bonds_data_request()


def read_database(event, conn=conn, path=path):
    
    query = f'''(SELECT * 
                FROM ff_cal
                WHERE event = {event}; 
                )'''

    df = pd.read_sql(query, conn)

    return df
