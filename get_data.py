import pandas as pd
import MetaTrader5 as mt5
import sqlite3
import time
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import datetime as dt

# This module gets data and updates the database

# Notes: 
# lfd, hfd = low frequency data, high frequency data

# REMEMBER TO REMOVE THE 'YEAR - 1" PART FROM FFCAL


mt5_timeframes = {
                '1': mt5.TIMEFRAME_M1,
                '5': mt5.TIMEFRAME_M5,
                '15': mt5.TIMEFRAME_M15
                }

mt5_symbols = [
    'EURUSD',
    'GBPUSD',
    'USDJPY',
    'NZDUSD',
    'USDCAD',
    'AUDUSD',
    'EURJPY',
    'GBPJPY',
    'CADJPY',
    'AUDJPY',
    'NZDJPY',
    'EURCAD',
    'EURAUD',
    'GBPCAD',
    'EURGBP',
    'AUDCAD',
    'NZDCAD',
    'AUDNZD',
    'GBPAUD',
    'GBPNZD',
    'EURNZD',
    'USDCHF',
    'GBPCHF',
    'CADCHF',
    'USDPLN',
    'USDSEK',
    'USDMXN',
    'USDZAR',
    'GBPSGD',
    'USDZAR',
    'XAUUSD',
    'XAGUSD',
    'XTIUSD',
    'DE30',
    'US30',
    'US500',
    'USTEC'
]

fin_symbols = [
    'MOO', 
    'HYG', 
    'VIXM', 
    'VIXY', 
    'XLF', 
    'XLU', 
    'XLY', 
    'XLP', 
    'IWF', 
    'IWD', 
    'BAC', 
    'REET'
]

te_countries = [
    'https://tradingeconomics.com/united-states/indicators',
    'https://tradingeconomics.com/euro-area/indicators',
    'https://tradingeconomics.com/united-kingdom/indicators',
    'https://tradingeconomics.com/japan/indicators',
    'https://tradingeconomics.com/canada/indicators',
    'https://tradingeconomics.com/australia/indicators',
    'https://tradingeconomics.com/new-zealand/indicators',
    'https://tradingeconomics.com/switzerland/indicators',
]
# This is to handle finnhub 'num_candles' request automatically
seconds_per_candle = {'1': 60,
                      '5': 300,
                     '15': 900,
                     '30': 1800,
                     '60': 3600,
                      'D': 86400
                     }


class UpdateDB():
    ''' This class has methods to both request data and
        update the SQLite db '''

    def connect_to_db():
        try:
            UpdateDB.conn = sqlite3.connect(r'C:\Users\Rudy\Desktop\codez\forex.db')
            UpdateDB.c = UpdateDB.conn.cursor()
        
        except Exception as e:
            print("\n ~~ ERROR 'get_data.py UpdateDB': problem connecting to forex.db ~~ ")
            print(e, '\n')

    def save_ohlc_to_db(df):
        for i in df.index:
            params = (  df.loc[i, 'datetime'], 
                        df.loc[i, 'symbol'], 
                        df.loc[i, 'timeframe'], 
                        df.loc[i, 'open'], 
                        df.loc[i, 'high'], 
                        df.loc[i, 'low'], 
                        df.loc[i, 'close'], 
                        df.loc[i, 'volume'])

            UpdateDB.c.execute("INSERT INTO ohlc VALUES (?,?,?,?,?,?,?,?)", params)
        UpdateDB.conn.commit()

    def save_hfd_to_db(df):
        for i in df.index:
            params = (  df.loc[i, 'date'], 
                        df.loc[i, 'time'], 
                        df.loc[i, 'ccy'], 
                        df.loc[i, 'event'], 
                        df.loc[i, 'actual'], 
                        df.loc[i, 'forecast'], 
                        df.loc[i, 'previous'])

            UpdateDB.c.execute("INSERT INTO hfd VALUES (?,?,?,?,?,?,?)", params)
        UpdateDB.conn.commit()

    def save_lfd_to_db(df):
        for i in df.index:
            params = (  df.loc[i, 'reference'], 
                        df.loc[i, 'category'], 
                        df.loc[i, 'name'], 
                        df.loc[i, 'last'], 
                        df.loc[i, 'previous'], 
                        df.loc[i, 'range'], 
                        df.loc[i, 'frequency'])

            UpdateDB.c.execute("INSERT INTO lfd VALUES (?,?,?,?,?,?,?)", params)
        UpdateDB.conn.commit()

    def _get_latest_ohlc_datetime(symbol, timeframe):
        ''' Get the last ohlc timestamp for each symbol '''

        params = (symbol, timeframe)
        # If the db is blank this will error with 'NoneType'
        try:
            UpdateDB.c.execute('''SELECT datetime 
                                    FROM ohlc
                                    WHERE symbol == ?
                                    AND timeframe == ?
                                    ORDER BY datetime DESC
                                    LIMIT 1''', (params))
            return int(UpdateDB.c.fetchone()[0])
        except: TypeError
        pass

        # dealing with single param explanation:
        # https://stackoverflow.com/questions/16856647/sqlite3-programmingerror-incorrect-number-of-bindings-supplied-the-current-sta
         
    def _set_start_and_end_times(symbol, timeframe, num_candles=999):
        ''' Start request 'start' time based on the last row in the db, 
            otherwise if there's no data just request a bunch of candles '''

        timeframe = str(timeframe)
        UpdateDB.end = round(time.time())        
        if UpdateDB._get_latest_ohlc_datetime(symbol, timeframe):
            UpdateDB.start = UpdateDB._get_latest_ohlc_datetime(symbol, timeframe)
        else: 
            UpdateDB.start = UpdateDB.end - (num_candles * seconds_per_candle[timeframe])

        return UpdateDB.start, UpdateDB.end

    def finnhub_ohlc_request(symbol, timeframe, num_candles=999):
        ''' Import note, this doesn't account for weekend gaps. So requesting small 
            amounts of M1 candles can cause problems on weekends.  Max returned candles
            seems to be about 650.  Finnhub only returns closed candles, not the current candle.  '''

        # Ensure timeframe is a string
        timeframe = str(timeframe)
        symbol = str(symbol)
        
        UpdateDB._set_start_and_end_times(symbol, timeframe)

        if 'OANDA' in symbol:
            r = requests.get(f'https://finnhub.io/api/v1/forex/candle?symbol={symbol}&resolution={timeframe}&from={UpdateDB.start}&to={UpdateDB.end}&token=budmt1v48v6ped914310')
        else:
            r = requests.get(f'https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution={timeframe}&from={UpdateDB.start}&to={UpdateDB.end}&token=budmt1v48v6ped914310')
        
        # Check for a bad request
        r = r.json()
        if r['s'] == 'no_data':
            print('Finnhub OHLC Request error. Check parameters:')
            print(' - symbol:', symbol)
            print(' - timeframe:', timeframe)
            print(' - start:', start)
            print(' - end:  ', end)
            print('current time:', time.time())
        else:
            UpdateDB.finnhub_df = pd.DataFrame(data=r)
            UpdateDB.finnhub_df = UpdateDB.finnhub_df.rename(columns={'t':'datetime', 'o':'open', 'h':'high', 'l':'low', 'c':'close', 'v':'volume'})

            # Ensure all data is SQLite friendly
            UpdateDB.finnhub_df['symbol'] = str(symbol)
            UpdateDB.finnhub_df['timeframe'] = str(timeframe)
            UpdateDB.finnhub_df['datetime'] = UpdateDB.finnhub_df['datetime'].astype(str)
            UpdateDB.finnhub_df['open'] = UpdateDB.finnhub_df['open'].astype(float)
            UpdateDB.finnhub_df['high'] = UpdateDB.finnhub_df['high'].astype(float)
            UpdateDB.finnhub_df['low'] = UpdateDB.finnhub_df['low'].astype(float)
            UpdateDB.finnhub_df['close'] = UpdateDB.finnhub_df['close'].astype(float)
            UpdateDB.finnhub_df['volume'] = UpdateDB.finnhub_df['volume'].astype(float)
            UpdateDB.finnhub_df = UpdateDB.finnhub_df[['datetime', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume']]
        
    def mt5_ohlc_request(symbol, timeframe):
        ''' Similar to Finnhub, MT5 will not return the current open candle. '''

        if not mt5.initialize(login=50341259, server="ICMarkets-Demo",password="ZhPcw6MG"):
            print("\n ~~~ ERROR: MT5 initialize() failed, error code =", mt5.last_error())
            quit()

        # Ensure proper data formatting
        timeframe = str(timeframe)
        timeframe = mt5_timeframes[timeframe]

        UpdateDB._set_start_and_end_times(symbol, timeframe)

        data = mt5.copy_rates_range(symbol, timeframe, UpdateDB.start, UpdateDB.end)
        UpdateDB.mt5_df = pd.DataFrame(data)
        UpdateDB.mt5_df = UpdateDB.mt5_df.rename(columns={'time': 'datetime', 'tick_volume': 'volume'})
        
        # Ensure all data is SQLite friendly
        UpdateDB.mt5_df['symbol'] = str(symbol)
        UpdateDB.mt5_df['timeframe'] = str(timeframe)
        UpdateDB.mt5_df['datetime'] = UpdateDB.mt5_df['datetime'].astype(str)
        UpdateDB.mt5_df['open'] = UpdateDB.mt5_df['open'].astype(float)
        UpdateDB.mt5_df['high'] = UpdateDB.mt5_df['high'].astype(float)
        UpdateDB.mt5_df['low'] = UpdateDB.mt5_df['low'].astype(float)
        UpdateDB.mt5_df['close'] = UpdateDB.mt5_df['close'].astype(float)
        UpdateDB.mt5_df['volume'] = UpdateDB.mt5_df['volume'].astype(float)

    def te_lfd_request(country):
        ''' Open each country's indicators web page '''

        # This returns a list of 14 df's
        data = pd.read_html(country)

        num = len(data)
        if num != 14:
            name = country.split('/')[3]
            print(f"\n ~~~ WARNING: TradingEconomics returned unusual results for {name}")
            print(f"Normally there are 14 df's but {num} were returned ~~~ \n")
        
        # The first and last are N/A (just a table of links)
        data = data[1:13]

        # Prepare data for combination into single df
        for num, df in enumerate(data):
            df['category'] = df.columns[0]
            df = df.rename(columns = {df.columns[0]: 'name'})
            df = df.drop(columns = 'Unnamed: 6')
            df = df.rename(str.lower, axis='columns')
            data[num] = df

        # Combine the list of dfs into a single df        
        UpdateDB.te_df = pd.concat(data)

        # Ensure all data is SQLite friendly
        UpdateDB.te_df['reference'] = UpdateDB.te_df['reference'].astype(str)
        UpdateDB.te_df['category'] = UpdateDB.te_df['category'].astype(str)
        UpdateDB.te_df['name'] = UpdateDB.te_df['name'].astype(str)
        UpdateDB.te_df['last'] = UpdateDB.te_df['last'].astype(float)
        UpdateDB.te_df['previous'] = UpdateDB.te_df['previous'].astype(float)
        UpdateDB.te_df['range'] = UpdateDB.te_df['range'].astype(str)
        UpdateDB.te_df['frequency'] = UpdateDB.te_df['frequency'].astype(str)

    def ff_hfd_request():
        # use creds to create a client to interact with the Google Drive API
        scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(r'C:\Users\Rudy\Desktop\codez\client_secret.json', scope)
        client = gspread.authorize(creds)

        # Find a workbook by name and open the first sheet
        sheet = client.open("ffcal").sheet1

        # Extract and print all of the values
        data = sheet.get_all_records()

        # Clean data cuz its super ugly
        df = pd.DataFrame(data)
        df = df[1:]
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

        # Convert blanks to nans and then fill               
        df.date[df.date == ''] = np.nan
        df.date = df.date.fillna(method='ffill')

        # Remove rows without data
        df = df[df.actual != 'Bank Holiday']
        df = df[df.time != '[TABLE]']
        df = df[df.previous != '']

        # Fix the date and add the current year
        year = dt.date.today().year -1
        for i in df.index:
            df.loc[i, 'date'] = df.loc[i, 'date'][3:] + ' ' + str(year) 

        # Ensure all data is SQLite friendly
        UpdateDB.cal_df = df
        UpdateDB.cal_df['date'] = UpdateDB.cal_df['date'].astype(str)
        UpdateDB.cal_df['time'] = UpdateDB.cal_df['time'].astype(str)
        UpdateDB.cal_df['ccy'] = UpdateDB.cal_df['ccy'].astype(str)
        UpdateDB.cal_df['event'] = UpdateDB.cal_df['event'].astype(str)
        UpdateDB.cal_df['actual'] = UpdateDB.cal_df['actual'].astype(str)
        UpdateDB.cal_df['forecast'] = UpdateDB.cal_df['forecast'].astype(str)
        UpdateDB.cal_df['previous'] = UpdateDB.cal_df['previous'].astype(str)



def main():        
    ''' Open the db connection, update the data '''
    
    UpdateDB.connect_to_db()

    # Update Finnhub data (confirmed working 1/1/2021)
    for symbol in fin_symbols:
        UpdateDB.finnhub_ohlc_request(symbol, 5)
        UpdateDB.save_ohlc_to_db(UpdateDB.finnhub_df)

    # Update MT5 data (confirmed working 1/1/2021)
    for symbol in mt5_symbols:
        UpdateDB.mt5_ohlc_request(symbol, 5)
        UpdateDB.save_ohlc_to_db(UpdateDB.mt5_df)

    # Update Forex Factory calendar
    UpdateDB.ff_hfd_request()
    UpdateDB.save_hfd_to_db(UpdateDB.cal_df)

    # Update Trading Economics data 
    for country in te_countries:
        UpdateDB.te_hfd_request(country)
        UpdateDB.save_lfd_to_db(UpdateDB.te_df)
        # Getting error: "sqlite3.InterfaceError: Error binding parameter 0 - probably unsupported type."
        # Even though I've explicitly set all of the data types...
main()


