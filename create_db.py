import sqlite3

path = r'C:\Users\Rudy\Desktop\codez\forex.db'
conn = sqlite3.connect(path)
#conn = sqlite.connect(':memory:')  would only live in RAM
c = conn.cursor()


''' THESE ARE THE RAW DATA TABLES '''

# Create the FF calendar table
# c.execute("""CREATE TABLE ff_cal_raw (
#             date TEXT,
#             time TEXT,
#             ccy TEXT,
#             event TEXT,
#             actual TEXT,
#             forecast TEXT,
#             previous TEXT
#             )""")

# Create trading economics data table
# Trade(name) Last(data) Reference(recent date) Previous(data) Range(hi:low) Frequency
# c.execute("""CREATE TABLE te_data_raw (
#             country TEXT,
#             date TEXT,
#             category TEXT,
#             name TEXT,
#             actual TEXT,
#             previous TEXT,
#             range TEXT,
#             frequency TEXT
#             )""")

# # Create the MT5/Finnhub forex OHLCV table
# c.execute("""CREATE TABLE ohlc_raw (
#             datetime TEXT,
#             symbol TEXT,
#             timeframe TEXT,
#             open REAL,
#             high REAL,
#             low REAL,
#             close REAL,
#             volume REAL
#             )""")


''' THESE ARE THE FORMATTED DATA TABLES '''

# Create the FF calendar table
c.execute("""CREATE TABLE ff_cal (
            date TEXT,
            time TEXT,
            ccy TEXT,
            event TEXT,
            actual TEXT,
            forecast TEXT,
            previous TEXT
            )""")

# Create trading economics data table
# Trade(name) Last(data) Reference(recent date) Previous(data) Range(hi:low) Frequency
c.execute("""CREATE TABLE te_data (
            country TEXT,
            date TEXT,
            category TEXT,
            name TEXT,
            actual TEXT,
            previous TEXT,
            range TEXT,
            frequency TEXT
            )""")

# I think i'll just use the raw ohlc data.. its pretty good as is
# c.execute("""CREATE TABLE ohlc (
#             datetime TEXT,
#             symbol TEXT,
#             timeframe TEXT,
#             open REAL,
#             high REAL,
#             low REAL,
#             close REAL,
#             volume REAL
#             )""")



conn.commit()
conn.close()
