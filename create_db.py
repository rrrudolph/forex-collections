import sqlite3

path = r'C:\Users\Rudy\Desktop\codez\forex.db'
conn = sqlite3.connect(path)
#conn = sqlite.connect(':memory:')  would only live in RAM
c = conn.cursor()

# Create the FF calendar table
# c.execute("""CREATE TABLE hfd (
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
c.execute("""CREATE TABLE lfd (
            country TEXT,
            date TEXT,
            category TEXT,
            name TEXT,
            actual TEXT,
            previous TEXT,
            range TEXT,
            frequency TEXT
            )""")

# # Create the MT5/Finnhub forex OHLCV table
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
