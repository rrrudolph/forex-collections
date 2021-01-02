import sqlite3

path = r'C:\Users\Rudy\Desktop\codez\forex.db'
conn = sqlite3.connect(path)
#conn = sqlite.connect(':memory:')  would only live in RAM
c = conn.cursor()

# Create the FF calendar table
c.execute("""CREATE TABLE hfd (
            datetime REAL,
            name TEXT,
            actual REAL,
            forecast REAL,
            previous REAL
            )""")

# Create trading economics data table
# Trade(name) Last(data) Reference(recent date) Previous(data) Range(hi:low) Frequency
c.execute("""CREATE TABLE lfd (
            date TEXT,
            category TEXT,
            name TEXT,
            actual REAL,
            previous REAL,
            range TEXT,
            frequency TEXT
            )""")

# # Create the MT5/Finnhub forex OHLCV table
c.execute("""CREATE TABLE ohlc (
            datetime TEXT,
            symbol TEXT,
            timeframe TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL
            )""")

conn.commit()
conn.close()
