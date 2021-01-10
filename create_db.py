import sqlite3

# This is the home location for path.  
# Other modules will access this one.
path = r'C:\Users\Rudy\Desktop\codez\forex.db'

def setup_conn(path):

    conn = sqlite3.connect(path)
    c = conn.cursor()
    return conn, c


def make_db_tables(conn, c):

    ''' THESE ARE THE RAW DATA TABLES '''

    # Create the FF calendar table
    c.execute("""CREATE TABLE ff_cal_raw (
                date TEXT,
                time TEXT,
                ccy TEXT,
                event TEXT,
                actual TEXT,
                forecast TEXT,
                previous TEXT
                )""")

    # Create trading economics data table
    c.execute("""CREATE TABLE te_data_raw (
                country TEXT,
                date TEXT,
                category TEXT,
                name TEXT,
                actual TEXT,
                previous TEXT,
                range TEXT,
                frequency TEXT
                )""")

    # # Create the MT5/Finnhub forex OHLCV table (will only use a raw table)
    c.execute("""CREATE TABLE ohlc_raw (
                datetime TEXT,
                symbol TEXT,
                timeframe TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
                )""")


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

    # Apparently a context manager with sqlite only handles database transactions 
    # and not the actual connection, so would need to call close() regardless.
    conn.commit()
    conn.close()

# make_db_tables(conn, c)