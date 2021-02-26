import sqlite3
from symbols_lists import fx_symbols, fin_symbols

# This is the home location for directory paths. 
ohlc_db = r'C:\Users\ru\forex\db\ohlc.db'
econ_db = r'C:\Users\ru\forex\db\economic_data.db'


def setup_conn(path):

    conn = sqlite3.connect(path)
    c = conn.cursor()
    return conn, c


def make_db_tables(fx_symbols, fin_symbols):

    conn, c = setup_conn(path)

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

    # Create trading economics data table
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

    # # Create the raw ohlcv tables
    # formatting isn't allowing in sql statements
    for symbol in zip(fx_symbols, fin_symbols):
        c.execute(f"""CREATE TABLE {symbol} (
                        datetime TEXT,
                        timeframe TEXT,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume REAL
                        )""")


    # Apparently a context manager with sqlite only handles database transactions 
    # and not the actual connection, so would need to call close() regardless.
    conn.commit()
    conn.close()

# if __name__ == "__main__":
#     make_db_tables(fx_symbols, fin_symbols)