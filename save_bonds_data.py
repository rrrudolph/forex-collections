import pandas as pd
import sqlite




OHLC_CON = sqlite3.connect(ohlc_db)


def _read_bonds(timeframe, sheet=bonds_sheet):
    ''' get the bond data and normalize for plotting with ohlc index data '''
    
    df = pd.DataFrame(sheet.get_all_values())

    # Scrub n Clean
    df = df.set_index(df.iloc[:, 0])
    df.columns = df.iloc[0]
    df = df.iloc[1:, 1:]
    df.index = pd.to_datetime(df.index)
    df = df.replace('', np.nan)
    df = df.fillna(method='ffill')
    df = df.astype(float)

    # Now append each columns data to the correct table
    # (first time requires a try block because tables wont exist)
    bonds_tables = [
        'US10Y',
        'US02Y',
        'DE10Y',
        'DE02Y',
        'GB10Y',
        'GB02Y',
        'JP10Y',
        'JP02Y',
        'AU10Y',
        'AU02Y',
        'NZ10Y',
        'NZ02Y',
        'CA10Y',
        'CA02Y',
        'CH10Y',
        'CH02Y',
    ]

    for table in bonds_tables:
        df = df[df.col =df
        try:
            last_row = pd.read_sql(f'SELECT datetime FROM {table} ORDER BY datetime DESC LIMIT 1', OHLC_CON)
        except:
            # Save all the 
