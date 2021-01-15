import pandas as pd
from create_db import setup_conn, path  # get the connection to the database
from ohlc_symbols import te_countries

conn, c = setup_conn(path)
# conn = setup_conn(path)[0]

def request_te_data(country):
    ''' Open each country's indicators web page '''

    # This returns a list of df's
    data = pd.read_html(country)

        # These are the returned dataframes that I'm interested in
    desired_data = [
        'GDP',
        'Labour',
        'Prices',
        'Money',
        'Trade',
        'Government',
        'Taxes',
        'Business',
        'Consumer',
        'Housing',
        'Health'
    ]

    # Only save a dataframe who's column[0]'s title is in desired_data
    remove = []      
    for idx, df in enumerate(data):
        if df.columns[0] not in desired_data:
           remove.append(idx)

    # Delete from the upper indices to lower, otherwise the order gets messed up
    remove.reverse()
    for df in remove:
        del data[df]

    # Convert the the web link into the country name
    country_name = country.split('/')[3]
 
    # Prepare data for combination into single df
    for num, df in enumerate(data):
        df['category'] = df.columns[0]
        df = df.rename(columns = {
                                df.columns[0]: 'event', 
                                'Reference': 'date'
                                })
        df = df.rename(str.lower, axis='columns')
        data[num] = df

    # Combine the list of dfs into a single df        
    te_df = pd.concat([df for df in data])

    # Ensure all data is SQLite friendly
    te_df['country'] = country_name
    te_df['date'] = te_df['date'].astype(str)
    te_df['category'] = te_df['category'].astype(str)
    te_df['event'] = te_df['event'].astype(str)
    te_df['last'] = te_df['last'].astype(str)
    te_df['previous'] = te_df['previous'].astype(str)
    te_df['range'] = te_df['range'].astype(str)
    te_df['frequency'] = te_df['frequency'].astype(str)

    te_df = te_df.dropna(axis=1)
    te_df = te_df.reset_index(drop=True)

    return te_df

def save_te_data_to_db(dfs, conn, c):

    # Create a single df out of the list of dfs that were passed in
    df = pd.concat([df for df in dfs])

    for i in df.index:
        params = (  
                    df.loc[i, 'country'], 
                    df.loc[i, 'date'], 
                    df.loc[i, 'category'], 
                    df.loc[i, 'event'], 
                    df.loc[i, 'last'], 
                    df.loc[i, 'previous'], 
                    df.loc[i, 'range'], 
                    df.loc[i, 'frequency']
                )

        c.execute("INSERT INTO te_data_raw VALUES (?,?,?,?,?,?,?,?)", params)
    conn.commit()


def run(conn, c, countries):
    
    te_dfs = [request_te_data(country) for country in countries]
    save_te_data_to_db(te_dfs, conn, c)

run(conn, c, te_countries)