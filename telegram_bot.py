from functools import cached_property
from typing import List
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from telegram import Update, ForceReply
from symbols_lists import indexes
from pathlib import Path
import pandas as pd
import numpy as np
import json
from indexes import _resample
from datetime import datetime
from derive_value_from_correlations import _add_ending_nans_to_df
from ohlc_request import mt5_ohlc_request
from tokens import INDEXES_PATH, econ_db, bot
import mplfinance as mpf
import sqlite3

ECON_CON = sqlite3.connect(econ_db)

def send_monthly_data():
    ''' Send the long term data as a nice chart :s '''

    econ_data = pd.read_sql(f'''SELECT * FROM forecast''', ECON_CON) 
    econ_data.datetime = pd.to_datetime(econ_data.datetime)
    econ_data = econ_data.set_index('datetime', drop=True).replace('', np.nan)
    
    # I need to get the sum of the most recent values from each event. I can
    # filter for where long_term.notna(), but my rolling window is unknown 
    # because not all currencies have the same number of long term events.
    
    data = {}
    for ccy in econ_data.ccy.unique():
        temp = econ_data[
            (econ_data.ccy == ccy) 
            & 
            (econ_data.long_term.notna())
        ]

        # Get the window sized based on number of unique values found
        roll_window = len(temp.ccy_event.unique())
        final_data = temp.long_term.rolling(roll_window).sum() / roll_window

        # The data needs to get aligned to whatever timeframe of ohlc it will be plotted on
        # Since there is sometimes multiple releases on a given day, group by day
        final_data.index = final_data.index.date
        data[ccy] = final_data.groupby([final_data.index]).sum()
        # data[ccy].index = pd.to_datetime(data[ccy].index)
        # data[ccy] = data[ccy].resample('1W')
        # my daily candles from IC Markets have the hours at 16:00 so add that lest you want funkydata

    # These long term data items were really only applicable to the currencies
    # with lots of unique events (E,U,G). 
    timeframe = 'D1'
    symbols = ['EURUSD', 'GBPUSD', 'EURGBP', 'USDCHF', 'NZDUSD', 'USDCAD', 'USDJPY']
    for symbol in symbols:
        df = mt5_ohlc_request(symbol, timeframe, num_candles=270)
        df.index = df.index.date
        df['long_term1'] = data[symbol[:3]]
        df['long_term2'] = data[symbol[3:]]
        df.long_term1 = df.long_term1.fillna(method='ffill')
        df.long_term2 = df.long_term2.fillna(method='ffill')
        df.index = pd.to_datetime(df.index)

        plots = []
        plots.append(mpf.make_addplot(df.long_term1, color='r', panel=1, width=2, secondary_y=True))
        plots.append(mpf.make_addplot(df.long_term2, color='b', panel=1, width=1, secondary_y=True))
        mpf.plot(df, type='candle', 
                tight_layout=True, 
                addplot=plots,
                show_nontrading=False, 
                volume=True,
                title=f'{symbol} {timeframe}',
                # style='classic',   ## black n white
                savefig=f'{symbol}_longterm.png'
                )
    
        # Send that pohoto
        # await bot.send_photo(chat_id=446051969, photo=open(Path(r'C:\Users\ru'+ f'\{symbol}_longterm.png'), 'rb'))

def send_images(text: List):
    ''' Send pics to telegram '''

    if len(text) > 1:
        period = text[1]
    else:
        period = '8H'

    for index in indexes:
        df = pd.read_parquet(Path(INDEXES_PATH, f'{index}_M5.parquet'))
        df = _resample(df, period).dropna()
        df.index = pd.to_datetime(df.index)
        # _add_ending_nans_to_df(df, 10)

        # Resampling to daily timeframe creates a Sunday candle and I want to combine that into Monday
        # Monday == 0, sunday == 6
        if period == '1D':
            row = df.index[1] - df.index[0]
            temp = df[df.index.weekday == 0]
            for i in temp.index:
                try:
                    df.loc[i, 'open'] = df.loc[i - 1 * row, 'open']
                    df.loc[i, 'low'] = min(df.loc[i, 'low'], df.loc[i - 1 * row, 'low'])
                    df.loc[i, 'high'] = max(df.loc[i, 'high'], df.loc[i - 1 * row, 'high'])
                    df.loc[i, 'volume'] += df.loc[i - 1 * row, 'volume']
                except:
                    pass
            # Now drop Sundays
            df = df[~(df.index.weekday == 6)]
                
        plots = []
        mpf.plot(df[-220:], type='candle', tight_layout=True, 
                addplot=plots,
                show_nontrading=False, volume=True,
                title=index + ' ' + period,
                # style='classic',   ## black n white
                savefig=f'{index}_{period}.png'
                )

        bot.send_photo(chat_id=446051969, photo=open(Path(r'C:\Users\ru'+ f'\{index}_{period}.png'), 'rb'))

def set_trade_direction(text: List):
    ''' Read directional trade commands and save to a file. If theres currenty
    an active direction then update the end time before creating a new line. '''

    opposites = {
        'buy': 'sell',
        'sell': 'buy'
    }

    # Maybe doesn't exist
    try:
        trades = pd.read_csv('trade_directions.csv')
    except:
        trades = pd.DataFrame(columns=['currency', 'direction', 'start', 'end'])
    
    trades.end = pd.to_datetime(trades.end)

    # Parse the list that was passed
    currency = text[0].upper()
    direction = text[1]
    start = datetime.now()

    # I may send "aud long 2 day" in which case 2 days will be the active trade period
    if len(text) > 2:
        active_period = f'{text[2]} {text[3]}'
        end = start + pd.Timedelta(active_period)
    # Otherwise default to 1 day
    else:
        end = start + pd.Timedelta('1 day')

    # See if a current opposing direction is active on that currency. If it is set its end to now()
    temp = trades[
        (trades.currency == currency)
        &
        (trades.end > datetime.now())
        &
        (trades.direction == opposites[direction])
        ]
        
    if not temp.empty:
        print('not empty')
        trades.loc[temp.index[-1], 'end'] = start
    
    # Add the new info to the dataframe
    if trades.empty:
        idx = 0
    else:
        idx = trades.index[-1] + 1
    trades.loc[idx, 'currency'] = currency
    trades.loc[idx, 'direction'] = direction
    trades.loc[idx, 'start'] = start
    trades.loc[idx, 'end'] = end

    # Save
    trades.to_csv('trade_directions.csv', index=False)

def stop_trading(text: List):
    ''' If 'stop aud' is received then I want to update the last end datetime for
    aud in the trade_directions file '''

    trades = pd.read_csv('trade_directions.csv')
    trades.end = pd.to_datetime(trades.end)

    # Change the last 'end' value (only if its still an active trade period, otherwise just ignore)
    currency = text[1].upper()
    date = trades[trades.currency == currency].tail(1)
    if pd.to_datetime(date.end.values[0]) > datetime.now():
        trades.loc[date.index, 'end'] = datetime.now()
    
    trades.to_csv('trade_directions.csv', index=False)
    #2021-10-03 20:52:21.594697

def send_active_currencies(update: Update):
    ''' List out which currencies and directions are currently active '''

    trades = pd.read_csv('trade_directions.csv')
    trades.end = pd.to_datetime(trades.end)

    active = trades[datetime.now() < trades.end]
    print(active)
    print('\n')
    for i in active.index:
        update.message.reply_text(active.loc[i, 'currency'] + ' ' + active.loc[i, 'direction'])

def parser(update: Update, context: CallbackContext) -> None:
    ''' This is the top level handler to parse the message and perform an action '''

    text = str(update.message.text).lower().split()
    print(text)

    # Check for a index picture update ("pics" / "pics 15min") 
    if text[0] == 'pics':
        send_images(text) 
    
    # Activate trading on certain currencies ("gbp buy")
    if text[0].upper() in indexes:
        set_trade_direction(text) 

    # Stop trading on certain currencies ("stop cad")
    if text[0] == 'stop':
        stop_trading(text)

    # Send current trading currencies and directions ("active")
    if text[0] == 'active':
        send_active_currencies(update)

    return

def main() -> None:
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater("1777249819:AAGiRaYqHVTCwYZMGkjEv6guQ1g3NN9LOGo")

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    # dispatcher.add_handler(CommandHandler("start", start))
    
    # on non command i.e message - echo the message on Telegram
    # dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, echo))
    dispatcher.add_handler(MessageHandler(Filters.text, parser))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()