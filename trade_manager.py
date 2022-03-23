from datetime import datetime
from pathlib import Path
import pandas as pd
import json
import MetaTrader5 as mt5
import time
from symbols_lists import etf_to_htf, basket_pairs, tf_to_sec
from tokens import mt5_login, mt5_server, mt5_pass
from pair_data import build_symbol_data_dict
from trade_scanner import trade_scanner
from tokens import bot


''' My typical order entry system is to enter on the breakout of my signal bar.
Since Im tracking signals on index charts that method will probably be too slow 
to then enter market orders on all the pairs. Instead I will take the index signal
and just put in stop orders on each pair. It won't result in exact order timings across
but it should be good enough. And I could also decrease the size of the last order to
fill potentially. '''

p = r'C:\Users\ru\forex'

if not mt5.initialize(login=mt5_login, server=mt5_server,password=mt5_pass):
    print("initialize() failed, error code =",mt5.last_error())
    quit()

# # # Test a partial close
request = {
    "symbol": 'BTCUSD.a',
    "action": mt5.TRADE_ACTION_PENDING,
    # 'position_by': 165137336,
    'volume': 0.1,
    "deviation": 30,
    'price': mt5.symbol_info('BTCUSD.a').ask + 1.0,
    'tp': 0.0,
    'sl': 0.0,
    "type": mt5.ORDER_TYPE_BUY_STOP,
    "type_time": mt5.ORDER_TIME_GTC,
    # "type": mt5.ORDER_TYPE_CLOSE_BY,
    "type_filling": mt5.ORDER_FILLING_IOC,
}
                                                                                                     
# send a trading request
# result = mt5.order_send(request)
# if result.retcode != mt5.TRADE_RETCODE_DONE:
#     print(" retcode={}".format(result.retcode))
# quit()
# orders = mt5.positions_get()
# for order in orders:
#     if order.symbol == 'BTCUSD.a':
#         print(order)
#         print('\n')
# quit()


# entry_signals = {}
# '''{'USD': 
#         {'M5': 'buy'/'sell'}
#     ...
#     }'''


def get_best_and_worst_pairs_of_baskets(positions) -> dict:
    ''' Returns a dictionary with keys in the format 'EUR_M5'
    which map to dicts called 'best' and 'worst'. Those dicts
    represent the most profitable and least profitable trades
    within the EUR_M5 basket. they themselves map to dicts with
    keys 'symbol', 'ticket', 'time_update'.
    {'EUR_M5':
        {'best': 
            {'symbol': symbol,
             'ticket': ticket,
             'profit': profit,
             'volume': volume,
             'direction': buy/sell,
             'time_update': server time of last update,
            }
        ...
        }
    ...
    }  '''


    # First check what baskets are active
    # baskets are categorized by timeframe as well as currency
    baskets = {}
    for position in positions:
        comment = position.comment.split(' ')
        # A manual trade won't have a comment
        if len(comment) != 2:
            continue

        # map mt5s enums to words
        direction = 'buy' if position.type == 0 else 'sell'
        
        # Create a new list if this basket hasn't been seen
        if f'{ccy}_{tf}' not in baskets:
            baskets[f'{ccy}_{tf}'] = []
        
        baskets[f'{ccy}_{tf}'].append({ 'symbol': position.symbol, 
                                        'ticket': position.ticket, 
                                        'profit': position.profit,
                                        'volume': position.volume,
                                        'direction': direction,
                                        'time_update': position.time_update,
                                    }
        )

    # Now get the best and worst pairs in each basket
    # (remember "baskets" is a dictionary of lists, each item
    # in the list is a dict to store to store info above
    best_worst = {}
    for basket in baskets:

        # Ensure at least 2 trades open
        if len(basket) < 2:
            continue

        best_profit = 0
        worst_profit = 0
        for num, trade in enumerate(baskets[basket]):
            # init values on first run
            if num == 0:
                best_profit = trade['profit']
                worst_profit = trade['profit']
                index_of_best = num
                index_of_worst = num
            
            if trade['profit'] > best_profit:
                best_profit = trade['profit']
                index_of_best = num
            
            if trade['profit'] < worst_profit:
                worst_profit = trade['profit']
                index_of_worst = num

        # Save those best and worst trades
        best_worst[basket] = {
            # This returns the dict of position params
            'best': baskets[basket][index_of_best], 
            'worst': baskets[basket][index_of_worst],
        }

    return best_worst

def to_him_who_has_more_shall_be_given():
    ''' Track the profit on the different pairs in a basket. 
    Every n bars reallocate n % of the trade volume from the least
    profitable to the most profitable. '''

    positions = mt5.positions_get()
    if len(positions) == 0:
        return

    # How much to reallocate at a time
    close_percent = 0.1

    # How many bars to wait in between reallocation
    reallocation_cycle = 3

    baskets = get_best_and_worst_pairs_of_baskets(positions)
    if not baskets:
        return

    # Get current server time (I think this is actually more like "last update time"
    # of the passed symbol. so it won't change if the market is closed)
    server_time = mt5.symbol_info('EURUSD')._asdict()['time']

    for basket in baskets:
        
        worst_position = baskets[basket]['worst']
        best_position = baskets[basket]['best']
        # print('\n',baskets[basket],'\n')
        # print('\n',best_position,'\n')
        # iono how i wanna quantify one trade being n% more profitable...
        # how do i give meaning to the profitability of a trade...
        # maybe i just say f it and go strictly time based  
        # so like.. after 3 bars start moving funds
        # and move 5% every 3 bar or something. that 3% will become a smaller
        # and smaller number each time but i can solve that later if i want

        # See if 3 candles have closed since the last update to that trade
        time_passed = server_time - best_position['time_update']
        basket_timeframe = basket.split('_')[1]
        if time_passed >= tf_to_sec[basket_timeframe] * reallocation_cycle:

            # Reallocate the funds. So partial close on basket['worst'] and new entry on ['best']

            # 'volume' in dict is the current trade volume. im just goinng to override that value
            # with the amount i wanna close so I can pass that dict to close_trades() as is
            reallocation_lot_size = round(worst_position['volume'] * close_percent, 2)
            worst_position['volume'] = reallocation_lot_size
            close_trades([worst_position])

            symbol = best_position['symbol']
            direction = best_position['direction']
            enter_manual_trade(symbol, basket, basket_timeframe, reallocation_lot_size, direction, pair_data)            

def get_pairs_and_position_sizes(currency:str, timeframe:str, direction:str, data:dict) -> dict:
    ''' When a signal fires on an index (currency), get the list of permitted
    pairs to trade and their respective position sizes. For example, if I get a 
    "GBP buy" signal I can't trade any other currencies that I've set to "buy".
    The initial risk per basket will be 3% and it will split up across the pairs
    depending on the quality of setup on each pair. Max risk per pair is 2%. 
    A default SL distance of 1.5 ATR is used on the pairs to determine lot size. '''

    risk = 0.005
    max_risk = 0.02
    sl_atr = 1.5
    equity = mt5.account_info().equity

    ### GET LIST OF SYMBOLS TO TRADE ###

    # Check the trade_directions file and delete any pairs whose direction isn't allowed
    trade_directions = pd.read_csv(Path(p, 'trade_directions.csv'))
    trade_directions.end = pd.to_datetime(trade_directions.end)
    
    current = trade_directions[
        (datetime.now() < trade_directions.end)
        &
        (trade_directions.direction == direction)
        &
        (trade_directions.currency != currency)
        ]
    
    # Any rows that get returned are pairs that can't be traded
    filtered_pairs = []
    not_tradable = current.currency.tolist()
    
    #  Grab the list of pairs associated with that currency
    pairs = basket_pairs[currency]
    for pair in pairs:
        # Assuming the base/counter currency is not found in not_tradable, add it to new list
        if pair[:3] not in not_tradable and pair[3:] not in not_tradable:
            filtered_pairs.append(pair)

    if not filtered_pairs:
        return {}

    ### SET UNIQUE POSITION SIZING ON EACH SYMBOL ###

    risk_per_pair = risk / len(filtered_pairs)

    lot_sizes = {}
    '''{'EURUSD': 1.07, 'GBPEUR': 0.89 ...}'''
    for pair in filtered_pairs:
        this_pairs_risk = risk_per_pair
        
        ## Decrease risk if...

        # print(this_pairs_risk)
        # Not in zone
        htf1 = etf_to_htf[timeframe][0]
        htf2 = etf_to_htf[timeframe][1]
        if data[pair]['in_zone'][htf1] is False:
            if data[pair]['in_zone'][htf2] is False:
                this_pairs_risk -= 0.1 * risk_per_pair

        # Not with HTF trend (put this into pair dict)
        if data[pair]['trend'][htf1] != direction:
            if data[pair]['trend'][htf2] != direction:
                this_pairs_risk -= 0.1 * risk_per_pair

        # Not currently at top of hour 
        # honestly iono about this one, i should get stats on it
        # to know how mucconsiderwtion it deserves
        minute = datetime.now().minute
        if 10 < minute < 50:
                this_pairs_risk -= 0.1 * risk_per_pair

        # Use that risk allocation with the default SL distance to determine lot size

        # Convert distance into number of points (ticks)
        atr_in_ticks = data[pair]['atr'][tf] * 10** data[pair]['digits']

        # print('atr',data[pair]['atr'][tf])
        # print('atr ticks', atr_in_ticks)
        distance = round(sl_atr * atr_in_ticks)
        # print(distance)

        # tick_value gives the value for a standard lot for 1 point (min_tick) increment
        loss_with_1_lot = distance * data[pair]['tick_value']
        # print('loss w one', loss_with_1_lot)
        this_pairs_risk = max_risk if this_pairs_risk > max_risk else this_pairs_risk
        # print('max?', this_pairs_risk)
        this_pairs_risk *= equity
        # print(this_pairs_risk)
        lot_size = round(this_pairs_risk / loss_with_1_lot, 2)

        lot_sizes[pair] = lot_size

    return lot_sizes

def enter_trades(symbol_and_sizes, pair_data, currency, timeframe, basket_direction, expire_after=3, suffix:str='.a') -> None:
    ''' Enter basket trades. expire_after is how many bars the order will live for '''

    opposite = {
        'buy': 'sell',
        'sell': 'buy'
    }

    # Get the entry parameters for each symbol in the dict
    for symbol, lot_size in symbol_and_sizes.items():
        
        # Reverse the trade direction if the currency is in ct pos
        direction = basket_direction if currency == symbol[:3] else opposite[basket_direction]
        
        print(symbol, direction,  lot_size, )

        atr = pair_data[symbol]['atr'][timeframe]
        info = mt5.symbol_info(symbol)


        # I dont trust pair_data to always have accurate prices so request them
        # Set the entry 1 ATR above current price
        if direction == 'buy':
            entry = info.ask + 0.5 * atr
            stop_loss = entry - 2 * atr
            take_profit = entry + 5 * atr
            type = mt5.ORDER_TYPE_BUY_STOP
        else:
            entry = info.bid - 0.5 * atr
            stop_loss = entry + 2 * atr
            take_profit = entry - 5 * atr
            type = mt5.ORDER_TYPE_SELL_STOP
        
        print('enter_trades() price:', float(round(entry, info.digits)))

        # Cancel orders after n bars if not filled
        expiry = info.time + (tf_to_sec[timeframe] * expire_after)
        
        if suffix:
            symbol = symbol.upper() + suffix
        request = {
            'action': mt5.TRADE_ACTION_PENDING,
            'symbol': symbol,
            'volume': float(round(lot_size, 2)), # int will error
            'price': float(round(entry, info.digits)),
            'sl': float(round(stop_loss, info.digits)),
            'tp': float(round(take_profit, info.digits)),
            'deviation': 10,
            'expiration': expiry,
            'type': type,
            'type_time': mt5.ORDER_TIME_SPECIFIED,
            'type_filling': mt5.ORDER_FILLING_IOC,
            # Comments are how I keep track of all needed info about a trade's basket
            'comment': f'{currency} {timeframe} {basket_direction}', 
        }
                                                                                                                        
        # send a trading request
        result = mt5.order_send(request)
        if result is not None:
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                bot.send_message(chat_id=446051969, text=f'Entry failed: {symbol} {timeframe} {direction} {result.retcode}')
                print(f'~~~ order entry error {symbol} {result.retcode}')
                print(request['expiration'])
        else:
            bot.send_message(chat_id=446051969, text=f'Entry failed: {symbol} {timeframe} {direction}. "result" is None.')
            print(f'~~~ order entry error {symbol} "result" is none')


def enter_manual_trade(symbol, basket, timeframe, volume, direction, pair_data) -> None:
    ''' This is used by to_him_who_has_more_shall_be_given() and enters
    a market order on a specific symbol'''

    # If '.a' or some other suffix got added to the symbol
    # remove that in order to access pair_data
    x = symbol.split('.')[0]
    digits = pair_data[x]['digits']
    atr = pair_data[x]['atr'][timeframe]

    if direction == 'buy':
        price = mt5.symbol_info(symbol).ask
        stop_loss = price - 1.5 * atr
        take_profit = price + 5 * atr
        type = mt5.ORDER_TYPE_BUY
    else:
        price = mt5.symbol_info(symbol).bid
        stop_loss = price + 1.5 * atr
        take_profit = price - 5 * atr
        type = mt5.ORDER_TYPE_SELL

    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': symbol,
        'volume': float(volume), # int will error
        'sl': round(stop_loss,  digits),
        'tp': round(take_profit,  digits),
        'deviation': 10,
        'type': type,
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
        'comment': f'{basket} reallocation', 
    }
                                                                                                                        
    # send a trading request
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        bot.send_message(chat_id=446051969, text=f'Manunal entry failed: {symbol} {timeframe} {direction}')
        print(f'manual entry failed on {symbol} rc: {result.retcode}')

def close_any_opposite_basket_trades(currency, new_direction) -> None: 
    ''' If a GBP Buy signal gets generated, I need to close any sells
    on those associated pairs before I open the buy. This will parse
    the open trades and close them '''

    for pos in mt5.positions_get():
        comment = pos.comment.split(' ')
        ccy = comment[0]
        # tf = comment[1]
        direction = 'buy' if pos.type == 0 else 'sell'

        if ccy == currency:

            # The direction of a trade won't match the baskets direction
            # if the currency is the counter currency
            if currency != pos.symbol[:3]:
                direction = 'sell' if direction == 'buy' else 'buy'

            if direction != new_direction:
                close_trades([{'ticket': pos.ticket}])

def close_trades(trades:list) -> None:
    ''' Pass a list of dicts. Dicts must have 'ticket' as key 
    and can optionally have 'volume' if you want a partial close. '''

    for trade in trades:
        # Get that position info 
        pos = mt5.positions_get(ticket=trade['ticket'])[0]
        
        if pos.type == 1:
            _type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(pos.symbol).ask
        else:
            _type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(pos.symbol).bid


        # If volume is passed for a partial close I'll want to maintain
        # the existing comment
        comment = pos.comment if 'volume' in trade else None 
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'position': pos.ticket,
            'symbol': pos.symbol,
            'volume': trade.get('volume', pos.volume),
            'type': _type,
            'price': price,
            'deviation': 10,
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
            'comment': comment
        }

        # Sometimes "result" comes back as None
        count = 0
        while count < 5:
            result = mt5.order_send(request)
            count += 1
            if result is not None:
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    bot.send_message(chat_id=446051969, text=f'Close trade failed: {pos.symbol} {direction} {result.retcode}')
                    print(f"trade_manager.py: close_trades(): Error closing {trade['ticket']}. Retcode {result.retcode}")
            else:
                bot.send_message(chat_id=446051969, text=f'Close trade failed: "result" is None. {pos.symbol} {direction}')
                print('trade_manager.py: close_trades(): "result" is None.')

entry_signals = {}
'''
{'USD': 
        {'M5': 'buy'/'sell'}
    ...
    }
'''

if __name__ == '__main__':
    ''' this thing doesn't check open orders so could potentially
    submit limitless trades'''
    
    while True:

        if datetime.now().second != 0:
            continue

        to_him_who_has_more_shall_be_given()
        # s = time.time()
        entry_signals = trade_scanner()
        # print('scanning for trades duration:', time.time() - s)
        if not entry_signals:
            continue

        # entry_signals = {}
        # entry_signals['USD'] = {}
        # entry_signals['USD']['M5'] = 'buy'

        symbol_data = build_symbol_data_dict(entry_signals)

        # Parse the entry_signals and open/close appropriate trades
        for ccy in entry_signals:
            for tf in entry_signals[ccy]:

                direction = entry_signals[ccy][tf]
                if direction:
                    # New basket trade signal
                    print('Trade Manager:', str(datetime.now()).split('.')[0], ccy, tf, direction)
                    close_any_opposite_basket_trades(ccy, direction)
                    symbols_and_sizes = get_pairs_and_position_sizes(ccy, tf, direction, symbol_data)
                    enter_trades(symbols_and_sizes, symbol_data, ccy, tf, direction)
                    bot.send_message(chat_id=446051969, text=f'{ccy} {tf} {direction}')

        # print('time', time.time() - s)



        
# I'd intended to use get_least_profitable_pairs_in_basket() to close
# those trades when an index chart hits its TP, but its much simpler to
# not continue tracking index data after the trades have been entered

def get_least_profitable_pairs_in_basket(currency, timeframe) -> list:
    ''' Get least profitable pairs within a basket.  The data
    returned will be a list of dicts which can be iterated 
    through to close those trades. Keys are symbol ticket profit. 
    If all trades in the basket are negative the whole basket is returned.'''

    trades = []
    for position in mt5.positions_get():
        comment = position.comment.split(' ')
        ccy = comment[0]
        tf = comment[1]

        if ccy == currency and tf == timeframe:
            trades.append({ 'symbol': position.symbol, 
                            'ticket': position.ticket, 
                            'profit': position.profit
                            }
                        )
    
    # Now drop the most profitable trade from the list 
    profit = 0
    for num, trade in enumerate(trades):
        if trade['profit'] > profit:
            profit = trade['profit']
            i = num
    if profit > 0:
        trades.pop(i)

    return trades
