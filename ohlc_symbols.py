import MetaTrader5 as mt5


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

te_ccys = {
    'united-states' : 'USD',
    'euro-area': 'EUR',
    'united-kingdom': 'GBP',
    'japan': 'JPY',
    'canada': 'CAD',
    'australia': 'AUD',
    'new-zealand': 'NZD',
    'switzerland': 'CHF'
}

# This is to handle finnhub 'num_candles' request automatically
seconds_per_candle = {'1': 60,
                      '5': 300,
                     '15': 900,
                     '30': 1800,
                     '60': 3600,
                      'D': 86400
                     }
