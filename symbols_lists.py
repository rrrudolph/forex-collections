

# Stocks, and some indices have been omitted
fx_symbols = [
    'EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', 
    'EUR/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD', 
    'EUR/GBP', 'EUR/JPY', 'GBP/JPY', 'CHF/JPY', 
    'GBP/CHF', 'EUR/AUD', 'EUR/CAD', 'AUD/CAD', 
    'AUD/JPY', 'CAD/JPY', 'NZD/JPY', 'GBP/CAD', 
    'GBP/NZD', 'GBP/AUD', 'AUD/NZD', 'USD/SEK', 
    'EUR/SEK', 'EUR/NOK', 'USD/NOK', 'USD/MXN', 
    'AUD/CHF', 'EUR/NZD', 'USD/ZAR', 'USD/HKD', 
    'ZAR/JPY', 'USD/TRY', 'EUR/TRY', 'NZD/CHF', 
    'CAD/CHF', 'NZD/CAD', 'TRY/JPY', 'USD/ILS', 
    'USD/CNH', 'AUS200', 'ESP35', 'FRA40', 
    'GER30', 'HKG33', 'JPN225', 'NAS100', 
    'SPX500', 'UK100', 'US30', 'Copper', 
    'USDOLLAR', 'US2000', 'USOil', 'UKOil', 
    'SOYF', 'NGAS', 'WHEATF', 'CORNF', 'XAU/USD', 'XAG/USD',
    'BTC/USD', 'BCH/USD', 'ETH/USD', 'LTC/USD',
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

spreads = [
    ['Copper, XAU/USD'],
    
]

fx_timeframes = {
    '1': 'm1',
    '5': 'm5',
    '15': 'm15',
    '60': 'H1',
    'D': 'D1'
}


# This is to handle finnhub 'num_candles' request automatically
seconds_per_candle = {
    '1': 60,
    '5': 300,
    '15': 900,
    '60': 3600,
    'D': 86400
}


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


bonds_table_nums = {
    'Australia': 2, 
    'Canada': 10, 
    'Germany': 19, 
    'Japan': 29, 
    'New Zealand': 39, 
    'Switzerland': 58, 
    'U.K.': 64, 
    'U.S.': 65
}

# {
#     'USD': [
#         'https://www.investing.com/rates-bonds/u.s.-2-year-bond-yield', 
#         'https://www.investing.com/rates-bonds/u.s.-10-year-bond-yield'
#         ],
        
#     'EUR': [
#         'https://www.investing.com/rates-bonds/germany-2-year-bond-yield', 
#         'https://www.investing.com/rates-bonds/germany-10-year-bond-yield'
#         ],

#     'GBP': [
#         'https://www.investing.com/rates-bonds/uk-2-year-bond-yield', 
#         'https://www.investing.com/rates-bonds/uk-10-year-bond-yield'
#         ],

#     'CAD': [
#         'https://www.investing.com/rates-bonds/canada-2-year-bond-yield', 
#         'https://www.investing.com/rates-bonds/canada-10-year-bond-yield'
#         ],

#     'AUD': [
#         'https://www.investing.com/rates-bonds/australia-2-year-bond-yield', 
#         'https://www.investing.com/rates-bonds/australia-10-year-bond-yield'
#         ], 

#     'JPY': [
#         'https://www.investing.com/rates-bonds/japan-2-year-bond-yield',  
#         'https://www.investing.com/rates-bonds/japan-10-year-bond-yield'
#         ], 

#     'NZD': [
#         'https://www.investing.com/rates-bonds/new-zealand-1-year', # yield
#         'https://www.investing.com/rates-bonds/new-zealand-10-years-bond-yield'
#         ], 

#     'CHF': [
#         'https://www.investing.com/rates-bonds/switzerland-2-year-bond-yield',
#         'https://www.investing.com/rates-bonds/switzerland-10-year-bond-yield'
#         ]
# }

