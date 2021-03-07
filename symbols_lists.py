
mt5_symbols = {
    'majors': [
        'EURUSD', 'GBPUSD', 'USDCHF', 'USDJPY', 'USDCAD', 
        'AUDUSD', 'AUDNZD', 'AUDCAD', 'AUDCHF', 'AUDJPY', 
        'CHFJPY', 'EURGBP', 'EURAUD', 'EURCHF', 'EURJPY', 
        'EURNZD', 'EURCAD', 'GBPCHF', 'GBPJPY', 'CADCHF', 
        'CADJPY', 'GBPAUD', 'GBPCAD', 'GBPNZD', 'NZDCAD', 
        'NZDCHF', 'NZDJPY', 'NZDUSD',
    ],
    'others': [
    'USDSGD', 'EURPLN', 
    'EURSEK', 'EURTRY', 'EURZAR', 'GBPSEK', 'USDCNH', 
    'USDHUF', 'USDMXN', 'USDNOK', 'USDPLN', 'USDRUB', 
    'USDSEK', 'USDTHB', 'USDTRY', 'USDZAR', 'DE30', 
    'UK100', 'US30', 'US500', 'USTEC',
    'XAGUSD', 'XAUUSD', 'XPDUSD', 'XPTUSD', 
    'XNGUSD', 'XTIUSD', 'XBRUSD', 'BTCUSD', 'BCHUSD', 
    'DSHUSD', 'ETHUSD', 'LTCUSD',
    'MOO.NYSE', 'XLF.NYSE', 'XLU.NYSE', 'XLP.NYSE',
    'XLE.NYSE', # ENERGY
    'XLI.NYSE', # INDUSTRIAL
    'XOP.NYSE', # OIL EXPORATION
    'VYM.NYSE', # HIGH YIELD BONDS
    'LQD.NYSE', # INVESTMENT GRADE CORPORATE BOND
    'TLT.NYSE', # 20 YR TREASURY
    'DBA.NYSE', # AGRICULTURE
    'EEM.NYSE', # EMERGING MARKETS
    'IEMG.NYSE', # EMERGING MARKETS
    'BAC.NYSE',
    'CBRE.NYSE', # REAL ESTATE

    ]
}

fin_symbols = [
    # 'MOO', 
    'HYG', 
    'VIXM', 
    'VIXY', 
    # 'XLF', 
    # 'XLU', 
    'XLY', 
    # 'XLP', 
    'IWF', 
    'IWD', 
    # 'BAC', 
    'REET',
    'OANDA:XCU_USD',
]

spreads = [
    'XTIUSD_XAUUSD',
    'XTIUSD_XAGUSD',
    'XTIUSD_XBRUSD',
    'XAGUSD_XAUUSD',
    'XPDUSD_XAGUSD',
    'XNGUSD_XAUUSD',
    'XCUUSD_XAUUSD',
    'XNGUSD_XTIUSD',
    'UK100_US30',
    'UK100_US500'
    'UK100_USTEC',
    'DE30_US30',
    'DE30_US500',
    'DE30_USTEC',
    'US30_US500',
    'US30_USTEC',
    'US500_USTEC',
    'XLF.NYSE_XLU.NYSE',
    'VYM.NYSE_LQD.NYSE',
    'VYM.NYSE_TLT.NYSE',
    'DBA.NYSE_MOO.NYSE',
    'BAC.NYSE_CBRE.NYSE',
    'XLE.NYSE_XLI.NYSE',
    'XOP.NYSE_LQD.NYSE',
    'EEM.NYSE_TLT.NYSE',
    'MOO.NYSE_MOO.NYSE',
    'XLP.NYSE_XOP.NYSE',
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

