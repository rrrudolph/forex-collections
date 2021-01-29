

# This is the entire list of fxcm symbols, saved directly from stdout
fx_symbols = ['EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', 'EUR/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP', 'EUR/JPY', 'GBP/JPY', 'CHF/JPY', 'GBP/CHF', 'EUR/AUD', 'EUR/CAD', 'AUD/CAD', 'AUD/JPY', 'CAD/JPY', 'NZD/JPY', 'GBP/CAD', 'GBP/NZD', 'GBP/AUD', 'AUD/NZD', 'USD/SEK', 'EUR/SEK', 'EUR/NOK', 'USD/NOK', 'USD/MXN', 'AUD/CHF', 'EUR/NZD', 'USD/ZAR', 'USD/HKD', 'ZAR/JPY', 'USD/TRY', 'EUR/TRY', 'NZD/CHF', 'CAD/CHF', 'NZD/CAD', 'TRY/JPY', 'USD/ILS', 'USD/CNH', 'AUS200', 'ESP35', 'FRA40', 'GER30', 'HKG33', 'JPN225', 'NAS100', 'SPX500', 'UK100', 'US30', 'Copper', 'CHN50', 'EUSTX50', 'VOLX', 'USDOLLAR', 'US2000', 'INDIA50', 'USOil', 'UKOil', 'SOYF', 'NGAS', 'USOilSpot', 'UKOilSpot', 'WHEATF', 'CORNF', 'Bund', 'XAU/USD', 'XAG/USD', 'EMBasket', 'JPYBasket', 'BTC/USD', 'BCH/USD', 'ETH/USD', 'LTC/USD', 'XRP/USD', 'CryptoMajor', 'EOS/USD', 'XLM/USD', 'ESPORTS', 'BIOTECH', 'CANNABIS', 'FAANG', 'CHN.TECH', 'CHN.ECOMM', 'USEquities', 'AIRLINES', 'CASINOS', 'TRAVEL', 'US.ECOMM', 'US.BANKS', 'US.AUTO', 'WFH', 'BA.us', 'BAC.us', 'DIS.us', 'F.us', 'JPM.us', 'PFE.us', 'T.us', 'XOM.us', 'AAPL.us', 'AMZN.us', 'BIDU.us', 'FB.us', 'GOOG.us', 'INTC.us', 'MSFT.us', 'ACA.fr', 'AI.fr', 'BNP.fr', 'AIR.fr', 'FP.fr', 'ORA.fr', 'MC.fr', 'RNO.fr', 'SAN.fr', 'ADS.de', 'ALV.de', 'BAYN.de', 'BMW.de', 'DAI.de', 'DBK.de', 'DPW.de', 'DTE.de', 'LHA.de', 'SAP.de', 'AZN.uk', 'BARC.uk', 'BATS.uk', 'BP.uk', 'GSK.uk', 'HSBA.uk', 'TSCO.uk', 'BABA.us', 'NFLX.us', 'TSLA.us', 'GLEN.uk', 'RDSB.uk', 'BYND.us', 'UBER.us', 'ZM.us', 'JD.us', 'PDD.us', 'TME.us', 'WB.us', 'BILI.us', 'NVDA.us']


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


fx_timeframes = [
    'm1',
    'm5',
    'm15',
    'H1',
    'D1',
]


# This is to handle finnhub 'num_candles' request automatically
seconds_per_candle = {'1': 60,
                      '5': 300,
                     '15': 900,
                     '30': 1800,
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

