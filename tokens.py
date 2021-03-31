import telegram
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Telegram bot
bot = telegram.Bot(token='1371103620:AAG_6iRGzmeGDTd0V-W2gPTIavI-gVSPolA')


# Finnhub token
fin_token = 'budmt1v48v6ped914310'


# Google Sheets API
SCOPE = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
CREDS = ServiceAccountCredentials.from_json_keyfile_name(r'C:\Users\ru\forex\gsheets_token.json', SCOPE)
CLIENT = gspread.authorize(CREDS)

#DASHBOARD sheet(0)
forecast_sheet = CLIENT.open('data').get_worksheet(1) 
ohlc_sheet  = CLIENT.open('data').get_worksheet(2) 
ff_cal_sheet = CLIENT.open('data').get_worksheet(4)
bonds_sheet = CLIENT.open('bonds').get_worksheet(1)


# Paths to different sqlite database files
indexes_db = r'C:\Users\ru\forex\db\indexes.db'
econ_db = r'C:\Users\ru\forex\db\economic_data.db'
correlation_db = r'C:\Users\ru\forex\db\correlation.db'
laptop_db = r'\\192.168.1.4\Documents\ohlc.db'


# FXCM token
fx_token = '9a018ed0b2a87bf2bfa22ed47b745e13715567fe'


# IC Markets MT5 account info
mt5_login = 50341259
mt5_pass = "ZhPcw6MG"
mt5_server = "ICMarkets-Demo"
