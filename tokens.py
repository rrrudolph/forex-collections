# import fxcmpy
import telegram
# import socketio
import gspread
from oauth2client.service_account import ServiceAccountCredentials



# Telegram bot
bot = telegram.Bot(token='1371103620:AAG_6iRGzmeGDTd0V-W2gPTIavI-gVSPolA')

# FXCM token
fx_token = '9a018ed0b2a87bf2bfa22ed47b745e13715567fe'

# Finnhub token
fin_token = 'budmt1v48v6ped914310'

# Google Sheets API
SCOPE = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
CREDS = ServiceAccountCredentials.from_json_keyfile_name(r'C:\Users\ru\forex\gsheets_token.json', SCOPE)
CLIENT = gspread.authorize(CREDS)

#DASHBOARD sheet(0)
forecast_sheet = CLIENT.open('data').get_worksheet(1) # 3rd sheet
ohlc_sheet  = CLIENT.open('data').get_worksheet(2) 
ff_cal_sheet = CLIENT.open('data').get_worksheet(3)
bonds_gsheet = CLIENT.open('data').get_worksheet(4)
adr_sheet = CLIENT.open('data').get_worksheet(3) #


# FXCM connection
# fxcm_con = fxcmpy.fxcmpy(
#         access_token=fx_token, 
#         log_level='error', 
#         server='demo', 
#         log_file='log.txt'
#     )