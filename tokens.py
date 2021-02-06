import fxcmpy
import socketio
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# FXCM token
fx_token = '9a018ed0b2a87bf2bfa22ed47b745e13715567fe'

# Finnhub token
fin_token = 'budmt1v48v6ped914310'

# FXCM connection
con = fxcmpy.fxcmpy(
    access_token=fx_token, 
    log_level='error', 
    server='demo', 
    log_file='log.txt'
)

# Google Sheets API
SCOPE = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
CREDS = ServiceAccountCredentials.from_json_keyfile_name('/home/r/forex/gsheets_token.json', SCOPE)
CLIENT = gspread.authorize(CREDS)

bonds_gsheet = CLIENT.open('data').get_worksheet(1)
ff_cal_sheet = CLIENT.open('data').sheet1