import tushare
import pandas 
import datetime
import datetime

token = '82153d897f40882279899c07feabcc257c359fb21232ff4952896688'
tushare.set_token(token)
pro = tushare.pro_api()
ticketRawData = pro.daily(ts_code='000001.SZ',start_data='20221001',end_date='20221117')
print(ticketRawData)
tickets =  ticketRawData.index.tolist()
dateToday = datetime.datetime.today().strftime('%Y%m%d')
file = './ticketListCN_'+dateToday+'.csv'
ticketRawData.to_csv(file)
print('Ticket saved.')

