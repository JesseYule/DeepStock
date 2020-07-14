import pandas as pd
import pandas_datareader.data as web
import datetime

code = "selected_industry_code.xlsx"
sheet = pd.read_excel(io=code, converters={u'A股代码': str})

for i in range(947, 1000):
    stockcode = str(sheet['A股代码'][i])+".SZ"

    start = datetime.datetime(2015, 7, 7)
    end = datetime.date.today()

    stock = web.DataReader(stockcode, "yahoo", start, end)

    if stock.shape[0] == 1216:
        filename = stockcode+".csv"
        stock.to_csv(filename)
        print(stockcode + " saved successfully")
    else:
        print(stockcode + " fail")

