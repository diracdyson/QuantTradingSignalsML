import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

class GARCHSignal():
    def __init__(self,interval = '1m',periodi= 1 ):
        # consider the top 10 holdings of the SPY
        # include the SPY for portfolio diversity tech growth
        self.period =periodi
        period = str(self.period)+'d'
        self.targetcol = '1DayRet_'+'Close'
        
        self.tickers = ['SPY']
        self.feat = ['Open','Low','High','Close','Volume']
        self.X = yf.download(tickers =self.tickers,period = period, interval = interval)[self.feat]

        #self.X.index = pd.DatetimeIndex(self.X.index).to_period('D')
        #self.X= self.X.dropna()
        print(self.X.shape)

        up = self.X[self.X.Close >= self.X.Open]
        down = self.X[self.X.Close < self.X.Open]


        col1 = 'green'
        col2 = 'red'

       # width = .4
      #  width2 = .05

#plot up prices
        plt.bar(up.index,up.Close-up.Open,width,bottom=up.Open,color=col1)
        plt.bar(up.index,up.High-up.Close,width2,bottom=up.Close,color=col1)
        plt.bar(up.index,up.Low-up.Open,width2,bottom=up.Open,color=col1)

#plot down prices
        plt.bar(down.index,down.Close-down.Open,width,bottom=down.Open,color=col2)
        plt.bar(down.index,down.High-down.Open,width2,bottom=down.Open,color=col2)
        plt.bar(down.index,down.Low-down.Close,width2,bottom=down.Close,color=col2)

#rotate x-axis tick labels
       # plt.xticks(rotation=45, ha='right')

#display candlestick chart
        plt.show()





g = GARCHSignal()

        