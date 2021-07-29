import numpy as np
import pandas as pd
import yfinance

class YF_loader():

    def __init__(self, tickers, start_date=pd.to_datetime('today').normalize()-pd.Timedelta('7day'), end_date=pd.to_datetime('today').normalize(), adjust = True, threads = True):

        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.adjust = adjust
        
        self.prices = self._download_pricing_data(threads)
        
    def _download_pricing_data(self, threads):

        tickers_string = " ".join(self.tickers)

        df = yfinance.download(tickers_string, start = self.start_date, end = self.end_date, 
                               auto_adjust = self.adjust)
        return df

    def get_ticker(self, ticker):
        if ticker not in self.tickers.values:
            raise Exception("Please enter valid ticker.")

        return self.prices[ticker]
    
