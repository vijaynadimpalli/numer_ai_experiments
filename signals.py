import numerapi
import numpy as np
import pandas as pd
from data_loader import YF_loader
from transformer import rsi 


public_id = 'C55HDTWFAMEWJJRPO5LE37ORLXJ3YJ4S'
secret_key = 'G362FO4TJ4IRTIALFQK3TLEYOVU7HMT462R6EGJ5SUHTH4C3IWRXNZIQUNO2JRTZ'

napi = numerapi.SignalsAPI(public_id=public_id, secret_key=secret_key)


def get_elgible_tickers():
    """
    Gets eligible tickers for submission
    """
    # read in list of active Signals tickers which can change slightly era to era
    eligible_tickers = pd.Series(napi.ticker_universe(), name='bloomberg_ticker')
    print(f"Number of eligible tickers: {len(eligible_tickers)}")

    # read in yahoo to bloomberg ticker map, still a work in progress, h/t wsouza
    ticker_map = pd.read_csv(
        'https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv'
    )
    print(f"Number of tickers in map: {len(ticker_map)}")

    # map eligible numerai tickers to yahoo finance tickers
    yfinance_tickers = eligible_tickers.map(
        dict(zip(ticker_map['bloomberg_ticker'], ticker_map['yahoo']))).dropna()
    bloomberg_tickers = ticker_map['bloomberg_ticker']
    print(f'Number of eligible, mapped tickers: {len(yfinance_tickers)}')
    return yfinance_tickers


def modelling(n_stocks = 100, duration_days=15):

    eligible_tickers = get_elgible_tickers()[:n_stocks]
    end_date = pd.to_datetime('today').normalize()
    start_date = end_date - duration_days * pd.Timedelta('1d')
    yf_loader = YF_loader(tickers=eligible_tickers,start_date=start_date, end_date=end_date)

    close_prices = yf_loader.prices['Close']
    rsi_value  = close_prices.apply(lambda x: rsi(x)).dropna()

    lagged_model(rsi_value)

    return rsi_value


def lagged_model(data, n_lags=5):

    tickers = data.columns

    lags_data = [data] * n_lags
    for ind in range(n_lags):
        lags_data[ind] = data.shift(ind+1).rename({x:f'{x}_lag{ind+1}' for x in data.columns},axis=1)

    data = pd.concat([data] + lags_data,axis=1).sort_index(axis=1).dropna()

    data.columns = pd.MultiIndex.from_product([tickers, ['lag_0', 'lag_1', 'lag_2','lag_3', 'lag_4', 'lag_5']])
    print(data)


def make_submission(dataframe):
    dataframe.to_csv('./output.csv')
    model_id = 'f6e62245-f89e-4b94-9855-dd8741e4b60a'
    submission = napi.upload_predictions('./output.csv', model_id=model_id)
    print(submission)

