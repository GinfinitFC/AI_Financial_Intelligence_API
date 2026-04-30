# app/services/stock_service.py

import yfinance as yf

def get_stock_history(ticker: str, period: str):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    return df.reset_index().to_dict(orient="records")