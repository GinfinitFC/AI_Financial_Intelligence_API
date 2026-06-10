# app/services/stock_service.py

import yfinance as yf
from app.utils.indicators import (
    calculate_sma,
    detect_golden_cross,
    detect_death_cross
)

def get_stock_history(ticker: str, period: str = "1y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    return df.reset_index().to_dict(orient="records")

def get_stock_signal(ticker: str, period: str = "1y"):

    stock = yf.Ticker(ticker)

    df = stock.history(period=period)

    #Calculate SMA50&200
    df["SMA50"] = calculate_sma(df["Close"], 50)

    df["SMA200"] = calculate_sma(df["Close"], 200)

    latest = df.iloc[-1]

    #Enrich signal response
    trend = "neutral"

    if latest["Close"] > latest["SMA50"] > latest["SMA200"]:
        trend = "bullish"

    elif latest["Close"] < latest["SMA50"] < latest["SMA200"]:
        trend = "bearish"

    price_vs_sma50 = (
        "above"
        if latest["Close"] > latest["SMA50"]
        else "below"
    )

    price_vs_sma200 = (
        "above"
        if latest["Close"] > latest["SMA200"]
        else "below"
    )
        
    return {
        "ticker": ticker,
        "current_price": round(float(latest["Close"]), 2),
        "sma50": round(float(latest["SMA50"]), 2),
        "sma200": round(float(latest["SMA200"]), 2),
        "price_vs_sma50": price_vs_sma50,
        "price_vs_sma200": price_vs_sma200,
        "trend": trend,
        "golden_cross": bool(detect_golden_cross(df)),
        "death_cross": bool(detect_death_cross(df))
    }