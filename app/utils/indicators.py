# app/utils/indicators.py

import pandas as pd


def calculate_sma(series, window: int):

    return series.rolling(window=window).mean()


def detect_golden_cross(df):

    if len(df) < 200:
        return False

    previous = df.iloc[-2]
    current = df.iloc[-1]

    return bool(
        previous["SMA50"] <= previous["SMA200"]
        and current["SMA50"] > current["SMA200"]
    )


def detect_death_cross(df):

    if len(df) < 200:
        return False

    previous = df.iloc[-2]
    current = df.iloc[-1]

    return bool(
        previous["SMA50"] >= previous["SMA200"]
        and current["SMA50"] < current["SMA200"]
    )