from __future__ import annotations
from datetime import datetime
from typing import Optional
import pandas as pd
import yfinance as yf


def fetch_prices(symbol: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval, auto_adjust=False)
    if hist.empty:
        raise ValueError(f"No data returned for {symbol}")
    hist.index = pd.to_datetime(hist.index)
    hist = hist.rename(columns={c: c.lower() for c in hist.columns})
    return hist[["open", "high", "low", "close", "volume"]]
