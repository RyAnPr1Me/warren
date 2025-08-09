from __future__ import annotations
import pandas as pd
import numpy as np

BASIC_FEATURES = [
    "ret_1d","ret_5d","ret_10d","vol_10d","rsi_14"
]

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["close"].pct_change()
    out["ret_5d"] = out["close"].pct_change(5)
    out["ret_10d"] = out["close"].pct_change(10)
    out["vol_10d"] = out["close"].pct_change().rolling(10).std()
    # Simple RSI
    delta = out["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    out["rsi_14"] = 100 - (100 / (1 + rs))
    return out
