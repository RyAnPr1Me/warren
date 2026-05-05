"""
Advanced Stock Data Feature Engineering Module (v2)

Ultra-comprehensive feature engineering pipeline for AI model training:
  • Parallel multi-symbol downloading (ThreadPoolExecutor) with per-symbol disk cache
  • 180+ technical indicators: standard + Ichimoku Cloud, VWAP, Donchian Channels,
    Hull MA, KAMA, TEMA, DEMA, Williams %R, MFI, CMF, Fisher Transform, Elder Ray,
    Aroon, Ultimate Oscillator, Parabolic SAR, and more
  • Multi-timeframe features (daily + weekly + monthly resampled back to daily)
  • Market-context & cross-asset features (SPY/QQQ/IWM correlations, VIX level,
    sector ETF rolling correlations)
  • Fundamental data snapshot (P/E, P/B, market cap, beta, dividend yield)
  • Multi-horizon prediction targets (next 1/5/10/21-day returns + direction)
  • Regime detection features (trend / mean-reversion, high / low volatility)
  • Hurst exponent for long-memory characterisation
"""

import os
import pickle
import logging
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import talib
from sklearn.preprocessing import MinMaxScaler, RobustScaler

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
PREDICTION_HORIZONS: List[int] = [1, 5, 10, 21]  # trading-day forecast horizons

SECTOR_MAP: Dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "META": "Technology", "NVDA": "Technology", "INTC": "Technology",
    "AMD":  "Technology", "ORCL": "Technology", "CSCO": "Technology",
    "IBM":  "Technology", "ADBE": "Technology", "CRM":  "Technology",
    "PYPL": "Technology", "QCOM": "Technology", "AVGO": "Technology",
    "TXN":  "Technology", "MU":   "Technology",
    "AMZN": "Consumer_Discretionary", "NFLX": "Consumer_Discretionary",
    "HD":   "Consumer_Discretionary", "NKE":  "Consumer_Discretionary",
    "SBUX": "Consumer_Discretionary", "MCD":  "Consumer_Discretionary",
    "LOW":  "Consumer_Discretionary", "TGT":  "Consumer_Discretionary",
    "ROST": "Consumer_Discretionary", "TJX":  "Consumer_Discretionary",
    "YUM":  "Consumer_Discretionary", "MAR":  "Consumer_Discretionary",
    "CMG":  "Consumer_Discretionary", "DIS":  "Consumer_Discretionary",
    "TSLA": "Consumer_Discretionary",
    "JPM": "Finance", "BAC": "Finance", "GS":  "Finance", "WFC": "Finance",
    "C":   "Finance", "MS":  "Finance", "BLK": "Finance", "AXP": "Finance",
    "V":   "Finance", "MA":  "Finance", "PNC": "Finance", "SCHW": "Finance",
    "CME": "Finance", "CB":  "Finance", "MMC": "Finance", "TFC": "Finance",
    "USB": "Finance", "ALL": "Finance", "AIG": "Finance", "BK":  "Finance",
    "JNJ":  "Healthcare", "PFE":  "Healthcare", "MRK":  "Healthcare",
    "ABBV": "Healthcare", "LLY":  "Healthcare", "ABT":  "Healthcare",
    "UNH":  "Healthcare", "TMO":  "Healthcare", "DHR":  "Healthcare",
    "BMY":  "Healthcare", "AMGN": "Healthcare", "MDT":  "Healthcare",
    "ISRG": "Healthcare", "GILD": "Healthcare", "CVS":  "Healthcare",
    "VRTX": "Healthcare", "ZTS":  "Healthcare", "REGN": "Healthcare",
    "HUM":  "Healthcare", "BIIB": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "EOG": "Energy",
    "PSX": "Energy", "PXD": "Energy", "VLO": "Energy", "SLB": "Energy",
    "MPC": "Energy", "OXY": "Energy",
    "PG":   "Consumer_Staples", "KO":   "Consumer_Staples",
    "PEP":  "Consumer_Staples", "WMT":  "Consumer_Staples",
    "COST": "Consumer_Staples", "MDLZ": "Consumer_Staples",
    "CL":   "Consumer_Staples", "EL":   "Consumer_Staples",
    "GE":   "Industrials", "HON":  "Industrials", "MMM":  "Industrials",
    "CAT":  "Industrials", "DE":   "Industrials", "BA":   "Industrials",
    "LMT":  "Industrials", "RTX":  "Industrials", "UPS":  "Industrials",
    "FDX":  "Industrials", "UNP":  "Industrials", "CSX":  "Industrials",
    "ETN":  "Industrials", "EMR":  "Industrials", "ITW":  "Industrials",
    "PH":   "Industrials", "GD":   "Industrials", "NSC":  "Industrials",
    "CARR": "Industrials", "PCAR": "Industrials",
    "T":     "Communication", "VZ":    "Communication",
    "TMUS":  "Communication", "CMCSA": "Communication",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D":   "Utilities", "AEP": "Utilities",
    "AMT":  "Real_Estate", "PLD":  "Real_Estate", "CCI":  "Real_Estate",
    "SPG":  "Real_Estate", "EQIX": "Real_Estate",
}

SECTOR_ETF_MAP: Dict[str, str] = {
    "Technology":             "XLK",
    "Finance":                "XLF",
    "Healthcare":             "XLV",
    "Energy":                 "XLE",
    "Consumer_Staples":       "XLP",
    "Consumer_Discretionary": "XLY",
    "Industrials":            "XLI",
    "Materials":              "XLB",
    "Utilities":              "XLU",
    "Real_Estate":            "XLRE",
    "Communication":          "XLC",
}

BROAD_MARKET_SYMBOLS: List[str] = ["SPY", "QQQ", "IWM"]
VIX_SYMBOL: str = "^VIX"

DEFAULT_SYMBOLS: List[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "JPM",  "BAC",  "GS",   "WFC",   "C",
    "JNJ",  "PFE",  "MRK",  "ABBV",  "UNH",
    "XOM",  "CVX",  "COP",  "SLB",   "EOG",
    "PG",   "KO",   "PEP",  "WMT",   "COST",
    "HD",   "NKE",  "SBUX", "MCD",   "NVDA",
]


# ══════════════════════════════════════════════════════════════════════════════
# Low-level helpers
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return a clean single-level OHLCV DataFrame for *symbol*.

    Handles MultiIndex columns returned by newer yfinance versions and drops
    duplicate columns, keeping the first occurrence.
    """
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        last_level = df.columns.get_level_values(-1)
        if symbol in last_level:
            try:
                df = df.xs(symbol, axis=1, level=-1)
            except Exception:
                df.columns = ["_".join(str(x) for x in col if x) for col in df.columns]
        else:
            df.columns = ["_".join(str(x) for x in col if x) for col in df.columns]
    try:
        idx = pd.Index(df.columns)
        if idx.duplicated().any():
            df = df.loc[:, ~idx.duplicated()]
    except Exception:
        pass
    df = df.rename(columns={"Adj Close": "Adj_Close", "adjclose": "Adj_Close"})
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns for {symbol}: {missing}")
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _cache_path(cache_dir: str, key: str) -> str:
    """Return full path for a pickle cache file."""
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{key}.pkl")


def _load_cache(cache_dir: Optional[str], key: str):
    """Return cached object or None."""
    if not cache_dir:
        return None
    path = _cache_path(cache_dir, key)
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass
    return None


def _save_cache(cache_dir: Optional[str], key: str, obj) -> None:
    """Persist *obj* to pickle cache."""
    if not cache_dir:
        return
    try:
        with open(_cache_path(cache_dir, key), "wb") as fh:
            pickle.dump(obj, fh)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Advanced indicator helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """Estimate Hurst exponent via simplified R/S analysis.

    Parameters
    ----------
    series : pd.Series
        Time series of returns (or prices) to analyse.
    max_lag : int
        Maximum lag length used in the R/S calculation.
        Larger values give a more stable estimate but require more data.

    Returns
    -------
    float
        Hurst exponent in [0, 1]:
        H ≈ 0.5  → random walk
        H > 0.5  → trending / long memory
        H < 0.5  → mean-reverting
    """
    ts = np.array(series.dropna())
    if len(ts) < max_lag * 2:
        return 0.5
    lags = range(2, max_lag)
    tau_list, rs_list = [], []
    for lag in lags:
        chunks = [ts[i: i + lag] for i in range(0, len(ts) - lag, lag)]
        rs_vals = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean = np.mean(chunk)
            deviate = np.cumsum(chunk - mean)
            r = deviate.max() - deviate.min()
            s = np.std(chunk, ddof=0)
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            tau_list.append(lag)
            rs_list.append(np.mean(rs_vals))
    if len(tau_list) < 2:
        return 0.5
    poly = np.polyfit(np.log(tau_list), np.log(rs_list), 1)
    return float(np.clip(poly[0], 0.0, 1.0))


def _rolling_hurst(series: pd.Series, window: int = 63, max_lag: int = 15) -> pd.Series:
    """Rolling Hurst exponent."""
    return series.rolling(window).apply(
        lambda x: _compute_hurst_exponent(pd.Series(x), max_lag=max_lag),
        raw=False,
    )


def _compute_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Add Ichimoku Cloud columns."""
    high, low, close = df["High"], df["Low"], df["Close"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun  = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou = close.shift(26)   # 26 periods in the past (look-back, not look-ahead)
    df["Ichimoku_Tenkan"]  = tenkan
    df["Ichimoku_Kijun"]   = kijun
    df["Ichimoku_SpanA"]   = span_a
    df["Ichimoku_SpanB"]   = span_b
    df["Ichimoku_Chikou"]  = chikou
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
    df["Above_Cloud"]  = (close > cloud_top).astype(int)
    df["Cloud_Width"]  = (span_a - span_b).abs()
    df["TK_Cross"]     = np.where(tenkan > kijun, 1, -1)
    return df


def _compute_vwap(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add rolling VWAP and price-to-VWAP deviation."""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"]
    df["VWAP"]          = (typical * vol).rolling(window).sum() / vol.rolling(window).sum()
    df["Price_to_VWAP"] = df["Close"] / df["VWAP"] - 1
    return df


def _compute_donchian(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add Donchian Channel features."""
    hi = df["High"].rolling(window).max()
    lo = df["Low"].rolling(window).min()
    span = (hi - lo).replace(0, np.nan)
    df[f"Donchian_High_{window}"] = hi
    df[f"Donchian_Low_{window}"]  = lo
    df[f"Donchian_Mid_{window}"]  = (hi + lo) / 2
    df[f"Donchian_Pos_{window}"]  = (df["Close"] - lo) / span
    return df


def _hull_ma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average = WMA( 2·WMA(n/2) − WMA(n), √n )."""
    half   = max(2, period // 2)
    sqrt_n = max(2, int(np.sqrt(period)))
    wma_h  = series.ewm(span=half,   adjust=False).mean()
    wma_n  = series.ewm(span=period, adjust=False).mean()
    raw    = 2 * wma_h - wma_n
    return raw.ewm(span=sqrt_n, adjust=False).mean()


def _compute_chaikin_mf(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    hl  = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl
    mfv = mfm * df["Volume"]
    return mfv.rolling(window).sum() / df["Volume"].rolling(window).sum()


def _compute_elder_ray(df: pd.DataFrame, period: int = 13) -> Tuple[pd.Series, pd.Series]:
    """Elder Ray: Bull Power = High − EMA, Bear Power = Low − EMA."""
    ema = pd.Series(talib.EMA(df["Close"].values.astype(float), timeperiod=period), index=df.index)
    return df["High"] - ema, df["Low"] - ema


def _compute_fisher_transform(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Fisher Transform of price position within its N-period range."""
    hi   = df["High"].rolling(period).max()
    lo   = df["Low"].rolling(period).min()
    span = (hi - lo).replace(0, np.nan)
    val  = (2 * (df["Close"] - lo) / span - 1).clip(-0.999, 0.999)
    return 0.5 * np.log((1 + val) / (1 - val))


def _parkinson_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Parkinson high-low volatility estimator (annualised)."""
    log_hl = np.log(df["High"] / df["Low"]) ** 2
    return np.sqrt(log_hl.rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252)


def _yang_zhang_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Yang-Zhang volatility estimator (annualised)."""
    k   = 0.34 / (1.34 + (window + 1) / (window - 1))
    log_ho = np.log(df["High"] / df["Open"])
    log_lo = np.log(df["Low"]  / df["Open"])
    log_co = np.log(df["Close"] / df["Open"])
    log_oc = np.log(df["Open"]  / df["Close"].shift(1))
    rs     = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    sigma_oc = log_oc.rolling(window).var()
    sigma_rs = rs.rolling(window).mean()
    sigma_co = log_co.rolling(window).var()
    yz = np.sqrt((sigma_oc + k * sigma_co + (1 - k) * sigma_rs).clip(0)) * np.sqrt(252)
    return yz


def _compute_multi_timeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge weekly and monthly resampled features back onto the daily index."""
    # ── Weekly ────────────────────────────────────────────────────────────────
    weekly = df[["Open", "High", "Low", "Close", "Volume"]].resample("W").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    weekly["W_Return"] = weekly["Close"].pct_change()
    weekly["W_Range"]  = (weekly["High"] - weekly["Low"]) / weekly["Open"].replace(0, np.nan)
    weekly["W_Vol"]    = weekly["W_Return"].rolling(4).std()
    weekly["W_RSI14"]  = pd.Series(
        talib.RSI(weekly["Close"].values.astype(float), timeperiod=14), index=weekly.index
    )
    weekly["W_MA4"]    = weekly["Close"].rolling(4).mean()
    weekly["W_MA12"]   = weekly["Close"].rolling(12).mean()
    weekly["W_Mom4"]   = weekly["Close"].pct_change(4)
    keep_w = ["W_Return", "W_Range", "W_Vol", "W_RSI14", "W_MA4", "W_MA12", "W_Mom4"]
    weekly_ff = weekly[keep_w].reindex(df.index, method="ffill")

    # ── Monthly ───────────────────────────────────────────────────────────────
    monthly = df[["Open", "High", "Low", "Close", "Volume"]].resample("ME").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    monthly["M_Return"] = monthly["Close"].pct_change()
    monthly["M_Range"]  = (monthly["High"] - monthly["Low"]) / monthly["Open"].replace(0, np.nan)
    monthly["M_Vol"]    = monthly["M_Return"].rolling(3).std()
    monthly["M_MA3"]    = monthly["Close"].rolling(3).mean()
    monthly["M_MA12"]   = monthly["Close"].rolling(12).mean()
    monthly["M_Mom3"]   = monthly["Close"].pct_change(3)
    keep_m = ["M_Return", "M_Range", "M_Vol", "M_MA3", "M_MA12", "M_Mom3"]
    monthly_ff = monthly[keep_m].reindex(df.index, method="ffill")

    return pd.concat([df, weekly_ff, monthly_ff], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Market-context data
# ══════════════════════════════════════════════════════════════════════════════

def get_market_context_data(
    start_date: str,
    end_date: str,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch broad-market & sector-ETF context data indexed by date.

    Returns daily columns: VIX level, SPY/QQQ/IWM returns, each sector-ETF
    return, and 20-day rolling volatility for each.
    """
    ctx_symbols = BROAD_MARKET_SYMBOLS + [VIX_SYMBOL] + list(SECTOR_ETF_MAP.values())
    series_dict: Dict[str, pd.Series] = {}
    for sym in ctx_symbols:
        cache_key = f"ctx_{sym.replace('^', '')}_{start_date}_{end_date}"
        raw = _load_cache(cache_dir, cache_key)
        if raw is None:
            try:
                raw = yf.download(
                    sym, start=start_date, end=end_date,
                    progress=False, auto_adjust=True,
                )
                _save_cache(cache_dir, cache_key, raw)
            except Exception as exc:
                logger.warning(f"Market context fetch failed for {sym}: {exc}")
                continue
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close_col = "Close" if "Close" in raw.columns else raw.columns[0]
        close = raw[close_col].squeeze()
        col = sym.replace("^", "")
        if sym == VIX_SYMBOL:
            series_dict["VIX"] = close.rename("VIX")
        else:
            ret = close.pct_change()
            series_dict[f"{col}_Ret"] = ret.rename(f"{col}_Ret")
            series_dict[f"{col}_Vol20"] = (
                ret.rolling(20).std() * np.sqrt(252)
            ).rename(f"{col}_Vol20")

    if not series_dict:
        logger.warning("No market context data could be fetched.")
        return pd.DataFrame()

    ctx = pd.concat(series_dict.values(), axis=1)
    ctx.index = pd.to_datetime(ctx.index)
    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# Per-symbol processing
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_raw(
    symbol: str,
    start_date: str,
    end_date: str,
    cache_dir: Optional[str],
) -> Optional[pd.DataFrame]:
    """Download (or load from cache) raw OHLCV data for *symbol*."""
    cache_key = f"{symbol}_{start_date}_{end_date}_raw"
    raw = _load_cache(cache_dir, cache_key)
    if raw is not None:
        return raw
    raw = yf.download(
        symbol, start=start_date, end=end_date,
        progress=False, auto_adjust=False,
        group_by="column", threads=True,
    )
    _save_cache(cache_dir, cache_key, raw)
    return raw


def _fetch_fundamentals(symbol: str, cache_dir: Optional[str]) -> Dict:
    """Return a dict of fundamental snapshot values from yfinance."""
    cache_key = f"{symbol}_fundamentals"
    cached = _load_cache(cache_dir, cache_key)
    if cached is not None:
        return cached
    try:
        info = yf.Ticker(symbol).info
        data = {
            "PE_Ratio":       info.get("trailingPE",     np.nan),
            "PB_Ratio":       info.get("priceToBook",    np.nan),
            "Market_Cap_B":   (info.get("marketCap", 0) or 0) / 1e9,
            "Beta":           info.get("beta",           np.nan),
            "Dividend_Yield": info.get("dividendYield",  0) or 0,
            "Profit_Margin":  info.get("profitMargins",  np.nan),
            "Revenue_Growth": info.get("revenueGrowth",  np.nan),
        }
        _save_cache(cache_dir, cache_key, data)
        return data
    except Exception as exc:
        logger.warning(f"Fundamentals fetch failed for {symbol}: {exc}")
        return {}


def _process_single_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    cache_dir: Optional[str],
    include_fundamentals: bool,
    mkt_ctx: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """Full feature-engineering pipeline for a single symbol.

    Returns a feature-rich DataFrame indexed by date, or None on failure.
    """
    cache_key = f"{symbol}_{start_date}_{end_date}_features"
    cached = _load_cache(cache_dir, cache_key)
    if cached is not None:
        logger.info(f"[{symbol}] Loaded {len(cached)} rows from cache.")
        return cached

    try:
        raw = _fetch_raw(symbol, start_date, end_date, cache_dir)
        df  = _normalize_ohlcv(raw, symbol)
    except Exception as exc:
        logger.error(f"[{symbol}] OHLCV fetch/normalize failed: {exc}")
        return None

    if df is None or len(df) < 200:
        logger.warning(f"[{symbol}] Insufficient data ({len(df) if df is not None else 0} rows), skipping.")
        return None

    df.index = pd.to_datetime(df.index)

    # ── Convenience aliases ──────────────────────────────────────────────────
    close  = df["Close"].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    open_  = df["Open"].astype(float)
    volume = df["Volume"].astype(float)

    df["Symbol"] = symbol

    # ── Core return / range features ─────────────────────────────────────────
    df["Return"]          = close.pct_change()
    df["Log_Return"]      = np.log(close / close.shift(1))
    df["Volume_Change"]   = volume.pct_change()
    df["Price_Range"]     = high - low
    df["Price_Range_Pct"] = (df["Price_Range"] / open_.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )

    # ── Multi-horizon targets ────────────────────────────────────────────────
    for h in PREDICTION_HORIZONS:
        fwd_ret = close.pct_change(h).shift(-h)
        df[f"Target_Return_{h}d"]    = fwd_ret
        df[f"Target_Direction_{h}d"] = (fwd_ret > 0).astype(int)
    # Keep legacy single-horizon names for backward compatibility
    df["Target_Next_Day_Return"]    = df["Target_Return_1d"]
    df["Target_Next_Day_Direction"] = df["Target_Direction_1d"]

    # ── Classic Moving Averages ───────────────────────────────────────────────
    for w in [5, 10, 20, 50, 100, 200]:
        df[f"MA_{w}"]         = talib.SMA(close.values, timeperiod=w)
        df[f"EMA_{w}"]        = talib.EMA(close.values, timeperiod=w)
        df[f"Close_to_MA_{w}"]  = close / pd.Series(talib.SMA(close.values, timeperiod=w), index=df.index) - 1
        df[f"MA_{w}_Slope"]   = pd.Series(talib.SMA(close.values, timeperiod=w), index=df.index).pct_change(5)

    # ── Advanced MAs ─────────────────────────────────────────────────────────
    df["DEMA_20"]   = talib.DEMA(close.values, timeperiod=20)
    df["TEMA_20"]   = talib.TEMA(close.values, timeperiod=20)
    df["KAMA_20"]   = talib.KAMA(close.values, timeperiod=20)
    df["HMA_20"]    = _hull_ma(close, 20)
    df["HMA_50"]    = _hull_ma(close, 50)
    df["WMA_20"]    = talib.WMA(close.values, timeperiod=20)

    # ── Momentum / Oscillators ───────────────────────────────────────────────
    macd, macd_sig, macd_hist = talib.MACD(close.values)
    df["MACD"]        = macd
    df["MACD_Signal"]  = macd_sig
    df["MACD_Hist"]    = macd_hist
    df["MACD_Cross"]   = np.where(macd > macd_sig, 1, -1)

    for w in [7, 14, 21]:
        df[f"RSI_{w}"] = talib.RSI(close.values, timeperiod=w)

    # Stochastic
    slowk, slowd = talib.STOCH(high.values, low.values, close.values)
    df["Stochastic_K"]    = slowk
    df["Stochastic_D"]    = slowd
    df["Stochastic_KD"]   = slowk - slowd

    # Fast Stochastic
    fastk, fastd = talib.STOCHF(high.values, low.values, close.values)
    df["StochF_K"] = fastk
    df["StochF_D"] = fastd

    # Stochastic RSI
    fastk_rsi, fastd_rsi = talib.STOCHRSI(close.values, timeperiod=14)
    df["StochRSI_K"] = fastk_rsi
    df["StochRSI_D"] = fastd_rsi

    df["CCI"]         = talib.CCI(high.values, low.values, close.values)
    df["ADX"]         = talib.ADX(high.values, low.values, close.values)
    df["PLUS_DI"]     = talib.PLUS_DI(high.values, low.values, close.values)
    df["MINUS_DI"]    = talib.MINUS_DI(high.values, low.values, close.values)
    df["DI_Diff"]     = df["PLUS_DI"] - df["MINUS_DI"]

    df["Williams_R"]  = talib.WILLR(high.values, low.values, close.values)
    df["MOM_10"]      = talib.MOM(close.values, timeperiod=10)
    df["ROC_10"]      = talib.ROC(close.values, timeperiod=10)
    df["ROC_20"]      = talib.ROC(close.values, timeperiod=20)
    df["TRIX_14"]     = talib.TRIX(close.values, timeperiod=14)
    df["DPO_20"]      = close - close.shift(11)  # Detrended Price Oscillator approx
    df["PPO"]         = talib.PPO(close.values)
    df["APO"]         = talib.APO(close.values)

    # Aroon
    aroon_dn, aroon_up = talib.AROON(high.values, low.values, timeperiod=14)
    df["Aroon_Up"]    = aroon_up
    df["Aroon_Down"]  = aroon_dn
    df["Aroon_Osc"]   = talib.AROONOSC(high.values, low.values, timeperiod=14)

    # Ultimate Oscillator
    df["ULTOSC"] = talib.ULTOSC(high.values, low.values, close.values)

    # Parabolic SAR
    df["SAR"] = talib.SAR(high.values, low.values)
    df["SAR_Signal"] = np.where(close > pd.Series(talib.SAR(high.values, low.values), index=df.index), 1, -1)

    # Fisher Transform
    df["Fisher_10"] = _compute_fisher_transform(df, period=10)
    df["Fisher_20"] = _compute_fisher_transform(df, period=20)

    # Elder Ray
    bull_p, bear_p = _compute_elder_ray(df, period=13)
    df["Elder_Bull_Power"] = bull_p
    df["Elder_Bear_Power"] = bear_p

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    for w in [5, 20]:
        upper, middle, lower = talib.BBANDS(close.values, timeperiod=w)
        upper  = pd.Series(upper,  index=df.index)
        middle = pd.Series(middle, index=df.index)
        lower  = pd.Series(lower,  index=df.index)
        df[f"BB_Upper_{w}"]    = upper
        df[f"BB_Middle_{w}"]   = middle
        df[f"BB_Lower_{w}"]    = lower
        df[f"BB_Width_{w}"]    = (upper - lower) / middle.replace(0, np.nan)
        df[f"BB_Position_{w}"] = (close - lower) / (upper - lower).replace(0, np.nan)

    # Keltner Channels (EMA ± 2*ATR)
    atr14 = pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=14), index=df.index)
    ema20 = pd.Series(talib.EMA(close.values, timeperiod=20), index=df.index)
    df["KC_Upper"] = ema20 + 2 * atr14
    df["KC_Lower"] = ema20 - 2 * atr14
    df["KC_Pos"]   = (close - df["KC_Lower"]) / (df["KC_Upper"] - df["KC_Lower"]).replace(0, np.nan)

    # ── Ichimoku Cloud ────────────────────────────────────────────────────────
    df = _compute_ichimoku(df)

    # ── VWAP ──────────────────────────────────────────────────────────────────
    df = _compute_vwap(df, window=20)

    # ── Donchian Channels ─────────────────────────────────────────────────────
    for w in [20, 55]:
        df = _compute_donchian(df, window=w)

    # ── Volume Indicators ─────────────────────────────────────────────────────
    df["OBV"]        = talib.OBV(close.values, volume.values)
    df["OBV_Change"] = pd.Series(talib.OBV(close.values, volume.values), index=df.index).pct_change()
    df["MFI_14"]     = talib.MFI(high.values, low.values, close.values, volume.values.astype(float), timeperiod=14)
    df["AD"]         = talib.AD(high.values, low.values, close.values, volume.values.astype(float))
    df["ADOSC"]      = talib.ADOSC(high.values, low.values, close.values, volume.values.astype(float))
    df["CMF_20"]     = _compute_chaikin_mf(df, window=20)

    vol_ratio = volume / volume.rolling(20).mean().replace(0, np.nan)
    df["Volume_Ratio_20"] = vol_ratio
    df["Volume_Surge"]    = (vol_ratio > 2.0).astype(int)

    # ── Volatility Indicators ─────────────────────────────────────────────────
    for w in [5, 10, 20, 30, 60]:
        df[f"Volatility_{w}"] = df["Log_Return"].rolling(w).std() * np.sqrt(252)

    df["ATR"]    = atr14
    df["ATR_21"] = talib.ATR(high.values, low.values, close.values, timeperiod=21)
    df["NATR"]   = talib.NATR(high.values, low.values, close.values)

    df["Parkinson_Vol_20"]   = _parkinson_volatility(df, window=20)
    df["YZ_Volatility_20"]   = _yang_zhang_volatility(df, window=20)
    df["Vol_Regime"]         = (df["Volatility_20"] > df["Volatility_20"].rolling(60).mean()).astype(int)

    # ── Trend indicators ──────────────────────────────────────────────────────
    df["Bull_Market"]      = (close > df["MA_200"]).astype(int)
    df["ADX_Trend"]        = (df["ADX"] > 25).astype(int)
    df["Market_Momentum"]  = (df["Return"].rolling(10).sum() > 0).astype(int)
    df["Price_52w_High"]   = close / high.rolling(252).max() - 1
    df["Price_52w_Low"]    = close / low.rolling(252).min() - 1

    # ── Gap features ─────────────────────────────────────────────────────────
    prev_close = close.shift(1)
    df["Gap_Up"]   = (open_ > prev_close).astype(int)
    df["Gap_Down"] = (open_ < prev_close).astype(int)
    df["Gap_Size"] = open_ / prev_close.replace(0, np.nan) - 1

    # ── Seasonality / calendar ────────────────────────────────────────────────
    df["Day_of_Week"] = df.index.dayofweek
    df["Month"]       = df.index.month
    df["Year"]        = df.index.year
    df["Quarter"]     = df.index.quarter
    df["Week_of_Year"] = df.index.isocalendar().week.astype(int)
    df["Is_Month_End"]   = df.index.is_month_end.astype(int)
    df["Is_Quarter_End"] = df.index.is_quarter_end.astype(int)
    for i in range(5):   # Mon-Fri
        df[f"Day_{i}"] = (df["Day_of_Week"] == i).astype(int)
    for i in range(1, 13):
        df[f"Month_{i}"] = (df["Month"] == i).astype(int)

    # ── Lagged return/volume features ────────────────────────────────────────
    for lag in [1, 2, 3, 5, 10, 21, 63]:
        df[f"Return_Lag_{lag}"]        = df["Return"].shift(lag)
        df[f"Log_Return_Lag_{lag}"]    = df["Log_Return"].shift(lag)
        df[f"Volume_Change_Lag_{lag}"] = df["Volume_Change"].shift(lag)

    # ── Rolling statistics ────────────────────────────────────────────────────
    for w in [5, 10, 21, 63]:
        df[f"Return_Mean_{w}"]  = df["Return"].rolling(w).mean()
        df[f"Return_Std_{w}"]   = df["Return"].rolling(w).std()
        df[f"Return_Min_{w}"]   = df["Return"].rolling(w).min()
        df[f"Return_Max_{w}"]   = df["Return"].rolling(w).max()
        df[f"Return_Skew_{w}"]  = df["Return"].rolling(w).skew()
        df[f"PV_Corr_{w}"]      = df["Return"].rolling(w).corr(df["Volume_Change"])

    # ── Z-score of price vs rolling mean ─────────────────────────────────────
    for w in [20, 60]:
        mu  = close.rolling(w).mean()
        sig = close.rolling(w).std().replace(0, np.nan)
        df[f"Price_ZScore_{w}"] = (close - mu) / sig

    # ── Regime detection: Hurst exponent ─────────────────────────────────────
    df["Hurst_63"]  = _rolling_hurst(df["Log_Return"], window=63,  max_lag=15)
    df["Hurst_126"] = _rolling_hurst(df["Log_Return"], window=126, max_lag=20)
    df["MeanRev_Regime"] = (df["Hurst_63"] < 0.45).astype(int)  # mean-reverting flag

    # ── Multi-timeframe features ──────────────────────────────────────────────
    df = _compute_multi_timeframe_features(df)

    # ── Sector encoding ───────────────────────────────────────────────────────
    sector = SECTOR_MAP.get(symbol, "Other")
    df["Sector"] = sector
    all_sectors = list(set(SECTOR_MAP.values())) + ["Other"]
    for s in all_sectors:
        df[f"Sector_{s}"] = int(sector == s)

    # ── Candlestick patterns ──────────────────────────────────────────────────
    pattern_funcs = [
        talib.CDLDOJI,        talib.CDLHAMMER,      talib.CDLENGULFING,
        talib.CDLMORNINGSTAR, talib.CDLEVENINGSTAR,  talib.CDLHARAMI,
        talib.CDLSHOOTINGSTAR, talib.CDLMARUBOZU,   talib.CDL3WHITESOLDIERS,
        talib.CDL3BLACKCROWS, talib.CDLDRAGONFLYDOJI, talib.CDLGRAVESTONEDOJI,
        talib.CDLSPINNINGTOP, talib.CDLHANGINGMAN,
    ]
    for fn in pattern_funcs:
        name = fn.__name__
        try:
            df[name] = fn(open_.values, high.values, low.values, close.values)
        except Exception:
            df[name] = 0

    # ── Market-context correlations ───────────────────────────────────────────
    if mkt_ctx is not None and not mkt_ctx.empty:
        # SPY rolling 20-day correlation
        spy_ret = mkt_ctx.get("SPY_Ret")
        if spy_ret is not None:
            spy_aligned = spy_ret.reindex(df.index)
            df["SPY_Corr_20"] = df["Return"].rolling(20).corr(spy_aligned)
            df["Beta_SPY_60"]  = (
                df["Return"].rolling(60).cov(spy_aligned)
                / spy_aligned.rolling(60).var().replace(0, np.nan)
            )
            df["Alpha_SPY_20"] = df["Return"] - df["Beta_SPY_60"] * spy_aligned

        # VIX
        vix = mkt_ctx.get("VIX")
        if vix is not None:
            vix_aligned = vix.reindex(df.index, method="ffill")
            df["VIX"]         = vix_aligned
            df["VIX_Change"]  = vix_aligned.pct_change()
            df["High_VIX"]    = (vix_aligned > 25).astype(int)
            df["Crisis_VIX"]  = (vix_aligned > 40).astype(int)

        # Sector ETF correlation
        etf_sym = SECTOR_ETF_MAP.get(sector)
        if etf_sym:
            etf_ret = mkt_ctx.get(f"{etf_sym}_Ret")
            if etf_ret is not None:
                etf_aligned = etf_ret.reindex(df.index)
                df["SectorETF_Corr_20"] = df["Return"].rolling(20).corr(etf_aligned)
                df["Excess_Ret_Sector"] = df["Return"] - etf_aligned

    # ── Fundamental features (static snapshot repeated across all rows) ───────
    if include_fundamentals:
        fundamentals = _fetch_fundamentals(symbol, cache_dir)
        for k, v in fundamentals.items():
            df[k] = float(v) if v is not None else np.nan

    # ── Drop rows dominated by NaN (from indicator warm-up or missing targets) ─
    # Include Target_Return_1d so the last row(s) with no valid forward return
    # (where (NaN > 0).astype(int) would silently produce a false 0 label) are
    # removed before training.
    df = df.dropna(subset=["MA_200", "RSI_14", "Ichimoku_Kijun", "Target_Return_1d"])

    _save_cache(cache_dir, cache_key, df)
    logger.info(f"[{symbol}] Processed {len(df)} rows with {len(df.columns)} features.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def get_feature_engineered_stock_data(
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_rows: int = 10_000,
    cache_dir: Optional[str] = None,
    n_jobs: int = 4,
    include_fundamentals: bool = False,
    include_market_context: bool = True,
    multi_horizon_targets: bool = True,
) -> pd.DataFrame:
    """Generate a comprehensive feature-engineered stock dataset for AI training.

    Parameters
    ----------
    symbols : list[str], optional
        Tickers to fetch. Defaults to ``DEFAULT_SYMBOLS`` (30 large-caps).
    start_date : str, optional
        ISO date string ``YYYY-MM-DD``. Defaults to 6 years ago.
    end_date : str, optional
        ISO date string ``YYYY-MM-DD``. Defaults to today.
    min_rows : int
        Warn if the combined dataset is smaller than this.
    cache_dir : str, optional
        Directory for per-symbol pickle caches. Pass ``None`` to disable.
    n_jobs : int
        Number of parallel worker threads for symbol downloading.
    include_fundamentals : bool
        If ``True``, attach yfinance fundamental snapshot columns.
    include_market_context : bool
        If ``True``, merge SPY/QQQ/VIX/sector-ETF context features.
    multi_horizon_targets : bool
        If ``True``, generate targets for all ``PREDICTION_HORIZONS``.
        Single-horizon legacy columns are always present.

    Returns
    -------
    pandas.DataFrame
        Feature-rich dataset ready for AI model training.
    """
    if symbols is None:
        symbols = DEFAULT_SYMBOLS
    # De-duplicate while preserving order
    seen: set = set()
    symbols = [s for s in symbols if not (s in seen or seen.add(s))]

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=6 * 365)).strftime("%Y-%m-%d")

    logger.info(
        f"Fetching {len(symbols)} symbols from {start_date} to {end_date} "
        f"(n_jobs={n_jobs}, cache={cache_dir!r})."
    )

    # ── Market context (fetched once, shared across workers) ─────────────────
    mkt_ctx: Optional[pd.DataFrame] = None
    if include_market_context:
        try:
            mkt_ctx = get_market_context_data(start_date, end_date, cache_dir)
            logger.info(f"Market context loaded: {list(mkt_ctx.columns[:6])} …")
        except Exception as exc:
            logger.warning(f"Could not load market context: {exc}")

    # ── Parallel symbol processing ────────────────────────────────────────────
    all_data: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = {
            pool.submit(
                _process_single_symbol,
                sym, start_date, end_date, cache_dir,
                include_fundamentals, mkt_ctx,
            ): sym
            for sym in symbols
        }
        for future in as_completed(futures):
            sym = futures[future]
            try:
                result = future.result()
                if result is not None and len(result) >= 100:
                    all_data.append(result)
            except Exception as exc:
                logger.error(f"[{sym}] Unhandled exception: {exc}")

    if not all_data:
        raise ValueError("No valid stock data was processed.")

    combined = pd.concat(all_data)
    combined = combined.reset_index()
    if "index" in combined.columns:
        combined.rename(columns={"index": "Date"}, inplace=True)
    if "Date" not in combined.columns and combined.index.name == "Date":
        combined = combined.reset_index()

    # Drop multi-horizon targets if not requested (keep legacy columns)
    if not multi_horizon_targets:
        drop_cols = [c for c in combined.columns
                     if c.startswith("Target_") and c not in
                     ("Target_Next_Day_Return", "Target_Next_Day_Direction")]
        combined.drop(columns=drop_cols, inplace=True, errors="ignore")

    n = len(combined)
    if n < min_rows:
        logger.warning(f"Generated {n} rows — fewer than requested {min_rows}.")
    else:
        logger.info(f"Generated {n} rows × {len(combined.columns)} features.")

    return combined


def save_data_to_csv(data: pd.DataFrame, file_path: str = "stock_data_features.csv") -> None:
    """Save the dataset to *file_path* in CSV format."""
    data.to_csv(file_path, index=False)
    logger.info(f"Data saved → {file_path}")


def normalize_features(
    data: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None,
    scaler_type: str = "minmax",
) -> pd.DataFrame:
    """Normalise numerical features.

    Parameters
    ----------
    data : DataFrame
    exclude_columns : list[str], optional
        Columns to leave as-is.  Defaults to date/symbol/target columns.
    scaler_type : {"minmax", "robust"}
        ``"minmax"`` → [0, 1];  ``"robust"`` → median/IQR scaling.
    """
    if exclude_columns is None:
        target_cols = [c for c in data.columns if c.startswith("Target_")]
        exclude_columns = (
            ["Date", "Symbol", "Sector"] + target_cols
        )
    normalized = data.copy()
    cols = [c for c in data.columns if c not in exclude_columns]
    numeric_cols = data[cols].select_dtypes(include=[np.number]).columns.tolist()

    scaler = RobustScaler() if scaler_type == "robust" else MinMaxScaler()
    # Replace inf/-inf with NaN before fitting; otherwise the scaler's percentile
    # computation is corrupted, producing NaN outputs for the entire column.
    data_to_scale = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
    normalized[numeric_cols] = scaler.fit_transform(data_to_scale)
    # Fill any remaining NaN with a neutral value that is meaningful for the
    # chosen scaler: 0.0 for RobustScaler (= median after transform) or 0.5 for
    # MinMaxScaler (= midrange of [0, 1]).
    nan_fill = 0.0 if scaler_type == "robust" else 0.5
    normalized[numeric_cols] = normalized[numeric_cols].fillna(nan_fill)
    return normalized


def normalize_train_test_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: str = "robust",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a scaler on the training set only, then transform both sets.

    This prevents data leakage: test-set statistics never influence the scaler
    used to normalise the training data.

    Parameters
    ----------
    X_train, X_test : DataFrame
        Feature matrices returned by :func:`split_train_test` or
        :func:`walk_forward_splits`.
    scaler_type : {"robust", "minmax"}
        Scaling method.
    """
    scaler = RobustScaler() if scaler_type == "robust" else MinMaxScaler()
    nan_fill = 0.0 if scaler_type == "robust" else 0.5

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    train_to_scale = X_train[numeric_cols].replace([np.inf, -np.inf], np.nan)
    scaler.fit(train_to_scale)

    X_train_norm = X_train.copy()
    X_test_norm  = X_test.copy()

    X_train_norm[numeric_cols] = pd.DataFrame(
        scaler.transform(train_to_scale),
        columns=numeric_cols, index=X_train.index,
    ).fillna(nan_fill)

    test_to_scale = X_test[numeric_cols].replace([np.inf, -np.inf], np.nan)
    X_test_norm[numeric_cols] = pd.DataFrame(
        scaler.transform(test_to_scale),
        columns=numeric_cols, index=X_test.index,
    ).fillna(nan_fill)

    return X_train_norm, X_test_norm


def split_train_test(
    data: pd.DataFrame,
    test_size: float = 0.2,
    time_based: bool = True,
    target_horizon: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train / test sets.

    Parameters
    ----------
    data : DataFrame
    test_size : float
        Fraction of data for test set.
    time_based : bool
        If ``True``, split chronologically (recommended for financial time series).
    target_horizon : int
        Which forecast horizon to use as *y* (1, 5, 10, or 21).
    """
    from sklearn.model_selection import train_test_split as _tts

    feature_cols = [
        c for c in data.columns
        if not c.startswith("Target_") and c not in ("Date", "Symbol", "Sector")
    ]
    numeric_features = data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    target_col = f"Target_Direction_{target_horizon}d"
    if target_col not in data.columns:
        target_col = "Target_Next_Day_Direction"

    if time_based:
        data = data.sort_values("Date")
        split_idx = int(len(data) * (1 - test_size))
        train, test = data.iloc[:split_idx], data.iloc[split_idx:]
        return (
            train[numeric_features],
            test[numeric_features],
            train[target_col],
            test[target_col],
        )
    else:
        X, y = data[numeric_features], data[target_col]
        return _tts(X, y, test_size=test_size, random_state=42)


def walk_forward_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    test_size: float = 0.1,
    target_horizon: int = 1,
):
    """Yield (X_train, X_test, y_train, y_test) for walk-forward CV.

    Each fold uses an expanding training window followed by a fixed test window,
    which is the correct approach for financial time-series models.
    """
    feature_cols = [
        c for c in data.columns
        if not c.startswith("Target_") and c not in ("Date", "Symbol", "Sector")
    ]
    numeric_features = data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    target_col = f"Target_Direction_{target_horizon}d"
    if target_col not in data.columns:
        target_col = "Target_Next_Day_Direction"

    data = data.sort_values("Date").reset_index(drop=True)
    n = len(data)
    fold_size = int(n * test_size)
    initial_train_end = n - n_splits * fold_size

    for fold in range(n_splits):
        train_end = initial_train_end + fold * fold_size
        test_end  = train_end + fold_size
        if train_end <= 0 or test_end > n:
            continue
        train = data.iloc[:train_end]
        test  = data.iloc[train_end:test_end]
        yield (
            train[numeric_features],
            test[numeric_features],
            train[target_col],
            test[target_col],
        )


def main():
    """Demonstrate usage: generate, save, normalise, and split the dataset."""
    try:
        stock_data = get_feature_engineered_stock_data(
            cache_dir=".cache",
            include_market_context=True,
        )
        save_data_to_csv(stock_data)
        normalized = normalize_features(stock_data, scaler_type="robust")
        X_train, X_test, y_train, y_test = split_train_test(normalized)

        print(f"\nDataset: {len(stock_data):,} rows × {len(stock_data.columns)} features")
        print(f"Train:   {len(X_train):,} samples")
        print(f"Test:    {len(X_test):,} samples")
        print(f"\nSample features (first 25 alphabetically):")
        for col in sorted(stock_data.columns)[:25]:
            print(f"  • {col}")
    except Exception as exc:
        logger.error(f"main() failed: {exc}")
        raise


if __name__ == "__main__":
    main()
