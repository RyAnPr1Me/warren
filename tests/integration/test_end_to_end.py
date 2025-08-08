from pathlib import Path
from src.data.ingestion.yfinance_fetcher import fetch_prices
from src.features.generator import compute_features

def test_pipeline_basic():
    df = fetch_prices("AAPL", period="1y")
    feat = compute_features(df)
    assert not feat.empty
    assert "ret_1d" in feat.columns
