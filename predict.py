"""
Warren — Real-Time Stock Prediction Inference Script

Loads a trained model and generates probability-calibrated predictions
for one or more stock symbols using the latest available market data.

Usage
-----
  python predict.py --symbol AAPL
  python predict.py --symbol AAPL MSFT NVDA --horizon 5
  python predict.py --symbol TSLA --model_dir models --top_features 15
  python predict.py --symbols_file watchlist.txt --output predictions.csv
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

# ── Rich terminal UI ──────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# ── Local imports ─────────────────────────────────────────────────────────────
from stock_data_generator import (
    get_feature_engineered_stock_data,
    normalize_features,
    PREDICTION_HORIZONS,
)
from train_stock_model import (
    build_model,
    compute_feature_importance,
    print_feature_importance,
    print_banner,
    BANNER,
    VERSION,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.FileHandler("predict.log")],
)
logger = logging.getLogger(__name__)

console = Console() if RICH_AVAILABLE else None


def cprint(msg: str, style: str = "") -> None:
    if RICH_AVAILABLE and console:
        console.print(msg, style=style)
    else:
        print(msg)


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_dir: str) -> tuple:
    """Load saved model, config, and feature names from *model_dir*.

    Returns (model, config, feature_names, device).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(model_dir, "final_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model checkpoint found in {model_dir!r}")

    checkpoint = torch.load(model_path, map_location=device)
    config        = checkpoint["config"]
    feature_names = checkpoint.get("feature_names", [])
    model = build_model(
        config.get("model_type", "hybrid"),
        len(feature_names) if feature_names else config.get("input_dim", 100),
        config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    cprint(
        f"[dim]Loaded model ({config.get('model_type','hybrid')}) from {model_path!r}  "
        f"({sum(p.numel() for p in model.parameters()):,} params)[/dim]"
    )
    return model, config, feature_names, device


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

def predict_symbols(
    symbols: List[str],
    model: torch.nn.Module,
    config: dict,
    feature_names: List[str],
    device: torch.device,
    horizon: int = 1,
    lookback_days: int = 365,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch recent data, engineer features, and run inference for each symbol.

    Returns a DataFrame with columns:
        Symbol, Date, Close, Horizon_Days, Pred_Prob, Direction, Confidence
    """
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    seq_length = config.get("seq_length", 30)

    rows = []
    spinner_cols = [SpinnerColumn(), TextColumn("[cyan]{task.description}")]

    if RICH_AVAILABLE and console:
        progress_ctx = Progress(*spinner_cols, console=console, transient=True)
    else:
        progress_ctx = _NullProgress()

    with progress_ctx as prog:
        task = prog.add_task(f"Fetching data for {len(symbols)} symbol(s)…", total=None)
        raw_data = get_feature_engineered_stock_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            min_rows=1,
            cache_dir=cache_dir,
            n_jobs=min(4, len(symbols)),
            include_market_context=True,
        )
        prog.update(task, description="Engineering features…")
        norm_data = normalize_features(raw_data, scaler_type="robust")

    for symbol in symbols:
        sym_df = norm_data[norm_data["Symbol"] == symbol].copy() if "Symbol" in norm_data.columns else norm_data.copy()
        if len(sym_df) < seq_length + 5:
            cprint(f"[yellow]⚠  {symbol}: not enough data ({len(sym_df)} rows), skipping.[/yellow]")
            continue

        num_cols = [c for c in feature_names if c in sym_df.columns]
        missing  = [c for c in feature_names if c not in sym_df.columns]
        if missing:
            logger.warning(f"{symbol}: {len(missing)} features missing, filling with 0.")
        feat_df  = sym_df[num_cols].copy()
        for mc in missing:
            feat_df[mc] = 0.0
        feat_df = feat_df[feature_names]

        # Fill NaN with column mean
        feat_df = feat_df.fillna(feat_df.mean()).fillna(0)

        # Build the last sequence window
        vals = feat_df.values.astype(np.float32)
        seq  = vals[-seq_length:]
        x    = torch.from_numpy(seq).unsqueeze(0).to(device)   # (1, T, D)

        with torch.no_grad():
            out = model(x)
            if isinstance(out, dict):
                key  = f"h{horizon}" if f"h{horizon}" in out else next(iter(out))
                logit = out[key].item()
            else:
                logit = out.item()

        prob      = float(torch.sigmoid(torch.tensor(logit)).item())
        direction = "UP ▲" if prob > 0.5 else "DOWN ▼"
        confidence = abs(prob - 0.5) * 2   # 0 = uncertain, 1 = certain

        last_close = sym_df["Close"].iloc[-1] if "Close" in sym_df.columns else np.nan
        last_date  = sym_df.index[-1] if isinstance(sym_df.index, pd.DatetimeIndex) else "latest"

        rows.append({
            "Symbol":       symbol,
            "Date":         str(last_date)[:10],
            "Close":        round(float(last_close), 2) if not np.isnan(last_close) else None,
            "Horizon_Days": horizon,
            "Pred_Prob":    round(prob, 4),
            "Direction":    direction,
            "Confidence":   round(confidence, 4),
        })
        logger.info(f"{symbol}: {direction}  prob={prob:.4f}  conf={confidence:.4f}")

    return pd.DataFrame(rows)


class _NullProgress:
    """Fallback context manager when rich is unavailable."""
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def add_task(self, *_, **__):
        return None
    def update(self, *_, **__):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Rich output
# ══════════════════════════════════════════════════════════════════════════════

def print_predictions(df: pd.DataFrame) -> None:
    """Print prediction results as a styled rich table."""
    if df.empty:
        cprint("[red]No predictions generated.[/red]")
        return

    if RICH_AVAILABLE and console:
        table = Table(
            title=f"Warren Predictions  ·  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            box=box.ROUNDED, header_style="bold cyan", show_edge=True,
        )
        table.add_column("Symbol",       style="bold yellow", width=8)
        table.add_column("Date",         style="dim",         width=12)
        table.add_column("Close",        justify="right",     width=10)
        table.add_column("Horizon",      justify="center",    width=9)
        table.add_column("Probability",  justify="right",     width=12)
        table.add_column("Direction",    justify="center",    width=10)
        table.add_column("Confidence",   justify="right",     width=12)
        table.add_column("Bar",          min_width=20)

        for _, row in df.sort_values("Pred_Prob", ascending=False).iterrows():
            prob = row["Pred_Prob"]
            conf = row["Confidence"]
            is_up  = "UP" in str(row["Direction"])
            dir_style  = "green" if is_up else "red"
            prob_style = "green" if prob > 0.55 else ("red" if prob < 0.45 else "yellow")
            conf_bar   = "█" * int(conf * 18) + "░" * (18 - int(conf * 18))
            table.add_row(
                str(row["Symbol"]),
                str(row["Date"]),
                f"${row['Close']:,.2f}" if row["Close"] else "—",
                f"{row['Horizon_Days']}d",
                f"[{prob_style}]{prob:.1%}[/{prob_style}]",
                f"[{dir_style}]{row['Direction']}[/{dir_style}]",
                f"{conf:.1%}",
                f"[{dir_style}]{conf_bar}[/{dir_style}]",
            )
        console.print(table)
    else:
        print(f"\n{'Symbol':<8} {'Date':<12} {'Close':>8} {'Prob':>7} {'Dir':<8} {'Conf':>7}")
        print("-" * 56)
        for _, row in df.sort_values("Pred_Prob", ascending=False).iterrows():
            print(
                f"{row['Symbol']:<8} {str(row['Date']):<12} "
                f"${row['Close']:>7.2f}  {row['Pred_Prob']:>6.1%}  "
                f"{row['Direction']:<8}  {row['Confidence']:>6.1%}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args) -> pd.DataFrame:
    print_banner()

    # ── Resolve symbols ───────────────────────────────────────────────────────
    symbols: List[str] = []
    if args.symbol:
        symbols = [s.strip().upper() for s in args.symbol]
    if args.symbols_file and os.path.exists(args.symbols_file):
        with open(args.symbols_file) as fh:
            symbols += [ln.strip().upper() for ln in fh if ln.strip()]
    symbols = list(dict.fromkeys(symbols))   # deduplicate preserving order
    if not symbols:
        cprint("[red]Error: no symbols specified. Use --symbol or --symbols_file.[/red]")
        sys.exit(1)

    cprint(f"[bold cyan]Generating predictions for:[/bold cyan] {', '.join(symbols)}")

    # ── Load model ────────────────────────────────────────────────────────────
    model, config, feature_names, device = load_model(args.model_dir)

    # ── Predict ───────────────────────────────────────────────────────────────
    horizon = args.horizon if args.horizon in PREDICTION_HORIZONS else 1
    preds   = predict_symbols(
        symbols, model, config, feature_names, device,
        horizon=horizon,
        lookback_days=args.lookback_days,
        cache_dir=args.cache_dir,
    )

    # ── Display ───────────────────────────────────────────────────────────────
    print_predictions(preds)

    # ── Feature importance ────────────────────────────────────────────────────
    if args.show_features and feature_names:
        cprint("\n[cyan]Computing feature importance…[/cyan]")
        imp_df = compute_feature_importance(
            model, feature_names, device,
            seq_length=config.get("seq_length", 30),
            top_n=args.top_features,
        )
        print_feature_importance(imp_df)

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.output and not preds.empty:
        preds.to_csv(args.output, index=False)
        cprint(f"[green]Predictions saved to {args.output}[/green]")

    return preds


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Warren — Real-Time Stock Prediction Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict.py --symbol AAPL\n"
            "  python predict.py --symbol AAPL MSFT NVDA --horizon 5\n"
            "  python predict.py --symbols_file watchlist.txt --output preds.csv\n"
            "  python predict.py --symbol TSLA --show_features --top_features 20\n"
        ),
    )
    ap.add_argument("--symbol",        nargs="+",  default=[],
                    help="One or more ticker symbols (e.g. AAPL MSFT)")
    ap.add_argument("--symbols_file",  type=str,   default=None,
                    help="Text file with one symbol per line")
    ap.add_argument("--model_dir",     type=str,   default="models",
                    help="Directory containing final_model.pth / best_model.pth")
    ap.add_argument("--horizon",       type=int,   default=1,
                    choices=PREDICTION_HORIZONS,
                    help="Forecast horizon in trading days (1, 5, 10, or 21)")
    ap.add_argument("--lookback_days", type=int,   default=400,
                    help="Days of history to fetch for feature engineering")
    ap.add_argument("--cache_dir",     type=str,   default=None,
                    help="Directory for data caching")
    ap.add_argument("--show_features", action="store_true",
                    help="Display feature importance after predictions")
    ap.add_argument("--top_features",  type=int,   default=20,
                    help="Number of top features to display")
    ap.add_argument("--output",        type=str,   default=None,
                    help="Optional CSV path to save predictions")
    main(ap.parse_args())
