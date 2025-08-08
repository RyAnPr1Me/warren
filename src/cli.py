from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.utils.config_loader import load_config
from src.data.ingestion.yfinance_fetcher import fetch_prices
from src.features.generator import compute_features, BASIC_FEATURES
from src.models.trainers.train_lstm import train_lstm
from src.models.architectures.transformer_enc import TransformerEncoderModel
from src.models.architectures.tcn import TCN
from src.features.selection import mutual_information_rank, select_top_k
from src.evaluation.metrics import regression_metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI Trading CLI Phase 1")
    sub = p.add_subparsers(dest="command")

    f_fetch = sub.add_parser("fetch", help="Fetch historical data")
    f_fetch.add_argument("symbol")
    f_fetch.add_argument("--period", default="5y")

    f_train = sub.add_parser("train", help="Train model")
    f_train.add_argument("symbol")
    f_train.add_argument("--period", default="5y")
    f_train.add_argument("--model", choices=["lstm","transformer","tcn"], default="lstm")
    f_train.add_argument("--mi-select", action="store_true", help="Apply mutual information feature selection")
    f_train.add_argument("--mi-top-k", type=int, default=50)

    f_predict = sub.add_parser("predict", help="Predict next-day return")
    f_predict.add_argument("symbol")

    return p


def cmd_fetch(args):
    df = fetch_prices(args.symbol, period=args.period)
    out_path = Path("data/raw") / f"{args.symbol}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"Saved {len(df)} rows to {out_path}")


def _prepare_dataset(symbol: str, period: str):
    raw = fetch_prices(symbol, period=period)
    feat = compute_features(raw)
    # Target: future 5-day return of close
    feat["target_5d"] = feat["close"].pct_change(5).shift(-5)
    feat = feat.dropna()
    return feat


def cmd_train(args):
    cfg = load_config()
    df = _prepare_dataset(args.symbol, args.period)
    features = [f for f in BASIC_FEATURES if f in df.columns]
    if args.mi_select:
        mi_df = mutual_information_rank(df, features, "target_5d")
        features = select_top_k(mi_df, args.mi_top_k)
        print(f"Selected top {len(features)} features via MI")
    if args.model == "lstm":
        model = train_lstm(
            df,
            features=features,
            target="target_5d",
            seq_len=cfg.seq_len,
            epochs=3,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
        )
    else:
        # Shared sequence dataset
        from src.models.trainers.train_lstm import SeqDataset
        import torch
        from torch.utils.data import DataLoader
        from torch import nn
        ds = SeqDataset(df, features, "target_5d", cfg.seq_len)
        if len(ds) == 0:
            raise SystemExit("Not enough data")
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.model == "transformer":
            model = TransformerEncoderModel(input_dim=len(features)).to(device)
        else:
            model = TCN(input_dim=len(features)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        loss_fn = nn.MSELoss()
        model.train()
        for epoch in range(1, 4):
            total = 0; count = 0
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(); pred = model(xb).squeeze(-1)
                loss = loss_fn(pred, yb); loss.backward(); opt.step()
                total += loss.item()*len(xb); count += len(xb)
            print(f"Epoch {epoch} ({args.model}) loss={total/count:.6f}")
        # move to cpu for saving
        model = model.to("cpu")
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True, parents=True)
    torch_path = save_dir / f"{args.symbol}_{args.model}.pt"
    import torch
    torch.save(model.state_dict(), torch_path)
    print(f"Model saved to {torch_path}")


def cmd_predict(args):
    cfg = load_config()
    # Use fixed 5y period for now (could be parameterized later)
    df = _prepare_dataset(args.symbol, "5y")
    # Simple naive last known feature row demonstration (not full sequence inference)
    last = df.iloc[-cfg.seq_len:]
    if len(last) < cfg.seq_len:
        raise SystemExit("Not enough data for prediction")
    import torch
    from src.models.architectures.lstm_baseline import LSTMBaseline
    model = LSTMBaseline(input_dim=len(BASIC_FEATURES))
    torch_path = Path("models") / f"{args.symbol}_lstm.pt"
    if not torch_path.exists():
        raise SystemExit("Model not trained yet")
    model.load_state_dict(torch.load(torch_path, map_location="cpu"))
    model.eval()
    import numpy as np
    import torch
    x = torch.tensor(last[BASIC_FEATURES].values, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = model(x).item()
    print(f"Predicted next 5-day return: {pred:.4%}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        raise SystemExit("Unknown command")

if __name__ == "__main__":
    main()
