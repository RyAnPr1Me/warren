from __future__ import annotations
import yaml
from pathlib import Path
from dataclasses import dataclass

CONFIG_PATH = Path("config/config.yaml")

@dataclass
class Config:
    data_root: str
    cache_ttl_minutes: int
    seq_len: int
    target: str
    horizons: list[int]
    batch_size: int
    epochs: int
    early_stop_patience: int
    lr: float
    optimizer: str
    mi_top_k: int
    importance_prune_pct: float
    serve_host: str
    serve_port: int


def load_config() -> Config:
    with open(CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f)
    return Config(
        data_root=raw["data"]["root"],
        cache_ttl_minutes=raw["data"]["cache_ttl_minutes"],
        seq_len=raw["model"]["seq_len"],
        target=raw["model"]["target"],
        horizons=raw["model"]["horizons"],
        batch_size=raw["train"]["batch_size"],
        epochs=raw["train"]["epochs"],
        early_stop_patience=raw["train"]["early_stop_patience"],
        lr=float(raw["train"]["lr"]),
        optimizer=raw["train"]["optimizer"],
        mi_top_k=raw["feature"]["mi_top_k"],
        importance_prune_pct=float(raw["feature"]["importance_prune_pct"]),
        serve_host=raw["serve"]["host"],
        serve_port=raw["serve"]["port"],
    )
