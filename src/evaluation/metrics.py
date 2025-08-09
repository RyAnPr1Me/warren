from __future__ import annotations
import numpy as np
import pandas as pd

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    diff = y_true - y_pred
    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    mape = float(np.mean(np.abs(diff / (y_true + 1e-9))))
    direction = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    return {"rmse": rmse, "mae": mae, "mape": mape, "directional_accuracy": direction}
