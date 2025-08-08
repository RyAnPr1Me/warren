from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

class SimpleStacker:
    def __init__(self):
        self.meta = Ridge(alpha=1.0)
        self.fitted = False

    def fit(self, base_preds: dict[str, np.ndarray], y: np.ndarray):
        X = np.column_stack([base_preds[k] for k in sorted(base_preds)])
        self.meta.fit(X, y)
        self.fitted = True

    def predict(self, base_preds: dict[str, np.ndarray]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Stacker not fitted")
        X = np.column_stack([base_preds[k] for k in sorted(base_preds)])
        return self.meta.predict(X)
