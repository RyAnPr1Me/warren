from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression

def mutual_information_rank(df: pd.DataFrame, features: list[str], target: str, n_neighbors: int = 5) -> pd.DataFrame:
    X = df[features].fillna(0.0).values
    y = df[target].values
    mi = mutual_info_regression(X, y, n_neighbors=n_neighbors, random_state=42)
    return pd.DataFrame({"feature": features, "mi": mi}).sort_values("mi", ascending=False).reset_index(drop=True)


def select_top_k(mi_df: pd.DataFrame, k: int) -> list[str]:
    return mi_df.head(k)["feature"].tolist()


def permutation_importance_placeholder():
    # Phase 2 stub — implement after baseline stacking ready
    pass
