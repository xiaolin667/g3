from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - np.sum(err**2) / denom) if denom > 0 else float("nan")
    smape = float(np.mean(2.0 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)))
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "smape": smape}


def add_average_ranks(metrics: pd.DataFrame, metric: str = "rmse") -> pd.DataFrame:
    rows = []
    for seed, seed_df in metrics.groupby("seed", dropna=False):
        pivot = seed_df.pivot_table(index="dataset", columns="model", values=metric, aggfunc="mean")
        ranks = pivot.rank(axis=1, method="average", ascending=True)
        for model, avg_rank in ranks.mean(axis=0).items():
            rows.append({"seed": seed, "model": model, f"avg_rank_{metric}": float(avg_rank)})
    return pd.DataFrame(rows)
