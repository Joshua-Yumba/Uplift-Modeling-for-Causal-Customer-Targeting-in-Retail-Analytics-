from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


def train_clv_model(rfm: pd.DataFrame) -> Tuple[GradientBoostingRegressor, dict]:
    features = ['recency', 'frequency', 'monetary_value', 'prob_alive', 'cluster', 'profit_adjusted']
    available = [f for f in features if f in rfm.columns]
    X = rfm[available].copy()
    # Coerce numerics and handle non-finite values
    X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(rfm['CLV'], errors='coerce')

    # Keep only rows with finite y and all feature values present
    mask = y.notna()
    mask &= X.notna().all(axis=1)
    X = X.loc[mask]
    y = y.loc[mask]

    # Simple split since no explicit holdout period is available
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    metrics = {
        'r2': float(r2_score(y_valid, preds)),
        'mae': float(mean_absolute_error(y_valid, preds))
    }
    return model, metrics


def predict_clv_ml(rfm: pd.DataFrame, model: GradientBoostingRegressor) -> pd.DataFrame:
    features = ['recency', 'frequency', 'monetary_value', 'prob_alive', 'cluster', 'profit_adjusted']
    available = [f for f in features if f in rfm.columns]
    X = rfm[available].copy()
    rfm['CLV_ML'] = model.predict(X)
    return rfm


