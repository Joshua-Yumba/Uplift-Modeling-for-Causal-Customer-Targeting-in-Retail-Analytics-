from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def label_churn(transaction_data: pd.DataFrame, horizon_days: int = 90) -> pd.Series:
    # transaction_data has columns: City, Order Date, Sales, Profit
    df = transaction_data.rename(columns={'City': 'customer_id', 'Order Date': 'date'}).copy()
    df['date'] = pd.to_datetime(df['date'])
    last_date = df['date'].max()
    cutoff = last_date - pd.Timedelta(days=horizon_days)
    last_by_customer = df.groupby('customer_id')['date'].max()
    # Churn label: 1 if no purchase in the last horizon_days
    labels = (last_by_customer < cutoff).astype(int)
    return labels


def train_churn_model(rfm: pd.DataFrame, labels: pd.Series):
    # Align labels to rfm index (customer_id)
    labels = labels.reindex(rfm.index).fillna(0).astype(int)
    features = ['recency', 'frequency', 'monetary_value', 'prob_alive', 'cluster', 'profit_adjusted']
    available = [f for f in features if f in rfm.columns]
    X = rfm[available].copy()
    y = labels

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_valid)[:, 1]
    auc = float(roc_auc_score(y_valid, proba)) if y_valid.nunique() > 1 else float('nan')
    metrics = {'auc': auc}
    return clf, metrics


def predict_churn(rfm: pd.DataFrame, clf) -> pd.DataFrame:
    features = ['recency', 'frequency', 'monetary_value', 'prob_alive', 'cluster', 'profit_adjusted']
    available = [f for f in features if f in rfm.columns]
    X = rfm[available].copy()
    rfm['churn_probability'] = clf.predict_proba(X)[:, 1]
    return rfm


