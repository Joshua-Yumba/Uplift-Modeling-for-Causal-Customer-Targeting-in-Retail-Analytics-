from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


RESULTS_DIR = Path("output/results")
FIGURES_DIR = Path("output/figures")
RUN_HISTORY_PATH = Path("output/pipeline_runs.json")


def load_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def format_currency(value: float) -> str:
    return f"₹{value:,.0f}" if pd.notna(value) else "—"


def compute_summary_metrics(clv_df: pd.DataFrame) -> Dict[str, float]:
    metrics = {
        "total_cities": int(len(clv_df)) if not clv_df.empty else 0,
        "avg_clv": float(clv_df["CLV"].mean()) if "CLV" in clv_df else 0.0,
        "total_revenue": float((clv_df["frequency"] * clv_df["monetary_value"]).sum())
        if {"frequency", "monetary_value"}.issubset(clv_df.columns)
        else 0.0,
        "avg_churn_probability": float(clv_df.get("churn_probability", pd.Series(dtype=float)).mean())
        if "churn_probability" in clv_df.columns
        else None,
    }
    return metrics


def load_pipeline_history() -> List[Dict]:
    if RUN_HISTORY_PATH.exists():
        try:
            return json.loads(RUN_HISTORY_PATH.read_text())
        except json.JSONDecodeError:
            return []
    return []


def save_pipeline_history(entry: Dict) -> None:
    history = load_pipeline_history()
    history.append(entry)
    RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_HISTORY_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False))


def build_history_entry(rfm: pd.DataFrame, ml_metrics=None, churn_metrics=None) -> Dict:
    entry = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "total_cities": int(len(rfm)),
        "avg_clv": float(rfm.get("CLV", pd.Series(dtype=float)).mean()),
        "median_clv": float(rfm.get("CLV", pd.Series(dtype=float)).median()),
        "clusters": int(rfm.get("cluster", pd.Series(dtype=int)).nunique()),
        "avg_recency": float(rfm.get("recency", pd.Series(dtype=float)).mean()),
    }
    if ml_metrics:
        entry.update({
            "ml_clv_r2": ml_metrics.get("r2"),
            "ml_clv_mae": ml_metrics.get("mae"),
        })
    if churn_metrics:
        entry.update({
            "churn_auc": churn_metrics.get("auc"),
            "avg_churn_probability": float(rfm.get("churn_probability", pd.Series(dtype=float)).mean())
            if "churn_probability" in rfm.columns
            else None,
        })
    return entry


