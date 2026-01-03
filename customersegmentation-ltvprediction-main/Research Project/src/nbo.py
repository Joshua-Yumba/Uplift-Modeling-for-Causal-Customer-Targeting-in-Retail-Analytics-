# src/nbo.py
import pandas as pd
from src.dashboard_utils import RESULTS_DIR

def run_nbo_recommendations(df: pd.DataFrame):
    df = df.copy()
    df["offer_score"] = df["CLV"] * (100 / (df["recency"] + 1))
    df["recommended_product"] = df["offer_score"].apply(
        lambda x: "Premium Bundle" if x > df["offer_score"].quantile(0.8) else "Standard Plan"
    )
    df = df.sort_values("offer_score", ascending=False)

    path = RESULTS_DIR / "nbo_recommendations.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"NBO saved: {path}")