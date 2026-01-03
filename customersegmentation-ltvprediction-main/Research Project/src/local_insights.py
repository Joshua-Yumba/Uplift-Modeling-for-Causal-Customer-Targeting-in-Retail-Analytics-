# src/local_insights.py
import pandas as pd
from typing import Generator

def generate_insight(query: str, df: pd.DataFrame) -> str:
    query = query.lower()
    total_cities = len(df)
    avg_clv = df['CLV'].mean()
    high_risk = (df.get('churn_probability', pd.Series([0])) > 0.7).mean() * 100
    top_city = df.nlargest(1, 'CLV')['customer_id'].iloc[0] if not df.empty else "N/A"

    if "top" in query or "best" in query:
        return f"The top-performing city is **{top_city}** with a CLV of {df['CLV'].max():,.0f}."
    elif "churn" in query or "risk" in query:
        return f"Currently, **{high_risk:.1f}%** of cities have a churn risk above 70%. Focus retention efforts on high-CLV, high-risk segments."
    elif "cluster" in query:
        clusters = df['cluster'].nunique() if 'cluster' in df.columns else 0
        return f"There are **{clusters} customer segments** identified using AI-driven GMM clustering."
    elif "forecast" in query:
        return f"Forecasting is active. Use the Forecast tab to explore 30-day CLV predictions per city."
    elif "nbo" in query or "offer" in query:
        return f"Next-Best-Offer engine is running. High-CLV, low-recency customers are prioritized for premium upsell."
    else:
        return f"Total cities: {total_cities:,}. Avg CLV: {avg_clv:,.0f}. Use tabs to explore CLV, churn, and forecasts."

def stream_insight(text: str) -> Generator[str, None, None]:
    for char in text:
        yield char