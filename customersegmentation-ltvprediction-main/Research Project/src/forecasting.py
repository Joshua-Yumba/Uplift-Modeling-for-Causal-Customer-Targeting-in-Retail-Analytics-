# src/forecasting.py
import pandas as pd
from datetime import datetime, timedelta
from src.dashboard_utils import RESULTS_DIR

def run_city_forecast(df: pd.DataFrame, days: int = 30):
    """
    Generate 30-day CLV forecast per city. Skips cities with invalid CLV.
    """
    forecast_data = []
    base_date = datetime.today().replace(day=1)
    
    # Filter valid CLV rows
    valid_df = df.dropna(subset=['CLV', 'customer_id']).copy()
    valid_df['CLV'] = pd.to_numeric(valid_df['CLV'], errors='coerce').fillna(0)

    for _, row in valid_df.iterrows():
        city = row["customer_id"]
        clv = float(row["CLV"])
        if clv <= 0:
            continue  # Skip zero/negative CLV

        for i in range(days):
            date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
            historical = clv * (1 + 0.001 * i)
            forecast = historical * 1.05
            forecast_data.append({
                "customer_id": city,
                "date": date,
                "historical_clv": int(round(historical)),
                "forecast_clv": int(round(forecast))
            })

    if not forecast_data:
        print("No valid CLV data for forecasting.")
        return

    result_df = pd.DataFrame(forecast_data)
    path = RESULTS_DIR / "forecast_results.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(path, index=False)
    print(f"Forecast saved: {path} ({len(result_df)} rows)")