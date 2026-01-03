# Customer Segmentation & LTV Analytics (INDIA_RETAIL_DATA)

End-to-end pipeline that cleans retail transactions, builds RFM features, segments cities, predicts lifetime value, and augments insights with ML-based CLV and churn propensity models. Visualizations and CSV reports are generated automatically.

## Features
- RFM aggregation using the `lifetimes` library (BG/NBD + Gamma-Gamma).
- Clustering by K-Means or optional auto-selected Gaussian Mixture (BIC).
- ML-based CLV regressor (Gradient Boosting) for cross-checking probabilistic CLV.
- Churn propensity model (Gradient Boosting classifier) with risk ranking.
- Plots: RFM distributions, elbow curve, cluster scatter, CLV distribution, CLV by cluster.
- CSV outputs: CLV predictions, segment analysis, top customers, top churn risks.
- Structured logging to console (and optionally to `output/logs/`).

## Project Structure
- `main.py` – orchestrates the entire workflow.
- `config/config.yaml` – tweak clustering, LTV, and AI feature toggles.
- `src/data_preprocessing.py` – cleans dataset, computes transaction summaries.
- `src/segmentation.py` – clustering utilities.
- `src/ltv_prediction.py` – probabilistic CLV estimation.
- `src/ml_clv.py` – Gradient Boosting CLV model.
- `src/churn.py` – churn labeling and classification.
- `src/visualization.py` – plot generators.
- `tests/` – basic unit tests for preprocessing, segmentation, and LTV.
- `output/` – generated figures and CSV results.

## Prerequisites
- Python 3.10+
- `INDIA_RETAIL_DATA.xlsx` placed in `data/raw/` (sheet name `retails`).
- (Optional) GPU not required; models are light-weight.

## Installation
```bash
git clone <repository-url>
cd Customer_Segmentation_LTV
python -m venv venv
./venv/Scripts/activate    # Windows PowerShell
pip install -r requirements.txt
```

## Configuration
Edit `config/config.yaml` as needed:
```yaml
clustering:
  n_clusters: 4
ltv:
  penalizer_coef_bgf: 0.001
  penalizer_coef_ggf: 0.0
  monthly_discount_rate: 0.01
  prediction_period_months: 12
ai:
  use_ml_clv: true              # train Gradient Boosting CLV
  use_churn: true               # train churn classifier
  use_auto_gmm_segmentation: false  # auto-select clusters via GMM+BIC
  max_gmm_components: 7
```

- Set `use_auto_gmm_segmentation: true` to let the pipeline pick cluster count automatically.
- Adjust `max_gmm_components` to cap clusters considered during BIC search.
- Change churn horizon by editing the `label_churn(... horizon_days=90)` call in `main.py`.

## Running the Pipeline
```bash
./venv/Scripts/activate
python main.py
```

The script will:
1. Load and clean the Excel dataset.
2. Build RFM metrics and (optionally) profit aggregation.
3. Segment cities using K-Means or auto GMM.
4. Estimate CLV via BG/NBD + Gamma-Gamma.
5. Train optional ML CLV and churn propensity models.
6. Generate plots and save CSV reports under `output/`.
7. Print key summaries and correlation checks to the console.

### Interactive Dashboard
```bash
./venv/Scripts/activate
streamlit run dashboard.py
```

- Unique "Customer Intelligence Command Center" UI with hero banner, KPI cards, and interactive tabs.
- Overview tab shows run history timeline, insights, and highlights.
- Segments tab offers interactive Plotly scatter/pie charts with cluster filters.
- CLV tab compares probabilistic vs ML CLV distributions.
- Churn tab surfaces risk distribution, threshold slider, and top-risk cities.
- Diagnostics tab reveals run logs and current configuration snapshot.

## Outputs
- `output/figures/`
  - `rfm_distributions.png`
  - `elbow_plot.png`
  - `cluster_scatter.png`
  - `clv_distribution.png`
  - `clv_by_cluster.png`
- `output/results/`
  - `clv_predictions.csv` (includes `CLV`, `CLV_ML`, `churn_probability` when enabled)
  - `segment_analysis.csv`
  - `top_customers.csv`
  - `top_churn_risk.csv`

## Testing
```bash
./venv/Scripts/activate
pytest
```

## Troubleshooting
- **Missing Excel engine**: install `openpyxl` (already listed in `requirements.txt`).
- **Negative monetary_value or profit**: handled via clipping in preprocessing.
- **Plot warnings about categorical units**: ensure numeric columns remain numeric; already cast where needed.
- **Large runtime**: reduce `n_clusters`/`max_gmm_components` or disable optional AI features.

Happy experimenting! Feel free to extend with feature importance plots, uplift modeling, or LLM-generated insights by adding new modules under `src/` and toggling them via config.