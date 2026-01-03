# main.py
import yaml
import pandas as pd
from src.data_preprocessing import load_and_clean_data, calculate_rfm
from src.segmentation import perform_clustering, perform_auto_gmm_segmentation
from src.ltv_prediction import predict_ltv
from src.visualization import plot_rfm, plot_elbow, plot_clusters, plot_clv, plot_clv_by_cluster
from src.logging_setup import logger
from src.ml_clv import train_clv_model, predict_clv_ml
from src.churn import label_churn, train_churn_model, predict_churn
from src.dashboard_utils import build_history_entry, save_pipeline_history

# --- NEW: NBO, UPLIFT, FORECASTING ---
from src.nbo import run_nbo_recommendations
from src.uplift import run_uplift_modeling
from src.forecasting import run_city_forecast


def main():
    # Load configuration
    try:
        with open(r"E:\Chosen\customersegmentation-ltvprediction-main\Research Project\config\config.yaml", 'r') as file:
            config = yaml.safe_load(file)
        logger.info("Configuration loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        raise

    # Create output directories
    import os
    os.makedirs('output/figures', exist_ok=True)
    os.makedirs('output/results', exist_ok=True)
    logger.info("Output directories created.")

    # Data preprocessing
    transaction_data = load_and_clean_data('data/raw/INDIA_RETAIL_DATA.xlsx')
    rfm = calculate_rfm(transaction_data)

    # Compute actual correlation
    actual_city_stats = transaction_data.groupby('City').agg(
        frequency=('Sales', 'count'),
        monetary_value=('Sales', 'mean')
    ).reset_index()
    actual_corr = actual_city_stats[['frequency', 'monetary_value']].corr().iloc[0, 1]
    logger.info(f"RFM data calculated for {len(rfm)} cities.")

    # Segmentation
    if config.get('ai', {}).get('use_auto_gmm_segmentation', False):
        rfm['cluster'] = perform_auto_gmm_segmentation(rfm, config.get('ai', {}).get('max_gmm_components', 7)).astype(int)
    else:
        rfm['cluster'] = perform_clustering(rfm, config['clustering']['n_clusters']).astype(int)
    logger.info("Clustering completed.")

    # LTV prediction
    rfm = predict_ltv(rfm, config['ltv'])
    logger.info("LTV prediction completed.")
    ml_metrics = None
    churn_metrics = None

    # Optional: ML-based CLV
    if config.get('ai', {}).get('use_ml_clv', False):
        model, ml_metrics = train_clv_model(rfm)
        rfm = predict_clv_ml(rfm, model)
        logger.info(f"ML CLV trained. R2={ml_metrics['r2']:.3f}, MAE={ml_metrics['mae']:.2f}")

    # Optional: Churn propensity
    if config.get('ai', {}).get('use_churn', False):
        churn_labels = label_churn(transaction_data, horizon_days=90)
        clf, churn_metrics = train_churn_model(rfm, churn_labels)
        rfm = predict_churn(rfm, clf)
        logger.info(f"Churn model trained. AUC={churn_metrics['auc']:.3f}")

    # Visualizations
    plot_rfm(rfm, 'output/figures/rfm_distributions.png')
    plot_elbow(rfm, 'output/figures/elbow_plot.png')
    plot_clusters(rfm, 'output/figures/cluster_scatter.png')
    plot_clv(rfm, 'output/figures/clv_distribution.png')
    plot_clv_by_cluster(rfm, 'output/figures/clv_by_cluster.png')
    logger.info("Visualizations generated.")

    # --- FIX: Save customer_id as COLUMN ---
    rfm_with_id = rfm.reset_index().rename(columns={'index': 'customer_id'})

    # Save core results
    rfm_with_id.to_csv('output/results/clv_predictions.csv', index=False)
    rfm.groupby('cluster')[['recency', 'frequency', 'monetary_value']].mean().to_csv('output/results/segment_analysis.csv')

    # Top 10 customers
    top_10 = rfm.sort_values('CLV', ascending=False).head(10).reset_index().rename(columns={'index': 'customer_id'})
    top_10.to_csv('output/results/top_customers.csv', index=False)

    # Top churn risk
    if 'churn_probability' in rfm.columns:
        top_churn = rfm.sort_values('churn_probability', ascending=False).head(20).reset_index().rename(columns={'index': 'customer_id'})
        top_churn.to_csv('output/results/top_churn_risk.csv', index=False)

    logger.info("Core results saved to CSV files.")

    # ------------------- NBO -------------------
    if config.get("ai", {}).get("use_nbo", False):
        try:
            run_nbo_recommendations(rfm_with_id)
            logger.info("NBO recommendations generated.")
        except Exception as e:
            logger.error(f"NBO failed: {e}")

    # ------------------- UPLIFT MODELING -------------------
    if config.get("ai", {}).get("use_uplift", False):
        try:
            run_uplift_modeling(rfm_with_id)
            logger.info("Uplift modeling completed.")
        except Exception as e:
            logger.error(f"Uplift modeling failed: {e}")

    # ------------------- FORECASTING -------------------
    if config.get("ai", {}).get("use_forecasting", False):
        try:
            run_city_forecast(rfm_with_id)
            logger.info("30-day CLV forecasting completed.")
        except Exception as e:
            logger.error(f"Forecasting failed: {e}")

    # Persist pipeline history
    history_entry = build_history_entry(rfm, ml_metrics=ml_metrics, churn_metrics=churn_metrics)
    save_pipeline_history(history_entry)
    logger.info("Run metrics appended to pipeline history.")

    # Print summary
    print("Segment Analysis:\n", pd.read_csv('output/results/segment_analysis.csv'))
    print("\nTop 10 Cities by CLV:\n", pd.read_csv('output/results/top_customers.csv'))
    print("\nCorrelation between frequency and monetary_value (RFM):",
          rfm[['frequency', 'monetary_value']].corr().iloc[0, 1])
    print("Correlation between frequency and monetary_value (Actual transactions):",
          actual_corr)
    if 'profit_adjusted' in rfm.columns:
        print("\nAverage Profit Adjusted by Cluster:\n", rfm.groupby('cluster')['profit_adjusted'].mean())
    logger.info("Summary printed to console.")


if __name__ == '__main__':
    main()