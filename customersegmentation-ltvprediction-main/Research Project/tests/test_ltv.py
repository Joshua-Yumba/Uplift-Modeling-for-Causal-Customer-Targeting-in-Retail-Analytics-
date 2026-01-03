import pytest
import pandas as pd
from src.ltv_prediction import predict_ltv

def test_predict_ltv():
    rfm = pd.DataFrame({
        'frequency': [1, 2, 3],
        'recency': [10, 20, 30],
        'T': [50, 50, 50],
        'monetary_value': [100, 200, 300]
    })
    config = {
        'penalizer_coef_bgf': 0.001,
        'penalizer_coef_ggf': 0.0,
        'monthly_discount_rate': 0.01,
        'prediction_period_months': 12
    }
    rfm_result = predict_ltv(rfm, config)
    assert 'predicted_purchases_30' in rfm_result.columns
    assert 'prob_alive' in rfm_result.columns
    assert 'expected_avg_profit' in rfm_result.columns
    assert 'CLV' in rfm_result.columns
    assert rfm_result['CLV'].min() > 0