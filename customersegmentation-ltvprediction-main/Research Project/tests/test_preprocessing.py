import pytest
import pandas as pd
from src.data_preprocessing import load_and_clean_data, calculate_rfm

def test_load_and_clean_data():
    df = load_and_clean_data('data/raw/INDIA_RETAIL_DATA.xlsx')
    assert 'City' in df.columns
    assert 'Order Date' in df.columns
    assert 'Sales' in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['Order Date'])
    assert df.shape[1] == 3  # City, Order Date, Sales (after grouping)

def test_calculate_rfm():
    sample_data = pd.DataFrame({
        'customer_id': [1, 1, 2],
        'date': [pd.to_datetime('2023-01-01'), pd.to_datetime('2023-02-01'), pd.to_datetime('2023-01-15')],
        'revenues': [100, 200, 150]
    })
    rfm = calculate_rfm(sample_data)
    assert 'frequency' in rfm.columns
    assert 'recency' in rfm.columns
    assert 'T' in rfm.columns
    assert 'monetary_value' in rfm.columns
    assert rfm.shape[0] == 2  # Two unique 'customers'