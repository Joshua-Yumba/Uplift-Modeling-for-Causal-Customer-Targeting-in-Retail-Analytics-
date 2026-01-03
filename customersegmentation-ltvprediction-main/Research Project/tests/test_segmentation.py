import pytest
import pandas as pd
from src.segmentation import perform_clustering, get_elbow_data

def test_perform_clustering():
    rfm = pd.DataFrame({
        'recency': [10, 20, 30],
        'frequency': [5, 10, 15],
        'monetary_value': [100, 200, 300]
    })
    clusters = perform_clustering(rfm, n_clusters=2)
    assert len(clusters) == 3
    assert len(set(clusters)) == 2

def test_get_elbow_data():
    rfm = pd.DataFrame({
        'recency': [10, 20, 30, 40],
        'frequency': [5, 10, 15, 20],
        'monetary_value': [100, 200, 300, 400]
    })
    inertia = get_elbow_data(rfm)
    assert len(inertia) == 7
    assert all(isinstance(i, float) for i in inertia)