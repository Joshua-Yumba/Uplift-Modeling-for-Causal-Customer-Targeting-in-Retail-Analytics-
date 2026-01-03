# Customer Segmentation & Lifetime Value (LTV) Prediction  
**A Unified AI Framework Using RFM, Clustering, and Probabilistic Modeling**

---

## Overview

This project implements a **modular, research-grade AI pipeline** for **customer segmentation** and **lifetime value (LTV) prediction** using real retail transaction data.

**Core Features**:
- RFM (Recency, Frequency, Monetary) analysis  
- Clustering with K-Means and Gaussian Mixture Models (GMM)  
- LTV prediction using BG/NBD + Gamma-Gamma models  
- Interactive **Streamlit dashboard** with Plotly visualizations  
- Fully configurable via `config.yaml`  
- Reproducible, production-ready, and open-source  

---

## Project Structure
.
├── main.py                     # Main pipeline execution
├── dashboard.py                # Streamlit web dashboard
├── config/
│   └── config.yaml             # Model & pipeline settings
├── data/
│   └── raw/
│       └── INDIA_RETAIL_DATA.xlsx
├── src/
│   ├── data_preprocessing.py   # Data loading & cleaning
│   ├── segmentation.py         # Clustering logic
│   ├── ltv_prediction.py       # BG/NBD + Gamma-Gamma
│   ├── visualization.py        # Plotly charts
│   ├── dashboard_utils.py      # CSV handling, metrics
│   └── logging_setup.py        # Structured logging
├── output/
│   ├── results/                # CSV outputs
│   └── figures/                # PNG visualizations
└── README.md


---

## Requirements

```txt
pandas
numpy
scikit-learn
lifetimes
plotly
streamlit
pyyaml
matplotlib
seaborn
