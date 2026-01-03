# Uplift-Modeling-for-Causal-Customer-Targeting-in-Retail-Analytics-
This repository implements an end-to-end retail analytics pipeline that extends traditional Customer Lifetime Value (CLV) modeling by integrating uplift modeling for causal marketing decision-making.

While standard CLV-based segmentation identifies high-value customers, it does not capture whether a customer’s behavior changes because of a marketing intervention. This project addresses that limitation by estimating individual treatment effects, enabling marketers to target customers who are most likely to respond positively to campaigns.

# The pipeline integrates:

RFM feature engineering

Probabilistic CLV prediction

Simulated treatment/control assignment

Machine-learning-based uplift estimation

The results demonstrate that uplift-based targeting outperforms random selection, improving marketing efficiency and decision quality.

Installation
# Requirements

Python 3.8+

Git

# Usage

Run the uplift modeling pipeline from the project root:

python src/uplift.py

# Workflow Summary

Load and clean transactional retail data

Compute RFM (Recency, Frequency, Monetary) features

Predict Customer Lifetime Value (CLV)

Assign simulated treatment and control groups

Train separate Random Forest models

Estimate individual uplift scores

Visualize top responders and feature importance

Export results for further analysis

The final output is a CSV file containing customer-level uplift scores and predicted CLV.

## Results

# Key findings include:

Customers ranked by uplift deliver higher expected returns than random targeting

CLV and behavioral features contribute strongly to treatment response prediction

The approach bridges predictive analytics and causal inference in retail marketing

These results highlight the practical value of uplift modeling for campaign optimization and personalization.

# Future Work

Planned extensions include:

AUUC (Area Under the Uplift Curve) for formal uplift evaluation

Causal Forests / Meta-Learners for more robust treatment effect estimation

Advanced feature importance analysis (SHAP, permutation importance)

LLM-generated insights to translate model outputs into business-friendly recommendations

Integration with real A/B test data for external validation


# Keywords

Uplift Modeling · Causal Inference · Customer Lifetime Value · Retail Analytics · Marketing Optimization · Machine Learning

