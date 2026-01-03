# src/uplift.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.ltv_prediction import predict_ltv

from src.dashboard_utils import RESULTS_DIR
from src.data_preprocessing import load_and_clean_data, calculate_rfm  # <- Added

def run_uplift_modeling(raw_file_path: str):
    """
    Uplift modeling to predict which customers are most likely to respond to a campaign.

    Steps:
    1. Load and preprocess raw transaction data
    2. Calculate RFM features (recency, frequency, monetary value)
    3. Assign treatment/control group
    4. Split dataset into features and target
    5. Train separate models for treatment and control
    6. Compute uplift score per customer
    7. Evaluate model using AUUC / baseline comparison
    8. Visualize top responders and feature importance
    """

    # -----------------------
    # Step 1: Load and preprocess data
    # -----------------------
    transaction_data = load_and_clean_data(raw_file_path)
    rfm = calculate_rfm(transaction_data)
    
    ltv_config={'penalizer_coef_bgf' :0.0,
                
                'penalizer_coef_ggf': 0.0,
                
                'prediction_period_months':6,
                
                'monthly_discount_rate': 0.01
                    }
    rfm=predict_ltv(rfm,config=ltv_config)

    df = rfm.reset_index().rename(columns={'index': 'customer_id'})  # Add customer_id column

    # -----------------------
    # Step 2: Assign treatment/control groups
    # -----------------------
    np.random.seed(42)
    df["treatment_group"] = np.where(np.random.rand(len(df)) < 0.5,"Treatment","Control")

    # Step 2b: Define target (response)
    df["response"] =((df["CLV"]>df["CLV"].median()) | (np.random.rand(len(df))<0.2)).astype(int)
    # -----------------------
    # Step 3: Prepare features
    # -----------------------
    feature_cols = [c for c in df.columns if c not in ["customer_id", "treatment_group", "response"]]
    X = df[feature_cols].copy()
    y = df["response"]

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    treat_idx = df[df["treatment_group"] == "Treatment"].index
    ctrl_idx = df[df["treatment_group"] == "Control"].index

    X_treat, y_treat = X.loc[treat_idx], y.loc[treat_idx]
    X_ctrl, y_ctrl = X.loc[ctrl_idx], y.loc[ctrl_idx]
    
    #----safety check for single-class
    if len(y_treat.unique())<2:
        print("Warning:Treatement group has only one class.Skipping uplift modeling.")
        return df
    if len(y_ctrl.unique())<2:
        print("Warning:Control group has only one class. Skipping uplift modeling.")
        return df

    # -----------------------
    # Step 4: Train separate models
    # -----------------------
    clf_treat = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_ctrl = RandomForestClassifier(n_estimators=100, random_state=42)

    clf_treat.fit(X_treat, y_treat)
    clf_ctrl.fit(X_ctrl, y_ctrl)

    # -----------------------
    # Step 5: Compute uplift scores
    # -----------------------
    prob_treat = clf_treat.predict_proba(X)[:, 1]
    prob_ctrl = clf_ctrl.predict_proba(X)[:, 1]

    df["uplift"] = (prob_treat - prob_ctrl).round(3)

    # -----------------------
    # Step 6: Evaluate uplift
    # -----------------------
    top20 = df.sort_values("uplift", ascending=False).head(20)

    plt.figure(figsize=(10,6))
    sns.barplot(x="customer_id", y="uplift", data=top20)
    plt.title("Top 20 Customers by Predicted Uplift")
    plt.xticks(rotation=45)
    plt.ylabel("Predicted Uplift")
    plt.tight_layout()
    plt.show()

    top20_sum = top20["uplift"].sum()
    random20_sum = df.sample(20, random_state=42)["uplift"].sum()
    print(f"Total predicted uplift (Top 20): {top20_sum:.3f}")
    print(f"Total predicted uplift (Random 20): {random20_sum:.3f}")
    print(f"Improvement over baseline: {top20_sum - random20_sum:.3f}")

    # -----------------------
    # Step 7: Feature importance
    # -----------------------
    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": clf_treat.feature_importances_
    }).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(12,6))
    sns.barplot(x="importance", y="feature", data=importances.head(15))
    plt.title("Top 15 Features Influencing Treatment Response")
    plt.tight_layout()
    plt.show()

    # -----------------------
    # Step 8: Save results
    # -----------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "uplift_results.csv"
    df[["customer_id", "response", "uplift", "treatment_group", "CLV"]].to_csv(path, index=False)
    print(f"Uplift results saved: {path}")

    return df

#-------------------------
#Run script directly 
#-------------------------
if __name__=="__main__":
    raw_file="data/raw/INDIA_RETAIL_DATA.xlsx"
# Update this path if needed 

    df_uplift= run_uplift_modeling(raw_file)
    print(df_uplift.head())