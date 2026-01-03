from lifetimes import BetaGeoFitter, GammaGammaFitter

def predict_ltv(rfm, config):
    """
    Predict Lifetime Value using BG/NBD and Gamma-Gamma models.
    Returns RFM DataFrame with predicted purchases, probability alive, expected profit, and CLV.
    """
    # BG/NBD Model for purchase frequency
    bgf = BetaGeoFitter(penalizer_coef=config['penalizer_coef_bgf'])
    bgf.fit(rfm['frequency'], rfm['recency'], rfm['T'])
    rfm['predicted_purchases_30'] = bgf.conditional_expected_number_of_purchases_up_to_time(
        30, rfm['frequency'], rfm['recency'], rfm['T']
    )
    rfm['prob_alive'] = bgf.conditional_probability_alive(rfm['frequency'], rfm['recency'], rfm['T'])

    # Gamma-Gamma Model for monetary value
    valid_mask = (rfm['monetary_value'] > 0) & (rfm['frequency'] > 0)
    ggf = GammaGammaFitter(penalizer_coef=config['penalizer_coef_ggf'])
    if valid_mask.any():
        ggf.fit(rfm.loc[valid_mask, 'frequency'], rfm.loc[valid_mask, 'monetary_value'])
        rfm['expected_avg_profit'] = 0.0
        rfm.loc[valid_mask, 'expected_avg_profit'] = ggf.conditional_expected_average_profit(
            rfm.loc[valid_mask, 'frequency'], rfm.loc[valid_mask, 'monetary_value']
        )
    else:
        rfm['expected_avg_profit'] = 0.0

    # Calculate CLV with profit adjustment if available (hypothetical; adjust based on dataset)
    rfm['CLV'] = 0.0
    if 'profit_adjusted' in rfm.columns and valid_mask.any():  # Assume profit data is preprocessed
        rfm.loc[valid_mask, 'CLV'] = ggf.customer_lifetime_value(
            bgf,
            rfm.loc[valid_mask, 'frequency'],
            rfm.loc[valid_mask, 'recency'],
            rfm.loc[valid_mask, 'T'],
            rfm.loc[valid_mask, 'profit_adjusted'],  # Use profit instead of monetary_value if available
            time=config['prediction_period_months'],
            discount_rate=config['monthly_discount_rate']
        )
    elif valid_mask.any():
        rfm.loc[valid_mask, 'CLV'] = ggf.customer_lifetime_value(
            bgf,
            rfm.loc[valid_mask, 'frequency'],
            rfm.loc[valid_mask, 'recency'],
            rfm.loc[valid_mask, 'T'],
            rfm.loc[valid_mask, 'monetary_value'],
            time=config['prediction_period_months'],
            discount_rate=config['monthly_discount_rate']
        )
    return rfm