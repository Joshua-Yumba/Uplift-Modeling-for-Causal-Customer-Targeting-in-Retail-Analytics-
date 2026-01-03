import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data

def load_and_clean_data(file_path):
    """
    Load and clean the INDIA_RETAIL_DATA dataset.
    Returns a DataFrame with transaction-level data grouped by City and Order Date.
    """
    try:
        df = pd.read_excel(file_path, sheet_name='retails')
        print(f"Loaded dataframe shape: {df.shape}, Profit min: {df['Profit'].min() if 'Profit' in df else 'N/A'}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please place 'INDIA_RETAIL_DATA.xlsx' in data/raw/.")
    
    # Data cleaning
    df = df[pd.notna(df['City'])]  # Remove records without City
    df = df[df['QtyOrdered'] > 0]  # Remove non-positive quantities
    df = df[df['Unit Price'] > 0]  # Ensure positive unit prices
    df = df[df['Sales'] > 0]  # Ensure positive sales

    # Handle negative profits by ensuring numeric type and clipping to 0
    if 'Profit' in df.columns:
        df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce').clip(lower=0)
        print(f"Profit min after clipping (transaction level): {df['Profit'].min()}")
    else:
        print("Warning: Profit column not found in dataset.")
        df['Profit'] = 0  # Default to 0 if missing

    # Outlier detection and capping for Sales and QtyOrdered
    for column in ['Sales', 'QtyOrdered']:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # Convert date columns with robust handling
    for col in ['Order Date', 'Ship Date']:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], origin='1899-12-30', unit='D', errors='coerce')
            else:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            df = df[pd.notna(df[col])]  # Remove rows with invalid dates

    # Drop unnecessary columns, excluding Profit
    unnecessary = ['Order Priority', 'Discount offered', 'Freight Expenses', 'Freight Mode', 'Segment', 
                   'Product Type', 'Product Sub-Category', 'Product Container', 'State', 'Region', 
                   'Country', 'Unit Price', 'QtyOrdered', 'Ship Date']
    df = df.drop(unnecessary, axis=1, errors='ignore')

    # Group into transaction data: sum Sales and Profit by City and Order Date
    transaction_data = df.groupby(['City', 'Order Date'])[['Sales', 'Profit']].sum().reset_index()
    print(f"Transaction data shape: {transaction_data.shape}, Profit min after grouping: {transaction_data['Profit'].min()}")
    return transaction_data

def calculate_rfm(transaction_data):
    """
    Calculate RFM metrics using Lifetimes.
    Returns a DataFrame with recency, frequency, T, and monetary_value, including profit.
    """
    # Work on a copy to avoid mutating the original transaction_data in callers
    tx = transaction_data.copy()
    tx.columns = ['customer_id', 'date', 'revenues', 'profit']  # City as customer_id
    observation_period_end = tx['date'].max()
    rfm = summary_data_from_transaction_data(
        tx,
        customer_id_col='customer_id',
        datetime_col='date',
        monetary_value_col='revenues',
        observation_period_end=observation_period_end
    )
    # Aggregate profit per customer (city)
    rfm['profit_adjusted'] = tx.groupby('customer_id')['profit'].sum().reindex(rfm.index, fill_value=0)
    print(f"RFM data shape: {rfm.shape}, Profit adjusted min: {rfm['profit_adjusted'].min()}")
    return rfm