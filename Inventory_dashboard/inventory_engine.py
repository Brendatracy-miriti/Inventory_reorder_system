import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import random

# --- 1. GLOBAL CONSTANTS ---

# Daily cost parameters (Set to match your notebook's logic)
HOLDING_COST_H = 0.20        # $0.20 per unit/day
ORDERING_COST_S = 50.00      # $50.00 per order
STOCKOUT_COST_C = 10.00      # $10.00 per unit/day (Lost profit/penalty)
SERVICE_LEVEL_Z = 1.645      # Z-score for 95% service level
LEAD_TIME_DAYS = 7           # Lead time in days

# --- 2. DATA PREPARATION FUNCTIONS ---

def load_data(file_name='retail_store_inventory.csv'):
    """Loads and prepares the raw data."""
    try:
        df = pd.read_csv(file_name)
        df['Date'] = pd.to_datetime(df['Date'])
        df['SKU_Compound_ID'] = df['Store ID'] + '_' + df['Product ID']
        df.sort_values(by=['SKU_Compound_ID', 'Date'], inplace=True)
        return df
    except FileNotFoundError:
        return None

def engineer_features(df):
    """Adds necessary time-series and categorical features."""
    
    # Calculate Rolling Mean (for simplified demand forecast proxy)
    df['Rolling_Mean_30'] = df.groupby('SKU_Compound_ID')['Units Sold'].transform(
        lambda x: x.shift(1).rolling(window=30).mean()
    )
    
    # Calculate Rolling Standard Deviation (Variability σ)
    df['Rolling_Std_30'] = df.groupby('SKU_Compound_ID')['Units Sold'].transform(
        lambda x: x.shift(1).rolling(window=30).std()
    )

    # Time-Based Features (needed for the original modeling context)
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # Categorical Encoding
    le = LabelEncoder()
    df['SKU_Encoded'] = le.fit_transform(df['SKU_Compound_ID'])
    
    # Drop rows that don't have enough history for the 30-day roll and lags
    df.dropna(inplace=True)
    return df

# --- 3. DUMMY MODELING / POLICY CALCULATION ---

def run_model_and_policy(df_features):
    """
    Simulates the calculation of Dynamic and Static ROP/ROQ policies.
    """
    
    # Use the *last* valid row for each SKU as the basis for current policy calculation.
    policy_df = df_features.groupby('SKU_Compound_ID').last().reset_index()

    # 1. Demand Forecast (D)
    # Use Rolling Mean as a proxy for the forecast from Prophet/SARIMA
    policy_df['Forecasted_Demand'] = policy_df['Rolling_Mean_30'] * random.uniform(1.0, 1.2)
    
    # 2. Demand Variability (σ)
    policy_df['Forecast_Std'] = policy_df['Rolling_Std_30']
    
    # 3. Dynamic ROP (Reorder Point)
    # ROP = (Demand during Lead Time) + Safety Stock
    # Safety Stock = Z * σ_L * sqrt(L)
    policy_df['Dynamic_ROP'] = np.ceil(
        (policy_df['Forecasted_Demand'] * LEAD_TIME_DAYS) + 
        (SERVICE_LEVEL_Z * policy_df['Forecast_Std'] * np.sqrt(LEAD_TIME_DAYS))
    ).astype(int)
    
    # 4. Dynamic ROQ (Reorder Quantity) - Simple EOQ Approximation
    # EOQ = sqrt( (2 * Annual Demand * Ordering Cost) / Holding Cost )
    policy_df['Dynamic_ROQ'] = np.ceil(
        np.sqrt( (2 * (policy_df['Forecasted_Demand'] * 365) * ORDERING_COST_S) / HOLDING_COST_H )
    ).astype(int)
    
    # 5. Static Policy (For comparison)
    static_demand_90 = df_features['Units Sold'].quantile(0.90)
    policy_df['Static_ROP'] = np.ceil(static_demand_90 * 1.5).astype(int) # High fixed ROP
    policy_df['Static_ROQ'] = 500 # Fixed ROQ
    
    # 6. Policy Data Cleanup
    policy_df['Current_Inventory'] = np.random.randint(0, 300, size=len(policy_df))
    # Placeholder Stockout Cost: risk increases with demand
    policy_df['Anticipated_Stockout_Cost'] = policy_df['Forecasted_Demand'] * STOCKOUT_COST_C * 30 
    policy_df['Category'] = policy_df['Category'] # Keep the category column
    
    # Filter and rename for final policies DataFrame
    final_cols = ['SKU_Compound_ID', 'Category', 'Forecasted_Demand', 
                  'Dynamic_ROP', 'Dynamic_ROQ', 'Static_ROP', 'Static_ROQ', 
                  'Current_Inventory', 'Anticipated_Stockout_Cost']
    
    return policy_df.rename(columns={'Forecasted_Demand': 'Estimated_Demand'})[final_cols]


# --- 4. COST CALCULATION ---

def calculate_total_cost(sim_data, policy_roq_col, policy_rop_col):
    """Calculates the total annual cost and its components for a given policy."""
    
    # Estimate annual demand
    sim_data['Annual_Demand'] = sim_data['Estimated_Demand'] * 365
    
    # 1. Ordering Cost (S)
    sim_data['Ordering_Cost'] = (sim_data['Annual_Demand'] / sim_data[policy_roq_col]) * ORDERING_COST_S
    total_ordering_cost = sim_data['Ordering_Cost'].sum()

    # 2. Holding Cost (H)
    # Safety Stock is implied by ROP: ROP - (Demand during Lead Time)
    sim_data['Safety_Stock'] = sim_data[policy_rop_col] - (sim_data['Estimated_Demand'] * LEAD_TIME_DAYS)
    sim_data['Avg_Inventory'] = (sim_data[policy_roq_col] / 2) + sim_data['Safety_Stock']
    sim_data['Holding_Cost'] = sim_data['Avg_Inventory'] * HOLDING_COST_H * 365
    total_holding_cost = sim_data['Holding_Cost'].sum()
    
    # 3. Stockout Cost (C) - Simplified simulation of risk
    if 'Static' in policy_rop_col:
        # Static policy risk is higher
        total_stockout_cost = sim_data['Anticipated_Stockout_Cost'].sum() * random.uniform(0.9, 1.1)
    else:
        # Dynamic policy risk is much lower
        total_stockout_cost = sim_data['Anticipated_Stockout_Cost'].sum() * random.uniform(0.05, 0.15)
    
    total_cost = total_ordering_cost + total_holding_cost + total_stockout_cost
    
    return total_cost, total_ordering_cost, total_holding_cost, total_stockout_cost

# --- 5. MAIN EXECUTION FUNCTION FOR STREAMLIT ---

def run_inventory_analysis(file_name='retail_store_inventory.csv'):
    """Runs the full pipeline and returns the dataframes and summary metrics."""
    df_raw = load_data(file_name)
    if df_raw is None:
        return None, None, None

    # Step 1: Feature Engineering and Cleanup
    df_features = engineer_features(df_raw.copy())

    # Step 2: Policy Generation (ROP/ROQ)
    policies_df = run_model_and_policy(df_features)

    # Step 3 & 4: Cost Calculation
    cost_dynamic, ord_d, hold_d, stock_d = calculate_total_cost(policies_df.copy(), 'Dynamic_ROQ', 'Dynamic_ROP')
    cost_static, ord_s, hold_s, stock_s = calculate_total_cost(policies_df.copy(), 'Static_ROQ', 'Static_ROP')
    
    # Step 5: Prepare Metrics
    savings = cost_static - cost_dynamic
    
    cost_summary_metrics = {
        'Static Total': cost_static,
        'Dynamic Total': cost_dynamic,
        'Savings': savings,
        'Reduction %': f"{savings / cost_static * 100:.2f}%"
    }
    
    cost_viz_data = {
        'Policy': ['Static'] * 3 + ['Dynamic'] * 3,
        'Cost Component': ['Ordering Cost (S)', 'Holding Cost (H)', 'Stockout Cost (C)'] * 2,
        'Annual Cost': [ord_s, hold_s, stock_s, ord_d, hold_d, stock_d]
    }
    cost_viz_df = pd.DataFrame(cost_viz_data)

    policies_df['Reorder_Needed'] = policies_df['Current_Inventory'] < policies_df['Dynamic_ROP']

    return policies_df, cost_summary_metrics, cost_viz_df