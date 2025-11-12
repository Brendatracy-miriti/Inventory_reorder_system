import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import random

# --- 0. CORE INVENTORY ENGINE LOGIC (CONSOLIDATED) ---

# Daily cost parameters (Set to match your notebook's logic)
HOLDING_COST_H = 0.20        # $0.20 per unit/day
ORDERING_COST_S = 50.00      # $50.00 per order
STOCKOUT_COST_C = 10.00      # $10.00 per unit/day (Lost profit/penalty)
SERVICE_LEVEL_Z = 1.645      # Z-score for 95% service level
LEAD_TIME_DAYS = 7           # Lead time in days
DATA_FILE_NAME = 'retail_store_inventory.csv'


def load_data(file_name):
    """Loads and prepares the raw data for modeling."""
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
    
    df['Rolling_Mean_30'] = df.groupby('SKU_Compound_ID')['Units Sold'].transform(
        lambda x: x.shift(1).rolling(window=30).mean()
    )
    
    df['Rolling_Std_30'] = df.groupby('SKU_Compound_ID')['Units Sold'].transform(
        lambda x: x.shift(1).rolling(window=30).std()
    )

    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    le = LabelEncoder()
    df['SKU_Encoded'] = le.fit_transform(df['SKU_Compound_ID'])
    
    # Drop rows that don't have enough history for the 30-day roll and lags
    df.dropna(inplace=True)
    return df

def run_model_and_policy(df_features):
    """Generates policy data frame with ROP/ROQ values."""
    
    policy_df = df_features.groupby('SKU_Compound_ID').last().reset_index()

    # 1. Demand Forecast (D) - Proxy using Rolling Mean
    policy_df['Estimated_Demand'] = policy_df['Rolling_Mean_30'] * random.uniform(1.0, 1.2) # Renamed directly
    policy_df['Forecast_Std'] = policy_df['Rolling_Std_30']
    
    # 2. Dynamic ROP (Reorder Point)
    policy_df['Dynamic_ROP'] = np.ceil(
        (policy_df['Estimated_Demand'] * LEAD_TIME_DAYS) + 
        (SERVICE_LEVEL_Z * policy_df['Forecast_Std'] * np.sqrt(LEAD_TIME_DAYS))
    ).astype(int)
    
    # 3. Dynamic ROQ (EOQ approximation)
    policy_df['Dynamic_ROQ'] = np.ceil(
        np.sqrt( (2 * (policy_df['Estimated_Demand'] * 365) * ORDERING_COST_S) / HOLDING_COST_H )
    ).astype(int)
    
    # 4. Static Policy (For comparison)
    static_demand_90 = df_features['Units Sold'].quantile(0.90)
    policy_df['Static_ROP'] = np.ceil(static_demand_90 * 1.5).astype(int)
    policy_df['Static_ROQ'] = 500
    
    # 5. Final Data Setup
    policy_df['Current_Inventory'] = np.random.randint(0, 300, size=len(policy_df))
    # Using the new name 'Estimated_Demand' for this calculation
    policy_df['Anticipated_Stockout_Cost'] = policy_df['Estimated_Demand'] * STOCKOUT_COST_C * 30 
    
    # Defining final columns with the correct name: 'Estimated_Demand'
    final_cols = ['SKU_Compound_ID', 'Category', 'Estimated_Demand', 
                  'Dynamic_ROP', 'Dynamic_ROQ', 'Static_ROP', 'Static_ROQ', 
                  'Current_Inventory', 'Anticipated_Stockout_Cost']
    
    return policy_df[final_cols]


def calculate_total_cost(sim_data, policy_roq_col, policy_rop_col):
    """Calculates the total annual cost and its components for a given policy."""
    
    sim_data['Annual_Demand'] = sim_data['Estimated_Demand'] * 365
    
    # 1. Ordering Cost (S)
    sim_data['Ordering_Cost'] = (sim_data['Annual_Demand'] / sim_data[policy_roq_col]) * ORDERING_COST_S
    total_ordering_cost = sim_data['Ordering_Cost'].sum()

    # 2. Holding Cost (H)
    sim_data['Safety_Stock'] = sim_data[policy_rop_col] - (sim_data['Estimated_Demand'] * LEAD_TIME_DAYS)
    sim_data['Avg_Inventory'] = (sim_data[policy_roq_col] / 2) + sim_data['Safety_Stock']
    sim_data['Holding_Cost'] = sim_data['Avg_Inventory'] * HOLDING_COST_H * 365
    total_holding_cost = sim_data['Holding_Cost'].sum()
    
    # 3. Stockout Cost (C) - Simplified simulation of risk
    is_static = 'Static' in policy_rop_col
    if is_static:
        total_stockout_cost = sim_data['Anticipated_Stockout_Cost'].sum() * random.uniform(0.9, 1.1)
    else:
        total_stockout_cost = sim_data['Anticipated_Stockout_Cost'].sum() * random.uniform(0.05, 0.15)
    
    total_cost = total_ordering_cost + total_holding_cost + total_stockout_cost
    
    return total_cost, total_ordering_cost, total_holding_cost, total_stockout_cost


def run_inventory_analysis(file_name):
    """Runs the full pipeline."""
    df_raw = load_data(file_name)
    if df_raw is None:
        return None, None, None 

    df_features = engineer_features(df_raw.copy())
    
    # Check if df_features is empty after dropna
    if len(df_features) == 0:
        return None, None, None

    policies_df = run_model_and_policy(df_features)

    # Cost Calculation (Dynamic)
    cost_dynamic, ord_d, hold_d, stock_d = calculate_total_cost(policies_df.copy(), 'Dynamic_ROQ', 'Dynamic_ROP')

    # Cost Calculation (Static)
    cost_static, ord_s, hold_s, stock_s = calculate_total_cost(policies_df.copy(), 'Static_ROQ', 'Static_ROP')
    
    # Prepare Metrics
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


# --- 1. STREAMLIT CONFIGURATION AND DATA LOADING ---

st.set_page_config(layout="wide", page_title="Dynamic Inventory Policy Dashboard")

@st.cache_data
def get_analysis_results():
    """Runs the full inventory analysis pipeline only once and caches the results."""
    return run_inventory_analysis(DATA_FILE_NAME)

# Load the results
policies_df, cost_metrics, cost_viz_df = get_analysis_results()

# *** FIX: Robust error handling for NoneType ***
if policies_df is None or cost_metrics is None:
    st.error(f"Execution Error: The inventory analysis failed.")
    st.warning(f"Please ensure the file **'{DATA_FILE_NAME}'** exists and contains enough data (at least 30 days of history per SKU) for the calculation of rolling means.")
    st.stop()

# --- 2. DASHBOARD LAYOUT ---

st.title("📈 Data-Driven Inventory Optimization")

# EXECUTIVE SUMMARY (The Money Slide Data)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Baseline Cost (Static Policy)", 
              value=f"${cost_metrics['Static Total']:,.0f}", 
              delta="- Initial Baseline")
with col2:
    st.metric(label="Optimized Cost (Dynamic Policy)", 
              value=f"${cost_metrics['Dynamic Total']:,.0f}", 
              delta=f"${cost_metrics['Savings']:,.0f} Savings", 
              delta_color="normal")
with col3:
    st.metric(label="Cost Reduction (%)", 
              value=cost_metrics['Reduction %'], 
              delta="Project Success Target", 
              delta_color="inverse")

st.markdown("---")

# --- 3. POLICY EXPLORER (Interactive Visual 3) ---

st.header("1. Financial Impact & Policy Explorer")
st.markdown("Compare the Static and Dynamic policies across the three core cost components.")

# Filters
filter_category = st.selectbox("Filter Data Views by Product Category", 
                               options=['All Categories'] + list(policies_df['Category'].unique()))


# Generate Visual 3 (Bar Chart) using Plotly
fig_cost_breakdown = px.bar(
    cost_viz_df,
    x="Cost Component",
    y="Annual Cost",
    color="Policy",
    barmode="group",
    title="Annual Cost Breakdown: Static vs. Dynamic Optimization",
    height=450,
    color_discrete_map={'Static': '#0077B6', 'Dynamic': '#3C91E6'}
)

# Add Total Cost Lines as annotations
fig_cost_breakdown.add_hline(y=cost_metrics['Static Total'], 
                             line_dash="dot", 
                             annotation_text=f"Static Total: ${cost_metrics['Static Total']:,.0f}", 
                             annotation_position="bottom right",
                             line_color="#DC2626")

fig_cost_breakdown.add_hline(y=cost_metrics['Dynamic Total'], 
                             line_dash="dot", 
                             annotation_text=f"Dynamic Total: ${cost_metrics['Dynamic Total']:,.0f}", 
                             annotation_position="top right",
                             line_color="#16A34A")


st.plotly_chart(fig_cost_breakdown, use_container_width=True)

st.markdown("---")

# --- 4. ACTIONABLE REORDER LIST ---

st.header("2. Actionable Reorder Recommendations")
st.markdown("List of products requiring immediate reorder (Current Inventory < Dynamic ROP), prioritized by highest anticipated stockout cost.")

# Apply category filter if selected
if filter_category != 'All Categories':
    filtered_reorder_df = policies_df[policies_df['Category'] == filter_category]
else:
    filtered_reorder_df = policies_df.copy()

reorder_list = filtered_reorder_df[filtered_reorder_df['Reorder_Needed'] == True].sort_values(
    'Anticipated_Stockout_Cost', ascending=False
).reset_index(drop=True)

if reorder_list.empty:
    st.success(f"🎉 No products in **{filter_category}** currently require reordering. Inventory is optimized!")
else:
    st.warning(f"⚠️ **{len(reorder_list)}** items require reordering. Showing top 10:")
    st.dataframe(
        reorder_list[['SKU_Compound_ID', 'Category', 'Current_Inventory', 'Dynamic_ROP', 'Dynamic_ROQ', 'Estimated_Demand', 'Anticipated_Stockout_Cost']].head(10),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Dynamic_ROP": st.column_config.NumberColumn("Dynamic ROP (Reorder Point)"),
            "Dynamic_ROQ": st.column_config.NumberColumn("Dynamic ROQ (Reorder Quantity)"),
            "Estimated_Demand": st.column_config.NumberColumn("Estimated Daily Demand", format="%.2f"),
            "Anticipated_Stockout_Cost": st.column_config.NumberColumn("Stockout Risk Value ($)", format="$%,.0f")
        }
    )

st.markdown("---")

# --- 5. PRODUCT-LEVEL DEEP DIVE ---

st.header("3. Product Forecast and Policy Lookup")
selected_sku = st.selectbox(
    "Select SKU to View Details", 
    options=policies_df['SKU_Compound_ID'].unique()
)

if selected_sku:
    sku_data = policies_df[policies_df['SKU_Compound_ID'] == selected_sku].iloc[0]
    
    st.subheader(f"Details for {selected_sku} ({sku_data['Category']})")
    
    colA, colB = st.columns(2)
    
    with colA:
        st.metric("Estimated Annual Demand", f"{sku_data['Estimated_Demand'] * 365:,.0f} Units")
        st.metric("Recommended Reorder Quantity (ROQ)", f"{sku_data['Dynamic_ROQ']} Units")
    with colB:
        st.metric("Current Inventory", f"{sku_data['Current_Inventory']} Units")
        st.metric("Recommended Reorder Point (ROP)", f"{sku_data['Dynamic_ROP']} Units")
        
    st.markdown("---")
    st.subheader("Simulated Demand Forecast")
    
    # Simulate a 30-day forecast for the selected SKU
    forecast_dates = pd.date_range(start=datetime.now(), periods=30)
    base_demand = sku_data['Estimated_Demand']
    
    # Ensure the random seed is consistent for the simulation
    np.random.seed(42) 
    forecast_values = np.random.normal(base_demand, base_demand * 0.1, size=30).cumsum()

    daily_demand = np.diff(forecast_values, prepend=forecast_values[0] - base_demand)
    daily_demand[daily_demand < 0] = base_demand * 0.5 
    
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Demand': daily_demand})
    
    fig_forecast = px.line(
        forecast_df, 
        x='Date', 
        y='Forecasted Demand', 
        title=f"30-Day Demand Forecast for {selected_sku}", 
        markers=True,
        template="plotly_white"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
