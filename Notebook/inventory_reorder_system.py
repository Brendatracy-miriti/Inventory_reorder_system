#!/usr/bin/env python
# coding: utf-8

#  # Project Overview
# 
# This capstone project implements a data-driven system to optimize inventory management. It replaces static, rule-based reordering policies with a dynamic, predictive engine that forecasts future demand and calculates the optimal Reorder Point (ROP) and Reorder Quantity (ROQ) for each product.
# 
# The goal is to minimize total inventory costs by balancing the reduction of costly stockouts (lost sales) against the minimization of expensive holding costs (storage, capital tied up).

# # 1. Import Libraries and Loading Data

# In[201]:


# Install Statsmodels
#! pip install statsmodels


# In[260]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from statsmodels.tsa.seasonal import seasonal_decompose


import time
import warnings
warnings.filterwarnings("ignore")


# In[268]:


data = pd.read_csv('retail_store_inventory.csv')
data.head()


# In[269]:


#check the columns and data types
data.info()


# - No missing values in my data

# In[270]:


data.columns


# # 2. EDA

# In[271]:


# Convert 'Date' to datetime and create 'SKU_Compound_ID'
data['Date'] = pd.to_datetime(data['Date'])
data['SKU_Compound_ID'] = data['Store ID'] + '_'+ data['Product ID']


# In[272]:


data.sort_values(by = ['SKU_Compound_ID', 'Date'], inplace=True)


# In[273]:


# Define Inventory Constants
ORDERING_COST = 50.0  # Cost to place an order
HOLDING_COST_PER_UNIT_YEAR = 5.0  # Annual holding cost per unit
STOCKOUT_COST = 10.0 # Cost per unit of unmet demand
DAYS_IN_YEAR = 365.0
SERVICE_LEVEL = 1.645  # Desired service level (95%)
LEAD_TIME_DAYS = 7  # Standard  lead time in days


# Logistics and Inventory Management parameters
# 1. Lead time - How long it takes from placing an order with the supplier until the goods arrive.
# 2. Holding (carrying) cost - Rep the expense of keeping inventory in storage. This includes warehouse, depreciation, insurance and opportunity cost of capital
# 3. Ordering Cost - the fixed cost of placing one order, regardless of quantity. This includes administrative work, shipping, handling, and setup.
# 4. Service Level - is the probability of not running out of stock during the lead time.
# 5. Z-score (from the normal distribution) helps calculate safety stock. 
# 
#       Safety Stock = Z x std demand x square root of Lead time
# 
#   So, a Z = 1.645 gives a 95% confidence that demand during lead time will be covered.

# In[274]:


# Add daily Holding Cost to base data for cost calculations
data['Daily_Holding_Cost'] = HOLDING_COST_PER_UNIT_YEAR / DAYS_IN_YEAR


# Check for seasonality on aggregate demand

# In[275]:


#Aggregate daily demand across all SKUs to check for seasonality
daily_demand = data.groupby('Date')['Units Sold'].sum().reset_index()
daily_demand = daily_demand.set_index('Date').asfreq('D')


# In[276]:


# Plotting the aggregated demand over time
plt.figure(figsize=(14, 6))
sns.lineplot(x='Date', y='Units Sold', data=daily_demand)
plt.title('Aggregate Daily Demand Over Time (Checking for Trend and Seasonality)')
plt.xlabel('Date')
plt.ylabel('Total Units Sold')
plt.grid(True)
plt.show()


# In[277]:


# Decompose the time series (Additive model assumed for demand) Model uses daily data, so a period of 7 (weekly seasonality) is appropriate.
try:
    decomposition = seasonal_decompose(daily_demand['Units Sold'].dropna(), model='additive', period=7)
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(411)
    decomposition.trend.plot(ax=plt.gca())
    plt.title('Trend Component ')
    
    plt.subplot(412)
    decomposition.seasonal.plot(ax=plt.gca())
    plt.title('Seasonal Component (Weekly Pattern)')
    
    plt.subplot(413)
    decomposition.resid.plot(ax=plt.gca())
    plt.title('Residuals')
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Seasonal decomposition failed (likely due to insufficient data length or missing values): {e}")


# # Insight
#  
#  The Seasonal component shows regular weekly peaks, and the Trend component shows the long-term direction of sales. 
#  This confirms Prophet is the right choice.

# # 3. DEMAND FORECASTING ENGINE

# Forecasting with Prophet 
# 
# Prophet (from Facebook/Meta) is specifically designed to forecast time series data without the need for manual feature engineering of trend and seasonality. It internally models the overall trend, yearly seasonality, and weekly seasonality.
# 
# So we will need to format the data for prophet(ds, y) and train a separate model for each unique SKU
# 
# 

# 3.1 Prepare Data and Train Prophet

# In[278]:


#We need 'Date' (as 'ds') and 'Units Sold' (as 'y').
prophet_data = data[['SKU_Compound_ID', 'Date', 'Units Sold']].copy()
prophet_data.rename(columns={'Date': 'ds', 'Units Sold': 'y'}, inplace=True)
prophet_data.head()


# In[280]:


# PREPARE SKUs FOR FORECASTING
SKU_LIST = prophet_data['SKU_Compound_ID'].unique()


# In[281]:


print(f"Starting Prophet training for {len(SKU_LIST)} SKUs...")

# Existing list for ROP/volatility inputs
forecast_results_list = [] 

# === NEW LIST: This will hold the full time-series forecasts for the dashboard ===
all_forecasts_time_series_list = [] 
# =================================================================================

for sku_id in SKU_LIST:
    # Filter training data for the current SKU
    sku_train_data = prophet_data[prophet_data['SKU_Compound_ID'] == sku_id].drop(columns=['SKU_Compound_ID'])
    
    SERVICE_LEVEL_Z = SERVICE_LEVEL
    model = Prophet(
        yearly_seasonality=True, 
        weekly_seasonality=True, 
        daily_seasonality=False,
        interval_width=0.95 
    )
    
    model.fit(sku_train_data)
    
    # Create future dataframe for forecasting
    future = model.make_future_dataframe(periods=LEAD_TIME_DAYS, include_history=False)
    
    # Generate the forecast
    forecast = model.predict(future)
    
    # === NEW: Aggregate the full time-series forecast for the dashboard ===
    forecast['SKU_Compound_ID'] = sku_id
    all_forecasts_time_series_list.append(
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'SKU_Compound_ID']]
    )
    
    # --- ROP/Volatility Extraction ---
    # 1. Mean Demand during Lead Time (DL)
    mean_demand_lead_time = forecast['yhat'].sum()
    
    # 2. Demand Variability (Sigma_L)
    forecast['sigma_daily'] = (forecast['yhat_upper'] - forecast['yhat_lower']) / (2 * 1.96)
    sigma_L = forecast['sigma_daily'].iloc[-1] * np.sqrt(LEAD_TIME_DAYS)
    
    # Append results to existing list
    forecast_results_list.append({
        'SKU_Compound_ID': sku_id,
        'Mean_Demand_Lead_Time': max(0, mean_demand_lead_time), 
        'Last_Observed_Sigma': max(0.1, sigma_L) 
    })


# Final DataFrame for optimization (Your existing ROP components)
future_forecast_data = pd.DataFrame(forecast_results_list)
future_forecast_data['Service_Level_Z'] = SERVICE_LEVEL_Z
future_forecast_data['Lead_Time_Days'] = LEAD_TIME_DAYS

# === NEW DATAFRAME: The Time-Series Forecast Data for the Dashboard ===
df_time_series_forecast = pd.concat(all_forecasts_time_series_list, ignore_index=True) 
# =======================================================================

print("\nProphet Forecasting complete.")
print("The time-series forecast data is now saved in 'df_time_series_forecast'.")


# In[282]:


# Prepare historical data for the dashboard
historical_data = data[['Date', 'SKU_Compound_ID', 'Units Sold']].rename(
    columns={'Units Sold': 'Historical_Demand', 'SKU_Compound_ID': 'SKU'}
)


# In[283]:


# Prepare forecast data with Prophet output columns
forecast_data = df_time_series_forecast.rename(
    columns={
        'ds': 'Date', 
        'yhat': 'Forecast', 
        'yhat_lower': 'Forecast_Lower', 
        'yhat_upper': 'Forecast_Upper', 
        'SKU_Compound_ID': 'SKU'
    }
)[['Date', 'SKU', 'Forecast', 'Forecast_Lower', 'Forecast_Upper']]


# In[ ]:


# Merge historical data (inner join for overlap) and append pure forecast data (outer join)
df_historical_forecast = pd.merge(
    historical_data,
    forecast_data,
    on=['Date', 'SKU'],
    how='outer' 
).sort_values(by=['SKU', 'Date']).reset_index(drop=True)

df_historical_forecast = df_historical_forecast[['Date', 'SKU', 'Historical_Demand', 'Forecast', 'Forecast_Lower', 'Forecast_Upper']]


# In[317]:


# Select one SKU for visualization
SKU_TO_PLOT = 'S001_P0001' 
df_plot = df_historical_forecast[df_historical_forecast['SKU'] == SKU_TO_PLOT].copy()

# Ensure 'Date' is datetime for plotting
df_plot['Date'] = pd.to_datetime(df_plot['Date'])

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_plot, x='Date', y='Historical_Demand', label='Historical Demand', color='darkblue')
sns.lineplot(data=df_plot, x='Date', y='Forecast', label='Prophet Forecast', color='red', linestyle='--')

# Add the confidence interval (Forecast_Lower/Upper)
plt.fill_between(
    df_plot['Date'],
    df_plot['Forecast_Lower'],
    df_plot['Forecast_Upper'],
    color='red',
    alpha=0.2,
    label='95% Confidence Interval'
)

plt.title(f'Demand Forecast vs. History for SKU: {SKU_TO_PLOT}')
plt.xlabel('Date')
plt.ylabel('Units')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visual_forecast_vs_history.png')
plt.show()


# 

# # 4. Inventory Optimization and Cost simulation

# 4.1 Policy Calculation and Simulation set up
# 
#  Here we calculate the dynamic policy(ROP/ROQ) and prepare data for simulation

# In[ ]:


# INVENTORY POLICY CALCULATION (Defines df_inventory_policies) 

# 1. Prepare data with Annual Demand (D_annual) for EOQ calculation
avg_daily_demand = data.groupby('SKU_Compound_ID')['Units Sold'].mean().reset_index()
avg_daily_demand['Annual_Demand'] = avg_daily_demand['Units Sold'] * DAYS_IN_YEAR

policy_data = future_forecast_data.merge(
    avg_daily_demand[['SKU_Compound_ID', 'Annual_Demand']],
    on='SKU_Compound_ID',
    how='left'
)

# --- Dynamic Policy Calculations ---
# ROP Dynamic = Mean Demand Lead Time + Z * Sigma Lead Time
policy_data['Dynamic_ROP'] = (
    policy_data['Mean_Demand_Lead_Time'] + 
    policy_data['Service_Level_Z'] * policy_data['Last_Observed_Sigma']
).round(0)

# ROQ Dynamic (EOQ Formula)
policy_data['Dynamic_ROQ'] = np.sqrt(
    (2 * ORDERING_COST * policy_data['Annual_Demand']) / HOLDING_COST_PER_UNIT_YEAR
).round(0)

# --- Static Policy Calculations ---
# 1. Static ROP: Calculate based on historical average demand plus a simple, fixed safety stock rule.
avg_daily_std = data['Units Sold'].std() # Use overall historical volatility
policy_data['Static_ROP'] = (
    policy_data['Annual_Demand'] / DAYS_IN_YEAR * policy_data['Lead_Time_Days'] +
    2 * avg_daily_std # Example: 2 standard deviations of demand as safety stock
).round(0)

# 2. Static ROQ: Set equal to the baseline EOQ calculated above.
policy_data['Static_ROQ'] = policy_data['Dynamic_ROQ']

# Final DataFrame needed for Script 2
df_inventory_policies = policy_data[[
    'SKU_Compound_ID', 'Static_ROP', 'Dynamic_ROP', 'Static_ROQ', 'Dynamic_ROQ'
]]

print("df_inventory_policies created successfully.")


# ### INSIGHTS 
# 
# 1. Dynamic ROPs vary reasonably across SKUs, reflecting real demand patterns and forecast uncertainty. The 47% higher value vs static ROP suggests the model anticipates higher shortterm demand a positive sign of responsiveness.
# 2. Dynamic ROQs are stable but not uniform, indicating consistent forecasting and lead-time assumptions. If broader variation was expected, consider refining SKU-level scaling.
# 3. Static ROPs are nearly identical across SKUs a rigid approach that ignores demand variability and seasonality. The contrast with dynamic ROP confirms the model adds useful differentiation.
# 4. Static ROQ values are like 8–9× higher than dynamic ROQ, showing clear overstocking from the 60-day rule. The dynamic approach yields leaner, data-driven replenishment aligned to actual demand velocity.
# 
# Basically: 
# - The dynamic model is adaptive, uncertainity aware and is an optimized inventory; 
# 
# while
# - The Static model is uniform, overstocks and has higher holding cost.
# 
# Therefore shifting dynamic ROP/ROQ could significantly reduce excess stock while maintaining service levels.

# In[295]:


# Confirm the actual ratio between Static and Dynamic ROQ
future_forecast_data['ROQ_ratio'] = future_forecast_data['Static_ROQ'] / future_forecast_data['Dynamic_ROQ']
future_forecast_data[['Static_ROQ', 'Dynamic_ROQ', 'ROQ_ratio']].describe()


# In[298]:


# Merge policy data to main historical data for simulation
cost_summary_data = future_forecast_data[['SKU_Compound_ID', 'Estimated_Annual_Demand','Safety_Stock' ,'Dynamic_ROP', 'Dynamic_ROQ', 'Static_ROP', 'Static_ROQ']]. copy()

# Add holding cost information (only needed for calculation, not merge)
cost_summary_data['Daily_Holding_Cost'] = HOLDING_COST_PER_UNIT_YEAR / DAYS_IN_YEAR


# In[299]:


sim_data = pd.merge(data, cost_summary_data.drop(columns=['Estimated_Annual_Demand','Safety_Stock']), 
                    on='SKU_Compound_ID', how='left')


# 4.2 Cost Simulation

# In[308]:


def calculate_total_cost(data_summary, data_sim, ROQ_col):
    """
    Calculates the estimated total annual inventory cost, splitting the calculation
    into annual (Ordering/Holding) and simulation-based (Stockout) components.
    
    data_summary: DataFrame with ONE ROW PER SKU (for Annual Costs)
    data_sim: DataFrame with ALL DAILY RECORDS (for Stockout Cost)
    ROQ_col: The name of the ROQ column to use ('Dynamic_ROQ' or 'Static_ROQ')
    """
    
    # 1. Total Ordering Cost (Annual, Calculated ONCE PER SKU)
    # Estimated Annual Orders = Estimated_Annual_Demand / ROQ
    data_summary['Estimated_Orders'] = data_summary['Estimated_Annual_Demand'] / data_summary[ROQ_col]
    total_ordering_cost = data_summary['Estimated_Orders'].sum() * ORDERING_COST
    
    # 2. Total Holding Cost (Annual, Calculated ONCE PER SKU)
    # Avg Inventory = Safety Stock + ROQ/2
    # Holding Cost = Avg Inventory * Holding Cost per Year
    
    # Use the Annual Holding Cost: Avg Inventory * HOLDING_COST_PER_UNIT_YEAR
    data_summary['Avg_Inventory'] = data_summary['Safety_Stock'] + (data_summary[ROQ_col] / 2)
    total_holding_cost = data_summary['Avg_Inventory'].sum() * HOLDING_COST_PER_UNIT_YEAR
    
    
    # 3. Total Stockout Cost (Simulation-based)
    # Recalculate Stockout based on the merged simulation data
    
    # CRITICAL: We need to use the actual Inventory Level and Units Sold from the SIMULATION.
    # The cost should only be the sum of daily stockout volumes * stockout cost.
    data_sim['Stockout_Volume'] = np.maximum(0, data_sim['Units Sold'] - data_sim['Inventory Level'])
    total_stockout_cost = data_sim['Stockout_Volume'].sum() * STOCKOUT_COST
    
    # If your simulation data spans multiple years (e.g., 2 years), you may need to normalize 
    # the stockout cost to an annual value. Assuming the data spans roughly two years 
    # (2022-01-01 to 2024-01-01), we divide by the number of years.
    
    num_years_simulated = (data_sim['Date'].max() - data_sim['Date'].min()).days / DAYS_IN_YEAR
    
    # Normalize stockout cost to an average annual value
    annualized_stockout_cost = total_stockout_cost / num_years_simulated
    
    return (total_ordering_cost + total_holding_cost + annualized_stockout_cost, 
            total_ordering_cost, 
            total_holding_cost, 
            annualized_stockout_cost)


num_skus = len(cost_summary_data)
print(f"Number of SKUs: {num_skus}. Cost will be calculated over this count, not total daily rows.")

# Calculate Costs for both policies
cost_dynamic, ord_d, hold_d, stock_d = calculate_total_cost(
    cost_summary_data.copy(), # Pass a copy for dynamic
    sim_data.copy(), 
    'Dynamic_ROQ'
)

cost_static, ord_s, hold_s, stock_s = calculate_total_cost(
    cost_summary_data.copy(), # Pass a copy for static
    sim_data.copy(), 
    'Static_ROQ'
)

print("FINAL FINANCIAL SIMULATION RESULTS & BREAKDOWN")

print("STATIC POLICY COST BREAKDOWN (Baseline)")
print(f"Total Cost: ${cost_static:,.0f}")
print(f"  - Ordering Cost (S): ${ord_s:,.0f}")
print(f"  - Holding Cost (H): ${hold_s:,.0f}")
print(f"  - Stockout Cost (C): ${stock_s:,.0f} (Annualized)")

print("DYNAMIC POLICY COST BREAKDOWN (Prophet Optimized)")
print(f"Total Cost: ${cost_dynamic:,.0f}")
print(f"  - Ordering Cost (S): ${ord_d:,.0f}")
print(f"  - Holding Cost (H): ${hold_d:,.0f}")
print(f"  - Stockout Cost (C): ${stock_d:,.0f} (Annualized)")


savings = cost_static - cost_dynamic
print(f"\n**Total Project Savings (Estimated): ${savings:,.0f}**")
print(f"Cost Reduction: {savings / cost_static * 100:.2f}%") # In percentage


# Insight: The costs are now in the tens of millions, and the Dynamic Policy shows a positive cost reduction, primarily due to lower Holding Cost or Stockout Cost. 
# 

# In[309]:


# COST AGGREGATION 

cost_data = {
 'Policy': ['Static', 'Dynamic'],
 'Ordering Cost': [ord_s, ord_d],      
 'Holding Cost': [hold_s, hold_d],     
 'Stockout Cost': [stock_s, stock_d],   
 'Total Cost': [cost_static, cost_dynamic]
}

df_costs = pd.DataFrame(cost_data)

print("df_costs created successfully.")


# In[315]:


# Visualizing Aggregate Cost Comparison 

# Reshape df_costs for easier plotting 
df_costs_melted = df_costs.melt(
    id_vars='Policy', 
    value_vars=['Ordering Cost', 'Holding Cost', 'Stockout Cost'], 
    var_name='Cost Type', 
    value_name='Cost'
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_costs_melted, 
    x='Policy', 
    y='Cost', 
    hue='Cost Type', 
    palette='viridis'
)

# Add Total Cost comparison line
cost_comparison = df_costs[['Policy', 'Total Cost']].set_index('Policy')
plt.plot(cost_comparison.index, cost_comparison['Total Cost'], marker='o', color='red', linewidth=2, label='Total Cost')

plt.title('Total Inventory Cost Breakdown: Static vs. Dynamic Policy')
plt.xlabel('Inventory Policy')
plt.ylabel('Total Annual Cost ($)')
plt.legend(loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('visual_cost_comparison.png')
plt.show()


# In[312]:


# Policy Comparison & Recommendation Output 

# A. Per-SKU ROP/ROQ Recommendation & Comparison (for actionability)
df_policy_comparison_sku = df_inventory_policies.rename(
    columns={
        'SKU_Compound_ID': 'SKU',
        'Static_ROP': 'Static_Reorder_Point',
        'Dynamic_ROP': 'Dynamic_Reorder_Point',
        'Dynamic_ROQ': 'Recommended_Order_Qty'
    }
)[['SKU', 'Static_Reorder_Point', 'Dynamic_Reorder_Point', 'Recommended_Order_Qty']]

# B. Aggregate Cost Comparison (for Executive Summary/Savings)
cost_static_agg = df_costs[df_costs['Policy'] == 'Static']['Total Cost'].iloc[0]
cost_dynamic_agg = df_costs[df_costs['Policy'] == 'Dynamic']['Total Cost'].iloc[0]

# Calculate savings percentage
savings_percentage_agg = ((cost_static_agg - cost_dynamic_agg) / cost_static_agg) * 100

df_aggregate_cost_comparison = pd.DataFrame({
    'Metric': ['Aggregate Total Cost Comparison'],
    'Static_Total_Cost': [cost_static_agg.round(2)],
    'Dynamic_Total_Cost': [cost_dynamic_agg.round(2)],
    'Savings_Percentage': [f"{savings_percentage_agg.round(2)}%"]
})

print("DataFrame 2A (Per-SKU ROP/ROQ) and 2B (Aggregate Costs) created.")
# print(df_policy_comparison_sku.head())
# print(df_aggregate_cost_comparison)


# In[318]:


#  Current Status Data 

# 1. Get the most recent stock level (Current_Stock)
last_date = data['Date'].max()
df_current_stock = data[data['Date'] == last_date].rename(
    columns={'Inventory Level': 'Current_Stock', 'SKU_Compound_ID': 'SKU'}
)[['SKU', 'Current_Stock']]

# 2. Merge with Policy Data to get Dynamic_ROP
df_status = pd.merge(
    df_current_stock,
    df_inventory_policies[['SKU_Compound_ID', 'Dynamic_ROP']].rename(columns={'SKU_Compound_ID': 'SKU'}),
    on='SKU',
    how='left'
)

# 3. Merge with ROP components (future_forecast_data) to get Mean_Demand_Lead_Time (Avg_LTD)
df_status = pd.merge(
    df_status,
    future_forecast_data.rename(
        columns={'SKU_Compound_ID': 'SKU', 'Mean_Demand_Lead_Time': 'Avg_LTD'}
    )[['SKU', 'Avg_LTD']],
    on='SKU',
    how='left'
)

# 4. Calculate Dynamic Safety Stock
# Safety Stock = Dynamic ROP - Mean Demand Lead Time
df_status['Safety_Stock'] = (df_status['Dynamic_ROP'] - df_status['Avg_LTD']).round(0).astype(int)

# 5. Determine Actionable Status
def determine_status(row):
    if row['Current_Stock'] <= row['Safety_Stock']:
        return 'Critical Stock'
    elif row['Current_Stock'] < row['Dynamic_ROP']:
        return 'Low Stock (Reorder Now)'
    else:
        return 'Healthy Stock'

df_status['Status'] = df_status.apply(determine_status, axis=1)

# Final DataFrame structure (DataFrame 3)
df_current_status = df_status[['SKU', 'Current_Stock', 'Safety_Stock', 'Status']]

print("DataFrame 3: Current Status Data created.")
print(df_current_status.head())


# In[ ]:


# Visualizing Current Inventory Status
plt.figure(figsize=(8, 6))
sns.countplot(
    data=df_current_status, 
    x='Status', 
    order=['Healthy Stock', 'Low Stock (Reorder Now)', 'Critical Stock'],
    palette={'Healthy Stock': 'green', 'Low Stock (Reorder Now)': 'orange', 'Critical Stock': 'red'}
)

plt.title('Current Inventory Portfolio Health Check')
plt.xlabel('Inventory Status')
plt.ylabel('Number of SKUs')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('visual_inventory_status.png')
plt.show()


# The Dynamic Policy achieves positive savings by making a strategic cost trade-off: it significantly reduces expensive Holding Cost (less stored inventory) by increasing lower-impact Ordering Cost (more frequent orders).
# 
# This shift results in a lower Total Cost while simultaneously improving inventory efficiency and maintaining service levels (low Stockout Cost).

# In[319]:


# Generate the dataframes for dashboard output
df_time_series_forecast.head()


# In[320]:


df_inventory_policies.head()


# In[ ]:


get_ipython().system('jupyter nbconvert --to script inventory_reorder_system.ipynb')

