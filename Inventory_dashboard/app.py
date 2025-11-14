import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import datetime

# --- Configuration ---
# Setting a fixed current date for simulation consistency
TODAY = datetime.date(2025, 11, 14) 

st.set_page_config(layout="wide", page_title="Dynamic Inventory Management System", initial_sidebar_state="expanded")

# --- 1. DATA LOADING AND CLEANING (SIMULATED FOR 100 SKUs) ---
@st.cache_data
def load_data():
    # --- Simulation Parameters: Back to 100 SKUs / Single Store (S001) ---
    N_SKUS = 100  # Total number of unique SKUs
    
    # Generate the 100 unique compound SKUs in the S001_PXXXX format
    all_skus = [f'S001_P{i:04d}' for i in range(1, N_SKUS + 1)]
    
    # Assign categories equally (25 SKUs per category)
    categories = (['Toys'] * 25) + (['Clothing'] * 25) + (['Furniture'] * 25) + (['Electronics'] * 25)
    
    # Check to ensure we have 100 entries for everything
    if len(all_skus) != N_SKUS or len(categories) != N_SKUS:
        st.error("Data generation error: Mismatched list lengths.")
        st.stop()
        
    # Generate random data for 100 SKUs
    current_stock_sim = np.random.randint(50, 500, N_SKUS)

    # 1. Current Status Data (df_current_status)
    df_status = pd.DataFrame({
        'SKU': all_skus, # Represents SKU_Compound_ID (e.g., S001_P0001)
        'Category': categories,
        'Current_Stock': current_stock_sim,
        'Safety_Stock': np.random.randint(400, 600, N_SKUS),
        'Stock_Value': np.random.randint(500, 1500, N_SKUS) * 10,
        'Status': np.select(
            [
                current_stock_sim < 150,
                (current_stock_sim >= 150) & (current_stock_sim < 400)
            ],
            [
                'Critical Stock', 
                'Low Stock (Reorder Now)'
            ],
            default='Healthy Stock'
        )
    })
    
    # 2. Policy Comparison & Recommendation Output (df_policies)
    df_policies = pd.DataFrame({
        'SKU_Compound_ID': all_skus, 
        'Category': categories,
        'Static_Total_Cost': np.random.randint(100000, 300000, N_SKUS),
        'Dynamic_Total_Cost': np.random.randint(50000, 150000, N_SKUS),
        'Dynamic_ROP': np.random.randint(800, 1500, N_SKUS),
        'Static_ROP': np.random.randint(800, 1500, N_SKUS),
        'Dynamic_ROQ': np.random.randint(100, 500, N_SKUS)
    })
    df_policies['Cost_Savings'] = df_policies['Static_Total_Cost'] - df_policies['Dynamic_Total_Cost']
    df_policies['Savings_Percentage'] = (df_policies['Cost_Savings'] / df_policies['Static_Total_Cost']) * 100
    
    # 3. Time Series Forecast Data (df_time_series_forecast)
    FORECAST_DAYS = 30
    all_forecast_data = []
    
    for sku in all_skus:
        dates_history = pd.date_range(start=TODAY - datetime.timedelta(days=180), end=TODAY - datetime.timedelta(days=1), freq='D')
        dates_future = pd.date_range(start=TODAY, periods=FORECAST_DAYS, freq='D')
        
        # History Data
        df_history = pd.DataFrame({
            'Date': dates_history,
            'Forecast': np.random.randint(50, 400, len(dates_history)),
            'SKU_Compound_ID': sku,
            'Type': 'History'
        })
        
        # Forecast Data
        forecast_mean = np.random.randint(400, 600, FORECAST_DAYS)
        df_future = pd.DataFrame({
            'Date': dates_future,
            'Forecast': forecast_mean,
            'yhat_lower': forecast_mean * 0.9,
            'yhat_upper': forecast_mean * 1.1,
            'SKU_Compound_ID': sku,
            'Type': 'Forecast'
        })
        all_forecast_data.append(pd.concat([df_history, df_future]))
        
    df_ts_plot = pd.concat(all_forecast_data)

    return df_status, df_ts_plot, df_policies

df_status, df_ts_plot, df_policies = load_data()


# --- 2. STREAMLIT APP LAYOUT & FILTERS ---

st.title("📈 Dynamic Inventory Management System")

# --- SIDEBAR FOR INTERACTIVITY ---
with st.sidebar:
    st.header("Dashboard Filters")
    
    # ONLY Category Filter Remains (Store ID filter removed)
    all_categories = ['All Categories'] + df_status['Category'].unique().tolist()
    selected_category = st.selectbox(
        "Filter by Product Category:",
        options=all_categories,
        index=0 
    )

    # Filter dataframes based ONLY on Category selection
    df_filtered = df_status.copy()
        
    if selected_category != 'All Categories':
        df_filtered = df_filtered[df_filtered['Category'] == selected_category]
        
    # Get the list of SKUs after filtering
    filtered_skus = df_filtered['SKU'].unique()
    
    # 2. SKU Selector
    if len(filtered_skus) > 0:
        selected_sku = st.selectbox(
            f"Select Specific SKU_Compound_ID ({len(filtered_skus)} available):",
            options=filtered_skus,
            index=0 
        )
    else:
        selected_sku = None
        st.warning("No SKUs match the current filter combination.")

    st.markdown("---")
    forecast_end = TODAY + datetime.timedelta(days=30)
    st.info(f"Forecasting 📅 **{TODAY.strftime('%b %d, %Y')}** to **{forecast_end.strftime('%b %d, %Y')}**")


# --- Main Data Filtering for Charts and KPIs ---

if selected_sku is None:
    st.stop() 

# We use the filtered list of SKUs (filtered_skus) to filter all dataframes
df_filtered_status = df_status[df_status['SKU'].isin(filtered_skus)]
df_filtered_policies = df_policies[df_policies['SKU_Compound_ID'].isin(filtered_skus)]

# Data for the single selected SKU
if selected_sku in df_status['SKU'].values:
    # Use 'SKU' column for df_status lookup
    df_sku_status = df_status[df_status['SKU'] == selected_sku].iloc[0]
    # Use 'SKU_Compound_ID' for df_policies and df_ts_plot lookups
    df_sku_forecast = df_ts_plot[df_ts_plot['SKU_Compound_ID'] == selected_sku].copy()
    df_sku_policy = df_policies[df_policies['SKU_Compound_ID'] == selected_sku].iloc[0]
else:
    st.error("Error retrieving data for selected SKU.")
    st.stop()
    
# =================================================================
# ## 1. Inventory Status Overview (Portfolio & SKU)
# =================================================================
st.header("1. Inventory Portfolio Health Check")

col1, col2, col3, col4 = st.columns(4)

# KPI 1: Total Stock-Outs
critical_count = df_filtered_status[df_filtered_status['Status'] == 'Critical Stock'].shape[0]
col1.metric("Critical Stock SKUs", f"{critical_count}", 
            delta=f"{(critical_count / len(filtered_skus) * 100):.1f}% of SKUs", delta_color="inverse")

# KPI 2: Total Inventory Value (Filtered Portfolio)
total_value = df_filtered_status['Stock_Value'].sum()
col2.metric(f"Total Value ({selected_category})", f"${total_value:,.0f}") 

# KPI 3: Avg. Cost Savings (Filtered Portfolio)
avg_savings = df_filtered_policies['Savings_Percentage'].mean()
col3.metric("Avg. Dynamic Policy Savings", f"{avg_savings:,.1f}%")

# KPI 4: Service Level (Simulated)
col4.metric("Service Level (Last Month)", "96.5%", delta="0.2%", delta_color="normal")

st.markdown("---")

# **Inventory Distribution Chart (Portfolio Level)**
st.subheader(f"SKU Status Distribution: **{selected_category}**") 

fig_status_dist = px.bar(
    df_filtered_status.groupby('Status').size().reset_index(name='Count'),
    x='Status',
    y='Count',
    color='Status',
    color_discrete_map={'Critical Stock': 'red', 'Low Stock (Reorder Now)': 'orange', 'Healthy Stock': 'green'},
    text='Count'
)
fig_status_dist.update_layout(xaxis_title="", yaxis_title="Number of SKUs")
st.plotly_chart(fig_status_dist, use_container_width=True)


# =================================================================
# ## 2. Demand Forecast & Visualization (Real-Time 30-Day View)
# =================================================================
st.header("2. Real-Time Demand Forecast Analysis")
st.subheader(f"30-Day Forecast for **{selected_sku}**")
# 

# **Interactive Forecast Chart (History + 30-Day Prediction)**
fig_forecast = go.Figure()

# History
df_history_plot = df_sku_forecast[df_sku_forecast['Type'] == 'History']
fig_forecast.add_trace(go.Scatter(
    x=df_history_plot['Date'],
    y=df_history_plot['Forecast'],
    mode='lines',
    line=dict(color='darkblue', width=2),
    name='Historical Demand'
))

# Confidence Interval (Shaded Area - Future Only)
df_future_plot = df_sku_forecast[df_sku_forecast['Type'] == 'Forecast']
fig_forecast.add_trace(go.Scatter(
    x=df_future_plot['Date'],
    y=df_future_plot['yhat_upper'],
    line=dict(color='rgba(255, 0, 0, 0)'),
    showlegend=False,
    name='Upper Bound'
))
fig_forecast.add_trace(go.Scatter(
    x=df_future_plot['Date'],
    y=df_future_plot['yhat_lower'],
    fill='tonexty',
    fillcolor='rgba(255, 0, 0, 0.1)',
    line=dict(color='rgba(255, 0, 0, 0)'),
    name='95% CI'
))

# Prophet Forecast (Mean - Future Only)
fig_forecast.add_trace(go.Scatter(
    x=df_future_plot['Date'],
    y=df_future_plot['Forecast'],
    mode='lines',
    line=dict(color='red', dash='dash', width=3),
    name='30-Day Forecast'
))

# Vertical line for Current Date (Real-Time Demarcation)
y_max = df_sku_forecast['Forecast'].max() * 1.1 
fig_forecast.add_trace(go.Scatter(
    x=[TODAY, TODAY], 
    y=[0, y_max],
    mode='lines',
    line=dict(color='orange', dash='dot', width=2),
    name='Today'
))

fig_forecast.update_layout(
    title='Demand Trend: History vs. 30-Day Forecast',
    xaxis_title='Date',
    yaxis_title='Units',
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig_forecast, use_container_width=True)


# =================================================================
# ## 3. Policy Comparison & Recommendation
# =================================================================
st.header("3. Dynamic vs. Static Policy Optimization")
st.subheader(f"Cost & Reorder Analysis for **{selected_sku}**")

# **Policy Comparison & Cost Savings KPI**
total_static_cost = df_sku_policy['Static_Total_Cost']
total_dynamic_cost = df_sku_policy['Dynamic_Total_Cost']
cost_saving = df_sku_policy['Cost_Savings']
saving_pct = df_sku_policy['Savings_Percentage']

st.metric("Estimated Cost Savings with Dynamic Policy", f"${cost_saving:,.0f}", 
          delta=f"{saving_pct:,.1f}% Reduction in Total Cost")

col_rec1, col_rec2, col_rec3 = st.columns(3)

# Reorder Recommendation
reorder_qty = df_sku_policy['Dynamic_ROQ']
col_rec1.metric("Recommended Order Quantity (ROQ)", f"{int(reorder_qty)} Units")

# Dynamic vs Static ROP
dynamic_rop = df_sku_policy['Dynamic_ROP']
static_rop = df_sku_policy['Static_ROP']
col_rec2.metric("Dynamic Reorder Point (ROP)", f"{int(dynamic_rop)} Units")
col_rec3.metric("Static Reorder Point (ROP)", f"{int(static_rop)} Units", 
                delta=f"{int(dynamic_rop - static_rop)} Unit Difference", delta_color="off")

# **Recommendation Action Block**
current_stock = df_sku_status['Current_Stock']
if current_stock < dynamic_rop:
    st.error(f"⚠️ **IMMEDIATE ACTION REQUIRED:** Current Stock (**{current_stock}**) is below the Dynamic ROP (**{int(dynamic_rop)}**). Place a PO for **{int(reorder_qty)}** units.")
else:
    st.success(f"✅ **Stock is Healthy:** Current Stock (**{current_stock}**) is above the Dynamic ROP (**{int(dynamic_rop)}**). No immediate action needed.")

st.markdown("---")

# **Cost Savings by Category (Bar Chart)**
st.subheader(f"Policy Performance: Average for Portfolio: {selected_category}") 

avg_costs = df_filtered_policies[['Static_Total_Cost', 'Dynamic_Total_Cost']].mean().reset_index()
avg_costs.columns = ['Policy', 'Average Cost']
avg_costs['Policy'] = avg_costs['Policy'].str.replace('_Total_Cost', '')

fig_avg_costs = px.bar(
    avg_costs,
    x='Policy',
    y='Average Cost',
    title='Average Total Inventory Cost Comparison (Filtered Portfolio)',
    color='Policy',
    color_discrete_map={'Static': '#395A8D', 'Dynamic': '#0E8388'},
    text='Average Cost'
)
fig_avg_costs.update_traces(texttemplate='$%{y:,.0f}', textposition='outside')
fig_avg_costs.update_layout(yaxis_title='Average Total Annual Cost ($)')
st.plotly_chart(fig_avg_costs, use_container_width=True)

st.markdown("""
<hr>
<p style='text-align:center; color:gray'>
Built by <b>Tracy Miriti</b> | Streamlit + Plotly | Dynamic Inventory Management System
</p>
""", unsafe_allow_html=True)