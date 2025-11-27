import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    layout="wide",
    page_title="Inventory Reorder System",
    page_icon="📦",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem;}
    .sub-header {font-size: 1.2rem; color: #555; margin-bottom: 0.5rem;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
    .urgent-alert {background-color: #ffebee; border-left: 5px solid #f44336; padding: 1rem; margin: 1rem 0;}
    .success-alert {background-color: #e8f5e9; border-left: 5px solid #4caf50; padding: 1rem; margin: 1rem 0;}
    .warning-alert {background-color: #fff3e0; border-left: 5px solid #ff9800; padding: 1rem; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# ==================== DATA LOADING ====================
@st.cache_data
def load_inventory_data():
    """Load real inventory data from CSV files"""
    try:
        # Load master inventory table (from notebook analysis)
        master_inventory = pd.read_csv('../Data/master_inventory_policy.csv')
        
        # Load historical sales data
        historical_data = pd.read_csv('../Data/retail_store_inventory.csv')
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
        
        return master_inventory, historical_data
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.stop()

@st.cache_data
def prepare_dashboard_data(master_inventory, historical_data):
    """Prepare and enrich data for dashboard"""
    
    # Add category information (extract from SKU or assign based on product ID)
    def assign_category(sku):
        product_id = int(sku.split('_')[1][1:])
        if product_id <= 5:
            return 'Electronics'
        elif product_id <= 10:
            return 'Clothing'
        elif product_id <= 15:
            return 'Furniture'
        else:
            return 'Toys'
    
    master_inventory['Category'] = master_inventory['SKU_ID'].apply(assign_category)
    
    # Add current inventory simulation (replace with real data)
    np.random.seed(42)
    master_inventory['Current_Inventory'] = (
        master_inventory['Reorder_Point'] * np.random.uniform(0.5, 1.5, len(master_inventory))
    ).round(0).astype(int)
    
    # Calculate key metrics
    master_inventory['Reorder_Needed'] = (
        master_inventory['Current_Inventory'] < master_inventory['Reorder_Point']
    )
    
    master_inventory['Days_Stock_Remaining'] = (
        master_inventory['Current_Inventory'] / master_inventory['Avg_Daily_Demand']
    ).round(1)
    
    master_inventory['Stock_Status'] = pd.cut(
        master_inventory['Days_Stock_Remaining'],
        bins=[-np.inf, 3, 7, np.inf],
        labels=['Critical', 'Low', 'Healthy']
    )
    
    # Calculate inventory value (assuming unit price)
    master_inventory['Inventory_Value'] = master_inventory['Current_Inventory'] * 50
    
    return master_inventory, historical_data

# Load data
master_inventory, historical_data = load_inventory_data()
master_inventory, historical_data = prepare_dashboard_data(master_inventory, historical_data)

# ==================== SIDEBAR FILTERS ====================
st.sidebar.image("https://img.icons8.com/fluency/96/warehouse.png", width=80)
st.sidebar.title(" Dashboard Controls")

# Date selector
today = datetime.now().date()
st.sidebar.subheader(" Date Range")
date_range = st.sidebar.date_input(
    "Select Period",
    value=(today - timedelta(days=30), today),
    max_value=today
)

# Category filter
st.sidebar.subheader(" Filters")
categories = ['All Categories'] + sorted(master_inventory['Category'].unique().tolist())
selected_category = st.sidebar.selectbox("Category", categories)

# Stock status filter
status_options = ['All Status', 'Critical', 'Low', 'Healthy']
selected_status = st.sidebar.selectbox("Stock Status", status_options)

# Reorder filter
show_reorder_only = st.sidebar.checkbox("Show Only Reorder Needed", value=False)

# Apply filters
filtered_data = master_inventory.copy()

if selected_category != 'All Categories':
    filtered_data = filtered_data[filtered_data['Category'] == selected_category]

if selected_status != 'All Status':
    filtered_data = filtered_data[filtered_data['Stock_Status'] == selected_status]

if show_reorder_only:
    filtered_data = filtered_data[filtered_data['Reorder_Needed'] == True]

st.sidebar.markdown("---")
st.sidebar.info(f" **{len(filtered_data)}** SKUs displayed")

# ==================== MAIN DASHBOARD ====================

# Header
st.markdown('<p class="main-header"> Inventory Reorder Management System</p>', unsafe_allow_html=True)
st.markdown(f"**Last Updated:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")

# ==================== KEY METRICS ROW ====================
st.markdown("### 📊 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

# KPI 1: Total SKUs
with col1:
    total_skus = len(filtered_data)
    st.metric("Total SKUs", f"{total_skus}")

# KPI 2: Reorder Needed
with col2:
    reorder_count = filtered_data['Reorder_Needed'].sum()
    reorder_pct = (reorder_count / total_skus * 100) if total_skus > 0 else 0
    st.metric("Reorder Needed", f"{reorder_count}", delta=f"{reorder_pct:.1f}%", delta_color="inverse")

# KPI 3: Critical Stock
with col3:
    critical_count = (filtered_data['Stock_Status'] == 'Critical').sum()
    st.metric("Critical Stock", f"{critical_count}", delta="Urgent", delta_color="inverse")

# KPI 4: Total Inventory Value
with col4:
    total_value = filtered_data['Inventory_Value'].sum()
    st.metric("Inventory Value", f"${total_value:,.0f}")

# KPI 5: Avg Days Stock
with col5:
    avg_days = filtered_data['Days_Stock_Remaining'].mean()
    st.metric("Avg Days Stock", f"{avg_days:.1f} days")

st.markdown("---")

# ==================== URGENT REORDER ALERTS ====================
urgent_reorders = filtered_data[filtered_data['Reorder_Needed']].sort_values('Days_Stock_Remaining')

if len(urgent_reorders) > 0:
    st.markdown("### 🚨 Urgent Reorder Alerts")
    
    # Show top 5 most urgent
    top_urgent = urgent_reorders.head(5)
    
    for idx, row in top_urgent.iterrows():
        days_left = row['Days_Stock_Remaining']
        alert_class = 'urgent-alert' if days_left < 3 else 'warning-alert'
        
        st.markdown(f"""
        <div class="{alert_class}">
            <strong>⚠️ {row['SKU_ID']}</strong> - {row['Category']}<br>
            Current Stock: <strong>{int(row['Current_Inventory'])}</strong> units | 
            Reorder Point: <strong>{int(row['Reorder_Point'])}</strong> units | 
            Days Remaining: <strong>{days_left}</strong> days<br>
            <strong>ACTION:</strong> Order <strong>{int(row['Reorder_Quantity_EOQ'])}</strong> units immediately!
        </div>
        """, unsafe_allow_html=True)
    
    if len(urgent_reorders) > 5:
        with st.expander(f"📋 View All {len(urgent_reorders)} Reorder Alerts"):
            st.dataframe(
                urgent_reorders[['SKU_ID', 'Category', 'Current_Inventory', 'Reorder_Point', 
                                'Days_Stock_Remaining', 'Reorder_Quantity_EOQ']],
                use_container_width=True,
                hide_index=True
            )
else:
    st.markdown("""
    <div class="success-alert">
        <strong>✅ All Clear!</strong> No urgent reorder alerts. All SKUs have healthy stock levels.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== VISUALIZATION SECTION ====================

# Row 1: Stock Status Distribution and Top SKUs
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📊 Stock Status Distribution")
    
    status_counts = filtered_data['Stock_Status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    
    fig_status = px.pie(
        status_counts,
        values='Count',
        names='Status',
        color='Status',
        color_discrete_map={'Critical': '#f44336', 'Low': '#ff9800', 'Healthy': '#4caf50'},
        hole=0.4
    )
    fig_status.update_traces(textposition='inside', textinfo='percent+label')
    fig_status.update_layout(showlegend=True, height=350)
    st.plotly_chart(fig_status, use_container_width=True)

with col_right:
    st.markdown("### 🔝 Top 10 SKUs by Demand")
    
    top_demand = filtered_data.nlargest(10, 'Avg_Daily_Demand')[['SKU_ID', 'Avg_Daily_Demand', 'Category']]
    
    fig_demand = px.bar(
        top_demand,
        x='Avg_Daily_Demand',
        y='SKU_ID',
        orientation='h',
        color='Category',
        text='Avg_Daily_Demand'
    )
    fig_demand.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    fig_demand.update_layout(yaxis={'categoryorder': 'total ascending'}, height=350)
    st.plotly_chart(fig_demand, use_container_width=True)

st.markdown("---")

# Row 2: Reorder Points vs Current Inventory
st.markdown("### 📉 Inventory Levels: Current Stock vs Reorder Points")

fig_inventory = go.Figure()

# Sort by days remaining for better visualization
display_data = filtered_data.sort_values('Days_Stock_Remaining').head(20)

fig_inventory.add_trace(go.Bar(
    x=display_data['SKU_ID'],
    y=display_data['Current_Inventory'],
    name='Current Stock',
    marker_color='lightblue'
))

fig_inventory.add_trace(go.Scatter(
    x=display_data['SKU_ID'],
    y=display_data['Reorder_Point'],
    mode='markers+lines',
    name='Reorder Point',
    marker=dict(size=10, color='red', symbol='diamond'),
    line=dict(color='red', dash='dash')
))

fig_inventory.add_trace(go.Scatter(
    x=display_data['SKU_ID'],
    y=display_data['Safety_Stock'],
    mode='lines',
    name='Safety Stock',
    line=dict(color='orange', dash='dot')
))

fig_inventory.update_layout(
    xaxis_title="SKU",
    yaxis_title="Units",
    hovermode='x unified',
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_inventory, use_container_width=True)

st.markdown("---")

# Row 3: Category Analysis and Cost Analysis
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Category Performance")
    
    category_summary = filtered_data.groupby('Category').agg({
        'SKU_ID': 'count',
        'Reorder_Needed': 'sum',
        'Inventory_Value': 'sum',
        'Annual_Inventory_Cost': 'sum'
    }).reset_index()
    
    category_summary.columns = ['Category', 'SKU_Count', 'Reorder_Count', 'Inventory_Value', 'Annual_Cost']
    
    fig_category = px.bar(
        category_summary,
        x='Category',
        y=['SKU_Count', 'Reorder_Count'],
        barmode='group',
        labels={'value': 'Count', 'variable': 'Metric'},
        color_discrete_map={'SKU_Count': '#1f77b4', 'Reorder_Count': '#ff7f0e'}
    )
    fig_category.update_layout(height=350)
    st.plotly_chart(fig_category, use_container_width=True)

with col2:
    st.markdown("###  Inventory Cost Analysis")
    
    fig_cost = px.scatter(
        filtered_data,
        x='Avg_Daily_Demand',
        y='Annual_Inventory_Cost',
        size='Reorder_Quantity_EOQ',
        color='Category',
        hover_data=['SKU_ID', 'Reorder_Point'],
        labels={'Avg_Daily_Demand': 'Daily Demand', 'Annual_Inventory_Cost': 'Annual Cost ($)'}
    )
    fig_cost.update_layout(height=350)
    st.plotly_chart(fig_cost, use_container_width=True)

st.markdown("---")

# ==================== DETAILED SKU ANALYSIS ====================
st.markdown("### 🔍 Detailed SKU Analysis")

# SKU selector
selected_sku = st.selectbox(
    "Select SKU for Detailed View",
    options=filtered_data['SKU_ID'].tolist()
)

if selected_sku:
    sku_data = filtered_data[filtered_data['SKU_ID'] == selected_sku].iloc[0]
    
    # SKU Details
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Stock", f"{int(sku_data['Current_Inventory'])}")
        st.metric("Reorder Point", f"{int(sku_data['Reorder_Point'])}")
    
    with col2:
        st.metric("Safety Stock", f"{int(sku_data['Safety_Stock'])}")
        st.metric("Reorder Quantity", f"{int(sku_data['Reorder_Quantity_EOQ'])}")
    
    with col3:
        st.metric("Avg Daily Demand", f"{sku_data['Avg_Daily_Demand']:.1f}")
        st.metric("Std Deviation", f"{sku_data['Std_Daily_Demand']:.1f}")
    
    with col4:
        st.metric("Days Stock Left", f"{sku_data['Days_Stock_Remaining']:.1f}")
        st.metric("Annual Cost", f"${sku_data['Annual_Inventory_Cost']:,.0f}")
    
    # Historical demand chart
    # Try to find SKU history - handle different possible column names
    try:
        # First, check what columns are available
        possible_sku_cols = [col for col in historical_data.columns if 'sku' in col.lower() or 'compound' in col.lower()]
        
        if len(possible_sku_cols) > 0:
            sku_col = possible_sku_cols[0]
            sku_history = historical_data[historical_data[sku_col] == selected_sku]
        else:
            sku_history = pd.DataFrame()  # Empty if no matching column found
    except Exception as e:
        st.warning(f"Could not load historical data: {str(e)}")
        sku_history = pd.DataFrame()
    
    if len(sku_history) > 0:
        st.markdown("#### 📈 Historical Demand Pattern")
        
        # Check what column name exists for units
        units_col = 'Units Sold' if 'Units Sold' in sku_history.columns else (
            'Units_Sold' if 'Units_Sold' in sku_history.columns else 'units_sold'
        )
        
        daily_demand = sku_history.groupby('Date')[units_col].sum().reset_index()
        daily_demand.columns = ['Date', 'Units_Sold']
        
        fig_history = go.Figure()
        
        fig_history.add_trace(go.Scatter(
            x=daily_demand['Date'],
            y=daily_demand['Units_Sold'],
            mode='lines',
            name='Daily Demand',
            line=dict(color='blue')
        ))
        
        # Add average demand line
        avg_demand = daily_demand['Units_Sold'].mean()
        fig_history.add_hline(
            y=avg_demand,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Avg: {avg_demand:.0f}",
            annotation_position="right"
        )
        
        fig_history.update_layout(
            xaxis_title="Date",
            yaxis_title="Units Sold",
            hovermode='x unified',
            height=350
        )
        
        st.plotly_chart(fig_history, use_container_width=True)

st.markdown("---")

# ==================== DATA EXPORT SECTION ====================
st.markdown("### 💾 Export Data")

col1, col2, col3 = st.columns(3)

with col1:
    # Export reorder list
    if st.button("📥 Download Reorder List"):
        reorder_list = filtered_data[filtered_data['Reorder_Needed']][
            ['SKU_ID', 'Category', 'Current_Inventory', 'Reorder_Point', 
             'Reorder_Quantity_EOQ', 'Days_Stock_Remaining']
        ]
        csv = reorder_list.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"reorder_list_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col2:
    # Export full inventory report
    if st.button("📥 Download Full Report"):
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"inventory_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col3:
    # Refresh data
    if st.button("🔄 Refresh Dashboard"):
        st.cache_data.clear()
        st.rerun()

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <strong>Inventory Reorder Management System</strong><br>
    Built with Streamlit & Plotly | Powered by Prophet Forecasting<br>
    © 2025 Tracy Miriti | Data-Driven Inventory Optimization
</div>
""", unsafe_allow_html=True)
