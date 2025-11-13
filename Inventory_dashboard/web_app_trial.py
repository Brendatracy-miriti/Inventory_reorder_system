import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Dynamic Inventory Management Dashboard",
    layout="wide",
    page_icon="📦"
)

st.title("📊 Dynamic Inventory Management Dashboard")
st.markdown("""
Explore and compare **Static vs Dynamic Inventory Policies**.  
Analyze **forecast trends**, **cost simulations**, and **stock alerts** with intelligent reorder recommendations.
""")

# -----------------------------
# SIDEBAR UPLOAD SECTION
# -----------------------------
st.sidebar.header("📁 Upload Data Files")
inventory_data = st.sidebar.file_uploader("Upload Inventory Data (e.g., retail_store_inventory.csv)", type=["csv"])
forecast_data = st.sidebar.file_uploader("Upload Forecast Data (e.g., future_forecast_data.csv)", type=["csv"])

if inventory_data and forecast_data:
    df_inventory = pd.read_csv(inventory_data)
    df_forecast = pd.read_csv(forecast_data)

    st.success("✅ Data successfully loaded!")

    # Rename columns if necessary
    rename_map = {
        'Policy': 'policy_type',
        'Total_Cost': 'total_cost',
        'Holding_Cost': 'holding_cost',
        'Inventory_Level': 'inventory_level',
        'Date': 'date',
        'Product': 'product_name',
        'Sales': 'sales'
    }
    df_inventory.rename(columns={k: v for k, v in rename_map.items() if k in df_inventory.columns}, inplace=True)
    df_forecast.rename(columns={k: v for k, v in rename_map.items() if k in df_forecast.columns}, inplace=True)

    # Ensure datetime
    if 'date' in df_inventory.columns:
        df_inventory['date'] = pd.to_datetime(df_inventory['date'])
    if 'date' in df_forecast.columns:
        df_forecast['date'] = pd.to_datetime(df_forecast['date'])

    # -----------------------------
    # CALCULATE REORDER POINT (ROP) & REORDER QUANTITY (ROQ)
    # -----------------------------
    # Assumption: You have 'daily_demand' and 'lead_time' columns or we can estimate from sales
    if 'sales' in df_inventory.columns:
        daily_demand = df_inventory.groupby('product_name')['sales'].mean()
        demand_std = df_inventory.groupby('product_name')['sales'].std()
        lead_time = 5  # days (adjust as per business)
        z_score = 1.65  # 95% service level

        reorder_point = daily_demand * lead_time + z_score * demand_std * np.sqrt(lead_time)
        reorder_qty = (daily_demand * (lead_time + 7)) - reorder_point  # 7-day buffer

        reorder_df = pd.DataFrame({
            'product_name': reorder_point.index,
            'reorder_point': reorder_point.values.round(2),
            'reorder_qty': reorder_qty.values.round(2)
        })

        df_inventory = pd.merge(df_inventory, reorder_df, on='product_name', how='left')

        # Identify low stock alerts
        df_inventory['low_stock_alert'] = df_inventory['inventory_level'] <= df_inventory['reorder_point']

    # -----------------------------
    # SUMMARY CARDS
    # -----------------------------
    st.subheader("📋 Inventory Overview")

    total_products = df_inventory['product_name'].nunique() if 'product_name' in df_inventory.columns else 0
    total_stock = df_inventory['inventory_level'].sum() if 'inventory_level' in df_inventory.columns else 0
    low_stock_count = df_inventory['low_stock_alert'].sum() if 'low_stock_alert' in df_inventory.columns else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Products", f"{total_products:,}")
    col2.metric("Total Stock Units", f"{total_stock:,}")
    col3.metric("Low Stock Alerts", f"{low_stock_count:,}")

    st.markdown("---")

    # -----------------------------
    # POLICY COST COMPARISON
    # -----------------------------
    st.subheader("💰 Policy Cost Comparison")

    if {'policy_type', 'total_cost', 'holding_cost'}.issubset(df_inventory.columns):
        cost_summary = df_inventory.groupby('policy_type')[['total_cost', 'holding_cost']].mean().reset_index()

        fig_cost = px.bar(
            cost_summary,
            x='policy_type',
            y=['total_cost', 'holding_cost'],
            barmode='group',
            title="Average Total & Holding Costs by Policy",
            labels={'value': 'Cost (KSh)', 'policy_type': 'Policy Type'},
            text_auto=True
        )
        st.plotly_chart(fig_cost, use_container_width=True)
    else:
        st.warning("⚠️ Missing columns for cost comparison. Expected: policy_type, total_cost, holding_cost.")

    # -----------------------------
    # SALES & INVENTORY TRENDS
    # -----------------------------
    if {'date', 'sales', 'inventory_level'}.issubset(df_inventory.columns):
        st.subheader("📈 Sales and Inventory Trends")

        fig_trend = px.line(
            df_inventory,
            x='date',
            y=['sales', 'inventory_level'],
            color='policy_type' if 'policy_type' in df_inventory.columns else None,
            title="Sales vs Inventory Over Time"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # -----------------------------
    # LOW STOCK ALERT TABLE
    # -----------------------------
    if 'low_stock_alert' in df_inventory.columns:
        st.subheader("🚨 Low Stock Alerts")

        low_stock_df = df_inventory[df_inventory['low_stock_alert'] == True][
            ['product_name', 'inventory_level', 'reorder_point', 'reorder_qty']
        ].drop_duplicates()

        if len(low_stock_df) > 0:
            st.dataframe(low_stock_df.style.highlight_min(subset=['inventory_level'], color='lightcoral'))
        else:
            st.success("✅ All products are above their reorder points!")

    # -----------------------------
    # FORECAST VISUALIZATION
    # -----------------------------
    if {'date', 'forecast'}.issubset(df_forecast.columns):
        st.subheader("🔮 Forecast Trends")

        fig_forecast = px.line(
            df_forecast,
            x='date',
            y='forecast',
            color='policy_type' if 'policy_type' in df_forecast.columns else None,
            title="Forecasted Demand Over Time"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

    # -----------------------------
    # DOWNLOAD SECTION
    # -----------------------------
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        "📥 Download Processed Data",
        df_inventory.to_csv(index=False).encode('utf-8'),
        file_name="processed_inventory_data.csv",
        mime='text/csv'
    )

else:
    st.info("👈 Upload both datasets to get started.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<hr>
<p style='text-align:center; color:gray'>
Built by <b>Tracy Miriti</b> | Streamlit + Plotly | Inventory Optimization Dashboard
</p>
""", unsafe_allow_html=True)
