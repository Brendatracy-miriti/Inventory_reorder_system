import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import random

# Core inventory cost parameters
HOLDING_COST_H = 0.20  # $0.20 per unit/day
ORDERING_COST_S = 50.00  # $50.00 per order
STOCKOUT_COST_C = 10.00  # $10.00 per unit/day
SERVICE_LEVEL_Z = 1.64
LEAD_TIME_DAYS = 7
DATA_FILE_NAME = 'retail_store_inventory.csv'
CURRENCY = '$'
LOW_STOCK_THRESHOLD = 50


@st.cache_data
def load_data(file_name: str):
	try:
		df = pd.read_csv(file_name)
	except FileNotFoundError:
		return None
	except Exception as e:
		st.error(f"Failed to read {file_name}: {e}")
		return None

	if 'Date' not in df.columns and 'date' in df.columns:
		df = df.rename(columns={'date': 'Date'})
	try:
		df['Date'] = pd.to_datetime(df['Date'])
	except Exception:
		st.error("Could not parse 'Date' column. Ensure it's present and parseable.")
		return None

	if 'Store ID' in df.columns and 'Product ID' in df.columns:
		df['SKU_Compound_ID'] = df['Store ID'].astype(str) + '_' + df['Product ID'].astype(str)
	elif 'SKU' in df.columns:
		df['SKU_Compound_ID'] = df['SKU'].astype(str)
	else:
		df['SKU_Compound_ID'] = df.index.astype(str)

	df = df.sort_values(['SKU_Compound_ID', 'Date']).reset_index(drop=True)
	return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	if 'Units_Sold' in df.columns and 'Units Sold' not in df.columns:
		df = df.rename(columns={'Units_Sold': 'Units Sold'})
	if 'Units Sold' not in df.columns:
		st.error("Input data must contain a 'Units Sold' column for feature engineering.")
		return pd.DataFrame()

	df['Rolling_Mean_30'] = df.groupby('SKU_Compound_ID')['Units Sold'].transform(
		lambda x: x.shift(1).rolling(window=30, min_periods=1).mean()
	)
	df['Rolling_Std_30'] = df.groupby('SKU_Compound_ID')['Units Sold'].transform(
		lambda x: x.shift(1).rolling(window=30, min_periods=1).std().fillna(0)
	)

	df['Month'] = df['Date'].dt.month
	df['DayOfWeek'] = df['Date'].dt.dayofweek
	df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

	le = LabelEncoder()
	df['SKU_Encoded'] = le.fit_transform(df['SKU_Compound_ID'].astype(str))

	df = df[~df['Rolling_Mean_30'].isna()].copy()
	return df


def run_model_and_policy(df_features: pd.DataFrame) -> pd.DataFrame:
	if df_features.empty:
		return pd.DataFrame()
	policy_df = df_features.groupby('SKU_Compound_ID').last().reset_index()
	# Use deterministic estimate (no random noise) for reproducibility
	policy_df['Estimated_Demand'] = policy_df['Rolling_Mean_30']
	policy_df['Forecast_Std'] = policy_df['Rolling_Std_30']
	policy_df['Safety_Stock'] = SERVICE_LEVEL_Z * policy_df['Forecast_Std'] * np.sqrt(LEAD_TIME_DAYS)
	policy_df['Dynamic_ROP'] = np.ceil((policy_df['Estimated_Demand'] * LEAD_TIME_DAYS) + policy_df['Safety_Stock']).astype(int)
	annual_demand = policy_df['Estimated_Demand'] * 365
	policy_df['Dynamic_ROQ'] = np.ceil(np.sqrt((2 * annual_demand * ORDERING_COST_S) / (HOLDING_COST_H * 365))).fillna(1).astype(int)
	if 'Units Sold' in df_features.columns:
		static_demand_90 = df_features['Units Sold'].quantile(0.90)
	else:
		static_demand_90 = df_features['Rolling_Mean_30'].quantile(0.90)

	# Static policy: compute Reorder Point and ROQ using a conservative (90th percentile) demand estimate
	# Static_ROP uses 90th percentile daily demand * lead time + safety (uses per-SKU forecast std for safety)
	policy_df['Static_ROP'] = np.ceil((static_demand_90 * LEAD_TIME_DAYS) + (SERVICE_LEVEL_Z * policy_df['Forecast_Std'] * np.sqrt(LEAD_TIME_DAYS))).astype(int)
	# Static ROQ computed via EOQ using the static (90th pct) annual demand as a conservative baseline
	static_annual_demand = static_demand_90 * 365
	policy_df['Static_ROQ'] = np.ceil(np.sqrt((2 * static_annual_demand * ORDERING_COST_S) / (HOLDING_COST_H * 365))).fillna(1).astype(int)
	# Set deterministic current inventory: prefer actual 'Inventory Level' when present; otherwise assume a conservative buffer
	if 'Inventory Level' in policy_df.columns:
		policy_df['Current_Inventory'] = policy_df['Inventory Level'].fillna(0).astype(int)
	else:
		# assume on-hand equals 7 days of estimated demand as a conservative default
		policy_df['Current_Inventory'] = (policy_df['Estimated_Demand'] * LEAD_TIME_DAYS).astype(int)

	# Deterministic anticipated stockout cost: estimated shortfall during lead time * stockout unit cost * 30-day horizon
	projected_shortfall = (policy_df['Estimated_Demand'] * LEAD_TIME_DAYS) - policy_df['Current_Inventory']
	projected_shortfall = projected_shortfall.clip(lower=0)
	policy_df['Anticipated_Stockout_Cost'] = projected_shortfall * STOCKOUT_COST_C * 30
	if 'Category' not in policy_df.columns:
		policy_df['Category'] = 'Uncategorized'
	cols = ['SKU_Compound_ID', 'Category', 'Estimated_Demand', 'Dynamic_ROP', 'Dynamic_ROQ', 'Static_ROP', 'Static_ROQ', 'Current_Inventory', 'Anticipated_Stockout_Cost']
	return policy_df[cols]


def calculate_total_cost(sim_data: pd.DataFrame, policy_roq_col: str, policy_rop_col: str, is_dynamic: bool = True):
	sim = sim_data.copy()
	sim['Annual_Demand'] = sim['Estimated_Demand'] * 365
	sim['Ordering_Cost'] = (sim['Annual_Demand'] / sim[policy_roq_col].replace(0, np.nan)) * ORDERING_COST_S
	total_ordering_cost = sim['Ordering_Cost'].fillna(0).sum()
	sim['Safety_Stock'] = (sim[policy_rop_col] - (sim['Estimated_Demand'] * LEAD_TIME_DAYS)).clip(lower=0)
	sim['Avg_Inventory'] = (sim[policy_roq_col] / 2).fillna(0) + sim['Safety_Stock'].fillna(0)
	sim['Holding_Cost'] = sim['Avg_Inventory'] * (HOLDING_COST_H * 365)
	total_holding_cost = sim['Holding_Cost'].fillna(0).sum()
	anticipated = sim.get('Anticipated_Stockout_Cost', pd.Series(0, index=sim.index)).fillna(0)
	# Use deterministic multipliers to compare policies consistently
	# Dynamic policy assumed to reduce realized stockouts (lower multiplier)
	if is_dynamic:
		factor = 0.10
	else:
		factor = 0.40
	total_stockout_cost = anticipated.sum() * factor
	total_cost = total_ordering_cost + total_holding_cost + total_stockout_cost
	return total_cost, total_ordering_cost, total_holding_cost, total_stockout_cost


def run_inventory_analysis(file_name: str):
	df_raw = load_data(file_name)
	if df_raw is None:
		return None, None, None, None
	df_features = engineer_features(df_raw.copy())
	if df_features is None or df_features.empty:
		return None, None, None, None
	policies_df = run_model_and_policy(df_features)
	cost_dynamic, ord_d, hold_d, stock_d = calculate_total_cost(policies_df.copy(), 'Dynamic_ROQ', 'Dynamic_ROP', is_dynamic=True)
	cost_static, ord_s, hold_s, stock_s = calculate_total_cost(policies_df.copy(), 'Static_ROQ', 'Static_ROP', is_dynamic=False)
	savings = cost_static - cost_dynamic
	reduction_pct = f"{(savings / cost_static * 100) if cost_static else 0:.2f}%"
	cost_summary_metrics = {
		'Static Total': cost_static,
		'Dynamic Total': cost_dynamic,
		'Savings': savings,
		'Reduction %': reduction_pct,
		# expose component breakdown for investigation
		'Static Ordering': ord_s,
		'Static Holding': hold_s,
		'Static Stockout': stock_s,
		'Dynamic Ordering': ord_d,
		'Dynamic Holding': hold_d,
		'Dynamic Stockout': stock_d,
	}
	cost_viz_data = {
		'Policy': ['Static'] * 3 + ['Dynamic'] * 3,
		'Cost Component': ['Ordering Cost (S)', 'Holding Cost (H)', 'Stockout Cost (C)'] * 2,
		'Annual Cost': [ord_s, hold_s, stock_s, ord_d, hold_d, stock_d]
	}
	cost_viz_df = pd.DataFrame(cost_viz_data)
	policies_df['Reorder_Needed'] = policies_df['Current_Inventory'] < policies_df['Dynamic_ROP']
	return policies_df, cost_summary_metrics, cost_viz_df, df_features


# --- 1. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Dynamic Inventory Policy Dashboard")

st.markdown("""
<style>
.stApp { background-color: #f0f2f6; }
h1 { color: #1E3A8A; border-bottom: 3px solid #BFDBFE; padding-bottom: 10px; }
.stMetric { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-left: 5px solid #3B82F6; }
.metric-savings-value { font-size: 28px; color: #059669; font-weight: 700; }
[data-testid="stMetricValue"] { font-size: 28px; color: #1E3A8A; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_analysis_results():
	return run_inventory_analysis(DATA_FILE_NAME)


# Load results
policies_df, cost_metrics, cost_viz_df, df_features = get_analysis_results()

if policies_df is None or cost_metrics is None:
	st.error("Execution Error: The inventory analysis failed.")
	st.warning(f"Please ensure the file '{DATA_FILE_NAME}' exists and contains at least 30 days of history per SKU.")
	st.stop()


st.title("Data-Driven Inventory Optimization")
st.markdown("---")

# KPI cards
total_products = policies_df['SKU_Compound_ID'].nunique()
total_stock = policies_df['Current_Inventory'].sum()
low_stock_alerts = policies_df[policies_df['Reorder_Needed']].shape[0]

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

with col_kpi1:
	st.metric(label="Total Unique Products (SKUs)", value=f"{total_products:,.0f}")
with col_kpi2:
	st.metric(label="Total Current Stock Units", value=f"{total_stock:,.0f}")
with col_kpi3:
	st.metric(label="Low Stock Alerts (Reorder Needed)", value=f"{low_stock_alerts:,.0f}", delta="Action Required", delta_color="inverse")
with col_kpi4:
	st.markdown(
		f"<div class='stMetric' style='border-left:5px solid #059669;'><p data-testid='stMetricLabel'>Estimated Annual Cost Savings</p><div class='metric-savings-value'>{CURRENCY} {cost_metrics['Savings']:,.0f}</div><p data-testid='stMetricDelta' style='color:#059669;'>{cost_metrics['Reduction %']} Cost Reduction</p></div>",
		unsafe_allow_html=True,
	)

st.markdown("---")

# Cost breakdown chart
fig_cost_breakdown = px.bar(
	cost_viz_df,
	x="Cost Component",
	y="Annual Cost",
	color="Policy",
	barmode="group",
	title="Annual Cost Breakdown: Static vs. Dynamic Optimization",
	height=500,
	color_discrete_map={'Static': '#EF4444', 'Dynamic': '#10B981'},
	labels={"Annual Cost": f"Annual Cost ({CURRENCY})", "Cost Component": "Inventory Cost Component"}
)

fig_cost_breakdown.add_hline(y=cost_metrics['Static Total'], line_dash="dot", annotation_text=f"Static Total: {CURRENCY} {cost_metrics['Static Total']:,.0f}", annotation_position="bottom right", line_color="#DC2626")
fig_cost_breakdown.add_hline(y=cost_metrics['Dynamic Total'], line_dash="dot", annotation_text=f"Dynamic Total: {CURRENCY} {cost_metrics['Dynamic Total']:,.0f}", annotation_position="top right", line_color="#10B981")
fig_cost_breakdown.update_layout(yaxis_tickprefix=CURRENCY, yaxis_tickformat=",.")

st.plotly_chart(fig_cost_breakdown, width='stretch', key='fig_cost_breakdown')

st.markdown("---")

# Trends
st.header("Historical Performance: Sales and Inventory Trends")

df_trends = df_features.groupby(df_features['Date'].dt.to_period('M')).agg(
	Total_Units_Sold=('Units Sold', 'sum'),
	Average_Inventory=('Inventory Level', 'mean')
).reset_index()
df_trends['Date'] = df_trends['Date'].astype(str)
df_trends_melt = df_trends.melt('Date', var_name='Metric', value_name='Value')

fig_trends = px.line(
	df_trends_melt,
	x='Date',
	y='Value',
	color='Metric',
	title='Monthly Aggregate Sales vs. Average Inventory Level',
	markers=True,
	height=450,
)
fig_trends.update_layout(xaxis_title='Month', yaxis_title='Units (Monthly Total/Average)')
st.plotly_chart(fig_trends, width='stretch', key='fig_trends')

st.markdown("---")

# Reorder recommendations
st.header("Actionable Reorder Recommendations")
col_filter_cat, col_filter_top = st.columns([3, 1])
with col_filter_cat:
	filter_category = st.selectbox("Filter Recommendations by Product Category", options=['All Categories'] + list(policies_df['Category'].unique()), key='filter_cat')
with col_filter_top:
	top_n = st.slider("Show Top N Items", min_value=5, max_value=50, value=10, step=5, key='top_n')

if filter_category != 'All Categories':
	filtered_reorder_df = policies_df[policies_df['Category'] == filter_category]
else:
	filtered_reorder_df = policies_df.copy()

reorder_list = filtered_reorder_df[filtered_reorder_df['Reorder_Needed'] == True].sort_values('Anticipated_Stockout_Cost', ascending=False).reset_index(drop=True)

if reorder_list.empty:
	st.success(f"No products in {filter_category} currently require reordering.")
else:
	st.warning(f"{len(reorder_list)} items require reordering. Showing top {top_n}:")
	st.dataframe(
		reorder_list[['SKU_Compound_ID', 'Category', 'Current_Inventory', 'Dynamic_ROP', 'Dynamic_ROQ', 'Estimated_Demand', 'Anticipated_Stockout_Cost']].head(top_n),
		use_container_width=True,
		hide_index=True,
	)

st.markdown("---")

# Product-level deep dive
st.header("Product Forecast and Policy Lookup")
selected_sku = st.selectbox("Select SKU to View Details", options=policies_df['SKU_Compound_ID'].unique())
if selected_sku:
	sku_data = policies_df[policies_df['SKU_Compound_ID'] == selected_sku].iloc[0]
	st.subheader(f"Details for {selected_sku} ({sku_data['Category']})")
	colA, colB, colC, colD = st.columns(4)
	with colA:
		st.metric("Current Inventory", f"{sku_data['Current_Inventory']} Units")
	with colB:
		st.metric("Dynamic ROP", f"{sku_data['Dynamic_ROP']} Units")
	with colC:
		st.metric("Dynamic ROQ", f"{sku_data['Dynamic_ROQ']} Units")
	with colD:
		st.metric("Est. Annual Demand", f"{sku_data['Estimated_Demand'] * 365:,.0f} Units")
	forecast_dates = pd.date_range(start=datetime.now(), periods=30)
	base_demand = sku_data['Estimated_Demand']
	daily_demand = np.random.normal(base_demand, base_demand * 0.1, size=30)
	daily_demand[daily_demand < 0] = base_demand * 0.5
	forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Demand (Units)': daily_demand})
	fig_forecast = px.line(forecast_df, x='Date', y='Forecasted Demand (Units)', title=f"30-Day Demand Forecast for {selected_sku}", markers=True)
	st.plotly_chart(fig_forecast, width='stretch', key=f'fig_forecast_{selected_sku}')
