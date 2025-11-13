**Dynamic Inventory Management System**

🚀 Project Overview

This project implements a data-driven solution to modernize inventory management. By replacing static, rule-of-thumb reorder policies with a dynamic, predictive engine, the system minimizes total inventory costs.

The core objective is to achieve optimal service levels (avoiding costly stockouts) while simultaneously minimizing holding costs (storage, capital tied up) by accurately forecasting future demand and automatically calculating the optimal Reorder Point (ROP) and Reorder Quantity (ROQ) for a portfolio of 100 unique, compound SKUs.

💾 Data Source

The concepts and structure for this project are inspired by a real-world retail scenario.

Source Dataset: Retail Store Inventory Forecasting Dataset

Link: https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset

Note: The data displayed in this dashboard is a simulated dataset created in the app.py script, leveraging NumPy and Pandas, to mimic the structure, time series demand, and key features (like stock levels and policies) of a large retail inventory portfolio for demonstration purposes.

🛠️ Technology Stack

Component

Technology

Purpose

Frontend/Dashboard

Python / Streamlit

Interactive visualization and user interface for decision-makers.

Data Processing

Python / Pandas

Data simulation, cleaning, and preparation.

Forecasting Model

Prophet (from Meta)

Time-series demand forecasting engine.

Deployment

GitHub / Streamlit Cloud

Version control and hosting.

📐 Project Architecture & Phases

The system follows a standard data science pipeline, from data generation to deployment:

Phase 1: Data Simulation and Engineering

A robust dataset is simulated using Pandas and NumPy to mimic real-world inventory data, featuring 100 products using a compound SKU format (e.g., S001_P00xx).

df_status: Current stock levels, safety stock targets, and an immediate status flag (Healthy, Low, Critical).

df_policies: Historical cost data for policy comparison.

Time Series Data: 180 days of historical demand data for each SKU is generated for the forecasting model.

Phase 2: Predictive Modeling (Prophet)

The raw time-series demand data for each of the 100 SKUs is fed into the Prophet model.

Prophet generates a 30-day forward forecast (yhat).

It also provides statistically derived upper and lower bounds for the forecast (yhat_upper, yhat_lower), which are crucial for safety stock calculations.

Phase 3: Inventory Policy Optimization

Using the probabilistic output from the Prophet model, the system calculates two critical inventory parameters for each SKU:

Dynamic Reorder Point (ROP): Calculated based on the forecasted demand during lead time and the desired service level (using the confidence interval from Prophet). This dynamically adjusts to demand volatility.

Dynamic Reorder Quantity (ROQ): Calculated based on optimal lot size models, ensuring the order quantity minimizes the combined cost of ordering and holding.

Phase 4: Dashboard Deployment and Visualization

The final policy and status data are ingested by the Streamlit application (app.py), which presents actionable insights to the user via a live, interactive dashboard.

🧠 Why the Prophet Model?

Prophet was chosen as the primary forecasting model for inventory demand due to its specific advantages in a business context:

Rationale

Description

Seasonality & Holidays

Prophet is explicitly designed to detect and model human-driven patterns, such as weekly, monthly, or yearly seasonality, without manual intervention.

Robust to Missing Data

Unlike traditional ARIMA models, Prophet handles gaps in the time series data (common in sales/inventory) and outliers gracefully.

Intuitive Parameters

The model's parameters (e.g., trend change points, seasonality modes) are easily understandable and configurable by non-statisticians (like inventory managers).

Confidence Intervals

Prophet natively provides forecast uncertainty (confidence intervals), which are essential inputs for calculating the statistical Safety Stock needed to meet a specific service level goal.

📊 Streamlit Dashboard Features

The dashboard provides a real-time, filtered view of the inventory portfolio, driven by the data generated in the back-end phases.

1. Portfolio Health Check

Filters: Allows filtering the entire view by Product Category (Toys, Clothing, Furniture, Electronics).

KPIs: Displays high-level metrics like Critical Stock SKUs, Total Inventory Value, and Average Dynamic Policy Savings.

Status Distribution: A bar chart visualizing the number of SKUs in Healthy, Low, or Critical stock status across the filtered portfolio.

2. Real-Time Demand Forecast

Interactive Chart: Displays the SKU-specific demand over the last 180 days of history, transitioning seamlessly into the 30-day Prophet forecast with its associated 95% confidence interval. This clearly shows expected demand and potential volatility.

3. Policy Comparison & Recommendation

Cost Savings: Highlights the monetary and percentage reduction in total inventory costs achieved by switching from the static to the Dynamic Policy.

Reorder Parameters: Compares the old Static ROP vs. the new Dynamic ROP and provides the Recommended Order Quantity (ROQ).

Action Block: A clear, immediate instruction (e.g., "IMMEDIATE ACTION REQUIRED") is displayed if the current stock falls below the calculated Dynamic ROP, alongside the recommended ROQ.

Cost Performance Chart: A bar chart comparing the average static vs. dynamic total inventory costs across the filtered category.

💻 Installation and Setup

To run this dashboard locally, follow these steps:

Clone the repository:

git clone [(https://github.com/Brendatracy-miriti/Inventory_reorder_system)]
cd Inventory_dashboard


Create a virtual environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate


Install dependencies:
You will need streamlit, pandas, numpy, and plotly.

pip install streamlit pandas numpy plotly


Run the application:

streamlit run app.py


The application will open automatically in your browser, typically at http://localhost:8501.


⚖️ License

This project is licensed under the MIT License.

This means you are free to:

Use the code for any purpose.

Modify the code.

Distribute the code, or components of it.

The only requirement is to include the original copyright and license notice in any substantial portions of the code. See the LICENSE file for full details.
