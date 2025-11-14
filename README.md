# Dynamic Inventory Management System

## 🚀 Project Overview

This capstone project implements a data-driven solution to modernize inventory management. By replacing static, rule-of-thumb reorder policies with a **dynamic, predictive engine**, the system minimizes total inventory costs.

The core objective is to achieve **optimal service levels** (avoiding costly stockouts) while simultaneously minimizing **holding costs** (storage, capital tied up) by accurately forecasting future demand and automatically calculating the optimal **Reorder Point (ROP)** and **Reorder Quantity (ROQ)** for a portfolio of **100 unique, compound SKUs**.

---

## 💾 Data & Licensing

### Source Dataset

The concepts and structure for this project are inspired by a real-world retail scenario.

| Metric | Detail |
| :--- | :--- |
| **Source Name** | Retail Store Inventory Forecasting Dataset |
| **Kaggle Link** | [https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset) |
| **Note** | The data displayed in the dashboard is a **simulated dataset** created in `app.py` using NumPy and Pandas to mimic the real-world structure for demonstration purposes. |

### ⚖️ License

This project is licensed under the **MIT License**. This permissive license allows you to use, modify, distribute, and even sell the code, provided you include the original copyright and license notice.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Dashboard** | Python / Streamlit | Interactive visualization and front-end for real-time decision-making. |
| **Data Logic** | Python / Pandas / NumPy | Data simulation, cleaning, and policy calculation. |
| **Forecasting** | **Prophet (from Meta)** | Time-series demand forecasting engine. |
| **Deployment** | GitHub / Streamlit Cloud | Version control and hosting. |

---

## 📐 Project Architecture & Phases

The system follows a standard data science pipeline:

| Phase | Title | Key Activities |
| :--- | :--- | :--- |
| **Phase 1** | **Data Simulation & Engineering** | Generate 100 compound SKUs (`S001_P00xx`), 180 days of historical demand, and initial policy/status data (`df_status`, `df_policies`). |
| **Phase 2** | **Predictive Modeling (Prophet)** | Fit the Prophet model to historical demand for each SKU, generating a **30-day forward forecast** including confidence intervals (`yhat_upper`, `yhat_lower`). |
| **Phase 3** | **Inventory Policy Optimization** | Calculate the **Dynamic ROP** (based on lead time demand & confidence interval) and **Dynamic ROQ** (optimal lot size) for each SKU. |
| **Phase 4** | **Dashboard Deployment** | Ingest final calculated policies and status into the **Streamlit application** (`app.py`) for visualization and actionable insights. |

---

## 🧠 Why the Prophet Model?

Prophet was chosen as the primary forecasting tool for inventory demand due to its business-friendly features:

| Rationale | Description |
| :--- | :--- |
| **Seasonality & Holidays** | Designed to automatically detect and model weekly, monthly, or yearly demand patterns, critical in retail. |
| **Robustness** | Highly tolerant of missing data and outliers, which are common in real-world sales logs. |
| **Safety Stock** | Natively provides **forecast uncertainty** (confidence intervals), which are statistically essential inputs for calculating Safety Stock and the Dynamic ROP. |
| **Intuitive** | Uses parameters that are easily understood by inventory managers, facilitating buy-in and operation. |

---

## 📊 Streamlit Dashboard Features

The dashboard provides a real-time, filtered view of the inventory portfolio:

### Portfolio Health Check

* **Filters**: View portfolio status by **Product Category** (Toys, Clothing, Furniture, Electronics).
* **KPIs**: Displays metrics like **Critical Stock SKUs**, **Total Inventory Value**, and **Average Dynamic Policy Savings**.
* **Status Distribution**: A chart visualizing the number of SKUs in **Healthy**, **Low**, or **Critical** stock status.

### Real-Time Demand Forecast

* **Interactive Chart**: Displays 180 days of history and the **30-day Prophet forecast** with its 95% confidence interval, showing expected demand and volatility for the selected SKU.



### Policy Comparison & Recommendation

* **Cost Savings**: Highlights the monetary and percentage reduction achieved by adopting the **Dynamic Policy**.
* **Reorder Parameters**: Compares **Static ROP** vs. **Dynamic ROP** and provides the precise **Recommended Order Quantity (ROQ)**.
* **Action Block**: Provides clear, immediate instruction (e.g., "IMMEDIATE ACTION REQUIRED") if current stock falls below the Dynamic ROP.

---

## 💻 Installation and Setup

To run this dashboard locally, follow these steps:

| Step | Command / Detail |
| :--- | :--- |
| **1. Clone Repository** | `git clone https://github.com/Brendatracy-miriti/Inventory_reorder_system.git` |
| | `cd Inventory_dashboard` |
| **2. Create Environment** | `python -m venv venv` |
| | `source venv/bin/activate` (Use `venv\Scripts\activate` on Windows) |
| **3. Install Dependencies** | `pip install streamlit pandas numpy plotly` |
| **4. Run Application** | `streamlit run app.py` |
| **Result** | The application opens automatically in your browser at `http://localhost:8501`. |