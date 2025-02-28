import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Load trained LightGBM model
lgb_model = joblib.load("lg.pkl")

# Streamlit UI setup
st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")
st.title("📊 Retail Sales Forecasting (LightGBM)")

# Sidebar for user inputs
st.sidebar.header("🔢 Input Features")

# User inputs
quantity = st.sidebar.number_input("Quantity", min_value=1, max_value=100, value=10)
price = st.sidebar.number_input("Price per unit", min_value=0.5, max_value=100.0, value=5.0)
year = st.sidebar.number_input("Year", min_value=2000, max_value=2030, value=2024)
month = st.sidebar.slider("Month", 1, 12, 1)
day = st.sidebar.slider("Day", 1, 31, 1)
weekday = st.sidebar.slider("Weekday (0=Monday, 6=Sunday)", 0, 6, 0)
hour = st.sidebar.slider("Hour", 0, 23, 12)

# Prepare input data
input_data = pd.DataFrame({
    "Quantity": [quantity],
    "Price": [price],
    "Year": [year],
    "Month": [month],
    "Day": [day],
    "Weekday": [weekday],
    "Hour": [hour]
})

# 📌 File Upload for Past Predictions
st.sidebar.header("📂 Upload Dataset for Analysis")
uploaded_file = st.sidebar.file_uploader("Upload CSV (e.g., r910.csv)", type=["csv"])

if uploaded_file:
    st.subheader("📊 Actual vs Predicted Sales Performance")

    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Convert InvoiceDate to datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df.dropna(subset=["InvoiceDate"], inplace=True)  # Remove rows with invalid dates

    # Feature Engineering
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day
    df["Weekday"] = df["InvoiceDate"].dt.weekday
    df["Hour"] = df["InvoiceDate"].dt.hour

    # Create Target Variable (Sales Revenue = Quantity * Price)
    df["Sales"] = df["Quantity"] * df["Price"]

    # Drop unnecessary columns safely
    cols_to_drop = ["Invoice", "StockCode", "Description", "InvoiceDate", "Customer ID", "Country"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Handle missing values using median
    df.fillna(df.median(), inplace=True)

    # Define Features and Target
    X = df.drop(columns=["Sales"])
    y = df["Sales"]

    # Select a subset of data for visualization
    sample_indices = df.sample(n=100, random_state=42).index
    X_sample, y_sample = X.loc[sample_indices], y.loc[sample_indices]

    # Generate Predictions
    lgb_preds = lgb_model.predict(X_sample)

    # 📌 Plot Actual vs Predicted Sales
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_sample, mode="lines+markers", name="Actual Sales", marker=dict(size=6)))
    fig.add_trace(go.Scatter(y=lgb_preds, mode="lines+markers", name="Predicted Sales", marker=dict(size=6)))

    fig.update_layout(
        title="📊 Actual vs Predicted Sales (Sample)",
        xaxis_title="Index",
        yaxis_title="Sales",
        legend_title="Legend",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

# 📌 Prediction Button for User Input
if st.button("🚀 Predict Sales"):
    lgb_pred = lgb_model.predict(input_data)[0]

    st.subheader("📊 Prediction")
    st.metric("Predicted Sales", f"${lgb_pred:.2f}")

    # 📌 **Line Chart - Sales Trend Simulation**
    st.subheader("📈 Sales Forecast Over Time (Simulated)")
    days = np.arange(1, 31)  # Simulating for 30 days
    sales_trend = lgb_pred + np.random.uniform(-10, 10, len(days))  # Adding slight variations
    trend_df = pd.DataFrame({"Day": days, "Predicted Sales": sales_trend})
    fig = px.line(trend_df, x="Day", y="Predicted Sales", markers=True, title="📈 Sales Forecast Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # 📌 **Histogram - Sales Distribution**
    st.subheader("📊 Sales Distribution")
    fig_hist, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sales_trend, bins=10, color="skyblue", edgecolor="black")
    ax.set_xlabel("Predicted Sales")
    ax.set_ylabel("Frequency")
    ax.set_title("Sales Distribution")
    st.pyplot(fig_hist)
