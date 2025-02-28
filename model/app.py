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

# Sidebar for user inputs (Single Prediction)
st.sidebar.header("🔢 Input Features")

quantity = st.sidebar.number_input("Quantity", min_value=1, max_value=100, value=10)
price = st.sidebar.number_input("Price per unit", min_value=0.5, max_value=100.0, value=5.0)
year = st.sidebar.number_input("Year", min_value=2000, max_value=2030, value=2024)
month = st.sidebar.slider("Month", 1, 12, 1)
day = st.sidebar.slider("Day", 1, 31, 1)
weekday = st.sidebar.slider("Weekday (0=Monday, 6=Sunday)", 0, 6, 0)
hour = st.sidebar.slider("Hour", 0, 23, 12)

# Prepare input data for single prediction
input_data = pd.DataFrame({
    "Quantity": [quantity],
    "Price": [price],
    "Year": [year],
    "Month": [month],
    "Day": [day],
    "Weekday": [weekday],
    "Hour": [hour]
})

# Sidebar for uploading a historical dataset
st.sidebar.header("📂 Upload Dataset for Analysis")
uploaded_file = st.sidebar.file_uploader("Upload CSV (e.g., r910.csv)", type=["csv"])

if uploaded_file:
    st.subheader("📊 Historical Analysis: Actual vs Predicted Sales")
    df = pd.read_csv(uploaded_file)

    # Convert InvoiceDate to datetime (adjust format if needed)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%d-%m-%Y %H:%M", errors="coerce")
    df.dropna(subset=["InvoiceDate"], inplace=True)

    # Feature Engineering: Extract date components
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day
    df["Weekday"] = df["InvoiceDate"].dt.weekday
    df["Hour"] = df["InvoiceDate"].dt.hour

    # Create target variable: Sales Revenue = Quantity * Price
    df["Sales"] = df["Quantity"] * df["Price"]

    # Verify required feature columns exist
    feature_columns = ['Quantity', 'Price', 'Year', 'Month', 'Day', 'Weekday', 'Hour']
    if not all(col in df.columns for col in feature_columns):
        st.error("Missing required columns for prediction!")
    else:
        # Prepare features and generate predictions for the full dataset
        X_full = df[feature_columns].fillna(0)
        df["Predicted_Sales"] = lgb_model.predict(X_full)

        # Method 1: Plot a sample of Actual vs Predicted Sales
        sample_indices = df.sample(n=100, random_state=42).index
        y_sample = df.loc[sample_indices, "Sales"]
        preds_sample = df.loc[sample_indices, "Predicted_Sales"]

        fig_sample = go.Figure()
        fig_sample.add_trace(go.Scatter(y=y_sample, mode="lines+markers", name="Actual Sales", marker=dict(size=6)))
        fig_sample.add_trace(
            go.Scatter(y=preds_sample, mode="lines+markers", name="Predicted Sales", marker=dict(size=6)))
        fig_sample.update_layout(
            title="📊 Sample: Actual vs Predicted Sales",
            xaxis_title="Sample Index",
            yaxis_title="Sales Revenue",
            template="plotly_white"
        )
        st.plotly_chart(fig_sample, use_container_width=True)

        # Method 2: Monthly Aggregation of Actual vs Predicted Sales
        df_monthly_actual = df.groupby(['Year', 'Month'], as_index=False)['Sales'].sum()
        df_monthly_pred = df.groupby(['Year', 'Month'], as_index=False)['Predicted_Sales'].sum()
        df_monthly = pd.merge(df_monthly_actual, df_monthly_pred, on=['Year', 'Month'], how='inner')
        df_monthly['Year-Month'] = df_monthly['Year'].astype(str) + "-" + df_monthly['Month'].astype(str).str.zfill(2)

        fig_monthly = px.line(df_monthly, x='Year-Month', y=['Sales', 'Predicted_Sales'],
                              color_discrete_map={'Sales': 'blue', 'Predicted_Sales': 'red'},
                              title="📊 Monthly Actual vs Predicted Sales",
                              labels={'value': 'Sales Revenue', 'Year-Month': 'Year-Month'},
                              markers=True)
        st.plotly_chart(fig_monthly, use_container_width=True)

# Single prediction using user input features
if st.button("🚀 Predict Sales"):
    pred_value = lgb_model.predict(input_data)[0]
    st.subheader("📊 Single Prediction")
    st.metric("Predicted Sales", f"${pred_value:.2f}")

    # Simulated Sales Trend Over Time
    st.subheader("📈 Simulated Sales Forecast Over 30 Days")
    days = np.arange(1, 31)
    simulated_sales = pred_value + np.random.uniform(-10, 10, len(days))
    trend_df = pd.DataFrame({"Day": days, "Predicted Sales": simulated_sales})
    fig_trend = px.line(trend_df, x="Day", y="Predicted Sales", markers=True, title="Sales Forecast Over Time")
    st.plotly_chart(fig_trend, use_container_width=True)

    # Sales Distribution Histogram
    st.subheader("📊 Sales Distribution Histogram")
    fig_hist, ax = plt.subplots(figsize=(8, 4))
    ax.hist(simulated_sales, bins=10, color="skyblue", edgecolor="black")
    ax.set_xlabel("Predicted Sales")
    ax.set_ylabel("Frequency")
    ax.set_title("Sales Distribution")
    st.pyplot(fig_hist)
