import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# LOAD FILE
df = pd.read_csv("credit_risk_predictions.csv")

st.set_page_config(page_title="Credit Risk Insights", layout="wide")

st.title("🛡️Credit Risk Prediction")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("High Risk %", f"{(df['Predicted_Probability'] > 0.5).mean() * 100:.1f}%")
col3.metric("Avg Debt Ratio", f"{df['DebtRatio'].mean():.2f}")

st.markdown("---")

# SIDEBAR FILTERS
st.sidebar.header("Filters")
age_range = st.sidebar.slider(
    "Age Range", int(df["age"].min()), int(df["age"].max()), (20, 60)
)
income_range = st.sidebar.slider(
    "Monthly Income",
    int(df["MonthlyIncome"].min()),
    int(df["MonthlyIncome"].max()),
    (2000, 10000),
)
threshold = st.sidebar.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.05)

df_filtered = df[
    (df["age"].between(*age_range)) & (df["MonthlyIncome"].between(*income_range))
].copy()

# RISK SEGMENTATION
st.subheader("🪙Risk Segmentation")
bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
labels = ["0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-0.7", "0.7-1.0"]
df_filtered["RiskBand"] = pd.cut(df_filtered["Predicted_Probability"], bins=bins, labels=labels)

band_counts = df_filtered["RiskBand"].value_counts().sort_index().reset_index()
band_counts.columns = ["RiskBand", "Count"]

fig1 = px.funnel(
    band_counts,
    x="Count",
    y="RiskBand",
    color="RiskBand",
    color_discrete_sequence=px.colors.qualitative.Set2,
    title="Customer classification according to defined risk categories"
)
st.plotly_chart(fig1, use_container_width=True)

# DEFAULT PROBABILITY DISTRIBUTION
st.subheader("🪙Default Probability Distribution")
df_filtered["RiskCategory"] = np.where(
    df_filtered["Predicted_Probability"] > threshold, "High Risk", "Low Risk"
)
fig2 = px.histogram(
    df_filtered,
    x="Predicted_Probability",
    nbins=40,
    color="RiskCategory",
    marginal="box",
    opacity=0.7,
    color_discrete_map={"High Risk": "red", "Low Risk": "green"},
    title="Distribution analysis of projected default probabilities"
)
st.plotly_chart(fig2, use_container_width=True)

# INCOME DISTRIBUTION
st.subheader("🪙Income Distribution Across Risk Bands")
fig3 = px.box(
    df_filtered,
    x="RiskBand",
    y="MonthlyIncome",
    color="RiskBand",
    color_discrete_sequence=px.colors.qualitative.Set2,
    title="Income patterns segmented by customer risk bands"
)
st.plotly_chart(fig3, use_container_width=True)

# CUSTOMER INSIGHTS
st.subheader("🔸Customer Insights")
st.dataframe(
    df_filtered[["age", "MonthlyIncome", "DebtRatio", "Predicted_Probability"]].head(50)
)
