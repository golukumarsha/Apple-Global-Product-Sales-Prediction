import streamlit as st
import pandas as pd
import joblib

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Apple Sales Prediction",
    page_icon="🍎",
    layout="wide"
)

# ==============================
# Load Model
# ==============================


@st.cache_resource
def load_model():
    return joblib.load("sales_model.pkl")


model = load_model()

# ==============================
# Header Section
# ==============================
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🍎 Apple Global Product Sales Prediction</h1>",
            unsafe_allow_html=True)
st.markdown("---")

# ==============================
# Sidebar Inputs
# ==============================
st.sidebar.header("📊 Enter Sales Details")

year = st.sidebar.number_input(
    "Year", min_value=2020, max_value=2030, value=2023)
unit_price = st.sidebar.number_input("Unit Price (USD)", value=1000.0)
discount_pct = st.sidebar.slider(
    "Discount (%)", min_value=0.0, max_value=50.0, value=10.0)
units_sold = st.sidebar.number_input("Units Sold", value=50)

# ==============================
# Create Input DataFrame
# ==============================
columns = model.feature_names_in_

input_data = pd.DataFrame(columns=columns)
input_data.loc[0] = 0

if "year" in input_data.columns:
    input_data.at[0, "year"] = year

if "unit_price_usd" in input_data.columns:
    input_data.at[0, "unit_price_usd"] = unit_price

if "discount_pct" in input_data.columns:
    input_data.at[0, "discount_pct"] = discount_pct

if "units_sold" in input_data.columns:
    input_data.at[0, "units_sold"] = units_sold

# ==============================
# Prediction Section
# ==============================
st.markdown("## 📈 Prediction Result")

if st.button("🚀 Predict Revenue"):
    prediction = model.predict(input_data)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("💰 Predicted Revenue (USD)", f"${prediction[0]:,.2f}")

    with col2:
        discounted_price = unit_price * (1 - discount_pct / 100)
        st.metric("🏷 Discounted Price (USD)", f"${discounted_price:,.2f}")

    st.success("Prediction completed successfully!")

st.markdown("---")
st.caption("Developed by Golu Kumar | Machine Learning Project 🚀")
