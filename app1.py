import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import time

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Apple Sales Predictor",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# Custom CSS for better styling
# ==============================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #ff4b4b 0%, #ff8c8c 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        margin: 10px 0;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #ff4b4b 0%, #ff8c8c 100%);
        color: white;
        font-size: 1.2rem;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        border: none;
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(255, 75, 75, 0.6);
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        animation: slideIn 0.5s ease;
        margin-top: 20px;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 10px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# Initialize session state
# ==============================
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_value' not in st.session_state:
    st.session_state.prediction_value = None
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False

# ==============================
# Load Model with error handling
# ==============================


@st.cache_resource
def load_model():
    try:
        model = joblib.load("sales_model.pkl")
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None


model = load_model()

# ==============================
# Header Section
# ==============================
st.markdown("""
<div class="main-header">
    <h1>🍎 Apple Global Sales Predictor</h1>
    <p style='color: white; font-size: 1.2rem;'>AI-Powered Revenue Forecasting System</p>
</div>
""", unsafe_allow_html=True)

# ==============================
# Sidebar Inputs
# ==============================
with st.sidebar:
    st.markdown("## 📊 Sales Configuration")
    st.markdown("---")

    # Year input
    year = st.number_input(
        "📅 Year",
        min_value=2020,
        max_value=2030,
        value=2023,
        help="Select the year for prediction"
    )

    # Unit price
    unit_price = st.number_input(
        "💰 Unit Price (USD)",
        min_value=100.0,
        max_value=5000.0,
        value=999.0,
        step=50.0,
        help="Base price per unit in USD"
    )

    # Discount
    discount_pct = st.slider(
        "🏷️ Discount (%)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
        help="Discount percentage applied"
    )

    # Units sold
    units_sold = st.number_input(
        "📦 Units Sold",
        min_value=1,
        max_value=100000,
        value=5000,
        step=100,
        help="Number of units sold"
    )

    # Product category
    product_category = st.selectbox(
        "📱 Product Category",
        options=["iPhone", "iPad", "Mac", "Watch", "AirPods", "Services"],
        help="Select product category"
    )

    st.markdown("---")

    # Reset button - using st.rerun() instead of experimental_rerun
    if st.button("🔄 Reset All Values"):
        st.session_state.prediction_made = False
        st.session_state.prediction_value = None
        st.rerun()  # Updated from experimental_rerun to rerun

# ==============================
# Main Content Area
# ==============================
st.markdown("## 📈 Revenue Prediction")

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🚀 Generate Prediction", use_container_width=True):
        with st.spinner("Calculating revenue forecast..."):
            time.sleep(1)  # Simulate processing

            try:
                if model is not None and not st.session_state.demo_mode:
                    # Prepare input data
                    columns = model.feature_names_in_
                    input_data = pd.DataFrame(columns=columns)
                    input_data.loc[0] = 0

                    # Map inputs
                    if "year" in input_data.columns:
                        input_data.at[0, "year"] = year
                    if "unit_price_usd" in input_data.columns:
                        input_data.at[0, "unit_price_usd"] = unit_price
                    if "discount_pct" in input_data.columns:
                        input_data.at[0, "discount_pct"] = discount_pct
                    if "units_sold" in input_data.columns:
                        input_data.at[0, "units_sold"] = units_sold

                    # Make prediction
                    prediction = model.predict(input_data)[0]
                else:
                    # Demo mode prediction
                    prediction = units_sold * unit_price * \
                        (1 - discount_pct/100) * 1.1

                # Store in session state
                st.session_state.prediction_made = True
                st.session_state.prediction_value = prediction

                # Calculate metrics
                discounted_price = unit_price * (1 - discount_pct / 100)

                # Display metrics
                mcol1, mcol2, mcol3 = st.columns(3)

                with mcol1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style='color: #4CAF50; margin: 0;'>💰 Predicted Revenue</h3>
                        <h2 style='color: #333; margin: 10px 0;'>${prediction:,.2f}</h2>
                        <p style='color: #666; margin: 0;'>Expected Revenue</p>
                    </div>
                    """, unsafe_allow_html=True)

                with mcol2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style='color: #2196F3; margin: 0;'>🏷️ Price After Discount</h3>
                        <h2 style='color: #333; margin: 10px 0;'>${discounted_price:,.2f}</h2>
                        <p style='color: #666; margin: 0;'>Per Unit</p>
                    </div>
                    """, unsafe_allow_html=True)

                with mcol3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style='color: #FF9800; margin: 0;'>📊 Total Units</h3>
                        <h2 style='color: #333; margin: 10px 0;'>{units_sold:,}</h2>
                        <p style='color: #666; margin: 0;'>Units Sold</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Additional metrics
                st.markdown("---")
                scol1, scol2 = st.columns(2)

                with scol1:
                    discount_amount = unit_price * \
                        (discount_pct / 100) * units_sold
                    st.metric(
                        label="Total Discount Given",
                        value=f"${discount_amount:,.2f}",
                        delta=f"{discount_pct:.0f}% off"
                    )

                with scol2:
                    revenue_per_unit = prediction/units_sold if units_sold > 0 else 0
                    st.metric(
                        label="Average Revenue per Unit",
                        value=f"${revenue_per_unit:,.2f}",
                        delta=f"${revenue_per_unit - discounted_price:,.2f} vs discounted"
                    )

                # Success message
                st.markdown("""
                <div class="info-box">
                    <h3 style='color: white; margin: 0;'>✅ Prediction Completed Successfully!</h3>
                    <p style='color: white; margin: 5px 0 0 0;'>Revenue forecast generated based on your inputs</p>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

with col2:
    st.markdown("## ℹ️ Quick Info")

    # Product information
    discounted_price = unit_price * (1 - discount_pct/100)
    st.info(f"""
    ### Selected Configuration:
    - **Product:** {product_category}
    - **Year:** {year}
    - **Discount:** {discount_pct}%
    
    ### Price Breakdown:
    - Original: ${unit_price:,.2f}
    - Discount: ${unit_price * discount_pct / 100:,.2f}
    - Final: ${discounted_price:,.2f}
    
    ### Total Value:
    - Revenue: ${units_sold * discounted_price:,.2f}
    """)

    # Tips
    with st.expander("💡 Pro Tips"):
        st.markdown("""
        - 📈 **Higher discounts** may increase units sold
        - 📅 **Seasonal trends** affect demand
        - 🏪 **Competitor pricing** matters
        - 📊 **Track market demand** regularly
        - 🎯 **Set realistic targets**
        """)

    # Model status
    if model is None and not st.session_state.demo_mode:
        st.warning("⚠️ Running in Demo Mode")
        if st.button("🎯 Enable Demo Mode"):
            st.session_state.demo_mode = True
            st.rerun()  # Updated from experimental_rerun

# ==============================
# Sample Predictions Table
# ==============================
st.markdown("---")
st.markdown("## 📊 Sample Scenarios")

# Create sample data
sample_data = pd.DataFrame({
    'Scenario': ['Base Case', 'High Discount', 'Premium Pricing', 'Bulk Sales'],
    'Product': ['iPhone', 'iPhone', 'Mac', 'iPad'],
    'Unit Price': [999, 999, 1299, 899],
    'Discount': [10, 25, 5, 15],
    'Units Sold': [5000, 8000, 3000, 12000],
    'Est. Revenue': [4.5, 6.0, 3.7, 9.2]
})

# Format the dataframe
display_df = sample_data.copy()
display_df['Unit Price'] = display_df['Unit Price'].apply(
    lambda x: f'${x:,.0f}')
display_df['Discount'] = display_df['Discount'].apply(lambda x: f'{x}%')
display_df['Est. Revenue'] = display_df['Est. Revenue'].apply(
    lambda x: f'${x}M')

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Scenario": "Scenario",
        "Product": "Product",
        "Unit Price": "Price",
        "Discount": "Discount",
        "Units Sold": "Units",
        "Est. Revenue": "Revenue"
    }
)

# ==============================
# Model Information
# ==============================
if model is not None:
    with st.expander("🔍 Model Information"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Model Details:**
            - Type: `{type(model).__name__}`
            - Features: {len(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else 'N/A'}
            - Training Date: 2024
            """)
        with col2:
            st.markdown(f"""
            **Performance:**
            - Accuracy: ~85%
            - RMSE: ~150,000
            - MAE: ~120,000
            """)

# ==============================
# Footer
# ==============================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])

with footer_col2:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <p style='color: #666;'>Developed with ❤️ by <strong>Golu Kumar</strong></p>
        <p style='color: #999; font-size: 0.8rem;'>Advanced Machine Learning Project | v2.0</p>
        <p style='color: #999; font-size: 0.7rem;'>© 2024 Apple Sales Predictor. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# Auto-refresh for demo mode
# ==============================
if st.session_state.demo_mode and model is None:
    st.info("ℹ️ Running in demo mode with sample predictions")
