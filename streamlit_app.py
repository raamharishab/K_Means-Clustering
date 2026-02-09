import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(page_title="K-Means Clustering", layout="wide")

st.title("🎯 K-Means Customer Segmentation")
st.markdown("Predict customer segment based on Annual Income and Spending Score")

# Clear cache button
if st.button("🔄 Reload Models"):
    st.cache_resource.clear()
    st.success("Cache cleared! Models reloaded.")
    st.rerun()

# Check if models exist
if not os.path.exists("means_model.pkl") or not os.path.exists("scaler.pkl"):
    st.error("⚠️ Model files not found! Please run the notebook first to generate the models.")
    st.stop()

# Load model and scaler with caching
@st.cache_resource
def load_models():
    model = joblib.load("means_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Create tabs for input method
input_tab1, input_tab2 = st.tabs(["📊 Slider Input", "⌨️ Text Input"])

with input_tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        annual_income_slider = st.slider(
            "Annual Income (k$)",
            min_value=15,
            max_value=137,
            value=50,
            step=1,
            key="slider_income"
        )
    
    with col2:
        spending_score_slider = st.slider(
            "Spending Score (1-100)",
            min_value=1,
            max_value=100,
            value=50,
            step=1,
            key="slider_spending"
        )
    
    annual_income = annual_income_slider
    spending_score = spending_score_slider

with input_tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        annual_income_input = st.number_input(
            "Annual Income (k$)",
            min_value=15,
            max_value=137,
            value=50,
            step=1,
            key="input_income"
        )
    
    with col2:
        spending_score_input = st.number_input(
            "Spending Score (1-100)",
            min_value=1,
            max_value=100,
            value=50,
            step=1,
            key="input_spending"
        )
    
    annual_income = annual_income_input
    spending_score = spending_score_input

# Predict button
if st.button("Predict Cluster", type="primary", use_container_width=True):
    try:
        # Prepare features
        features = np.array([[annual_income, spending_score]])
        scaled_features = scaler.transform(features)
        
        # Predict cluster
        cluster = model.predict(scaled_features)[0]
        
        # Display result
        st.success(f"✅ Predicted Cluster: **{cluster}**")
        
        # Debug info in expander
        with st.expander("🔍 Debug Info"):
            st.write(f"**Input Values:**")
            st.write(f"- Annual Income: ${annual_income}k")
            st.write(f"- Spending Score: {spending_score}/100")
            st.write(f"\n**Scaled Values (sent to model):**")
            st.write(f"- Income Scaled: {scaled_features[0][0]:.4f}")
            st.write(f"- Spending Scaled: {scaled_features[0][1]:.4f}")
            st.write(f"\n**Model Output:** Cluster {cluster}")
        
        # Show cluster interpretation
        cluster_descriptions = {
            0: "🔴 Low Income, Low Spending - Budget Conscious",
            1: "🟡 Low Income, High Spending - Shop Lovers",
            2: "🟢 Average Income, Average Spending - Standard Customers",
            3: "🔵 High Income, Low Spending - Savers",
            4: "🟣 High Income, High Spending - Premium Customers"
        }
        
        if cluster in cluster_descriptions:
            st.info(cluster_descriptions[cluster])
        
        # Display input values
        st.subheader("Your Input:")
        st.write(f"- Annual Income: **${annual_income}k**")
        st.write(f"- Spending Score: **{spending_score}/100**")
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses K-Means clustering to segment customers into 5 groups based on:
    - **Annual Income** (in thousands of dollars)
    - **Spending Score** (1-100 scale)
    
    ### 5 Customer Segments:
    1. **Cluster 0**: Budget Conscious
    2. **Cluster 1**: Shop Lovers  
    3. **Cluster 2**: Standard Customers
    4. **Cluster 3**: Savers
    5. **Cluster 4**: Premium Customers
    """)
    
    st.markdown("---")
    st.markdown("📊 Dataset: Mall Customers")
