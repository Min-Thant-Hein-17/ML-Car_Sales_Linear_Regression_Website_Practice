# app.py
# Simplified Streamlit app for Car Price Prediction (Single Prediction Only)

import os
import pickle
import pandas as pd
import streamlit as st
import numpy as np # Needed for the Is_4_Door logic

# --- 1. Streamlit Page Configuration ---
st.set_page_config(
    page_title="Simple Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# --- 2. Utility: Load Model (Cached) ---
@st.cache_resource
def load_model(model_path: str):
    """Loads the trained scikit-learn pipeline."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at '{model_path}'. Please run car_sales.py first.")
        st.stop()
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model pipeline
MODEL_PATH = "linear_regression_pipeline.joblib"
model = load_model(MODEL_PATH)

# --- 3. Main Application Layout ---
st.title("🚗 Simple Car Price Predictor")
st.caption("Predict car price using a loaded Linear Regression model.")

# Display model status
st.success(f"Model loaded successfully from: `{MODEL_PATH}`")

st.subheader("Enter car details to predict price")

# Define the columns for input fields
col1, col2, col3, col4 = st.columns(4)

# Get user input for car features
with col1:
    make = st.selectbox(
        "Make",
        options=["Toyota", "Honda", "BMW", "Nissan"], # Only show common makes for simplicity
        index=0
    )
with col2:
    colour = st.selectbox(
        "Colour",
        options=["Black", "White", "Blue", "Red"], # Only show common colours
        index=0
    )
with col3:
    # Set a more intuitive range and step for the odometer
    odometer_km = st.slider(
        "Odometer (KM)",
        min_value=0,
        max_value=250000,
        value=60000,
        step=5000
    )
with col4:
    doors = st.selectbox(
        "Doors",
        options=[2, 3, 4, 5],
        index=2
    )

# --- 4. Prediction Logic ---
predict_btn = st.button("🔮 Predict Price")

if predict_btn:
    try:
        # Convert Doors to the required 'Is_4_Door' binary feature
        is_4_door = 1 if float(doors) == 4.0 else 0

        # Create the DataFrame with the exact features/schema required by the model
        input_data = {
            'Make': [make],
            'Colour': [colour],
            'Odometer (KM)': [float(odometer_km)],
            'Is_4_Door': [is_4_door]
        }
        input_df = pd.DataFrame(input_data)

        # Make the prediction
        pred = model.predict(input_df)[0]
        
        # Display the result in a large, clear format
        st.metric(label="Predicted Price", value=f"**${pred:,.2f}**")
        
        # Optionally, show the final data sent to the model for verification
        with st.expander("Show Data Sent to Model"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"Prediction failed. An error occurred: {e}")

# --- Footer ---
st.divider()
st.info("Note: This simple app focuses on single predictions. Model trained with `car_sales.py`.")
