# app.py
# Ultra-Simplified Streamlit app for Car Price Prediction (Minimalist UI)

import os
import pickle
import pandas as pd
import streamlit as st

# --- 1. Configuration & Setup ---
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# --- 2. Utility: Load Model (Hidden from User) ---
@st.cache_resource
def load_model(model_path: str):
    """Loads the trained scikit-learn pipeline."""
    if not os.path.exists(model_path):
        # We handle the error internally and halt, so the user sees a clean error, not technical files.
        st.error("Prediction service is currently unavailable. Please contact the administrator.")
        st.stop()
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error("Prediction service is currently unavailable due to a model loading error.")
        st.stop()

# Load the model pipeline
MODEL_PATH = "linear_regression_pipeline.joblib"
model = load_model(MODEL_PATH)

# --- 3. Main Application Layout (User-Focused) ---
st.title("🚗 Predict Your Car's Value")

# This subheader is the primary instruction for the user
st.subheader("Enter your vehicle's details below to get an estimated price.")

# Define the columns for input fields
col1, col2, col3, col4 = st.columns(4)

# Get user input for car features
with col1:
    make = st.selectbox(
        "Manufacturer", # User-friendly label
        options=["Toyota", "Honda", "BMW", "Nissan"],
        index=0
    )
with col2:
    colour = st.selectbox(
        "Exterior Colour", # User-friendly label
        options=["Black", "White", "Blue", "Red"],
        index=0
    )
with col3:
    # Set a user-friendly range and step for the odometer
    odometer_km = st.slider(
        "Mileage (KM)", # User-friendly label
        min_value=0,
        max_value=250000,
        value=60000,
        step=5000
    )
with col4:
    doors = st.selectbox(
        "Number of Doors", # User-friendly label
        options=[2, 3, 4, 5],
        index=2
    )

# --- 4. Prediction Logic ---
st.write("") # Add some vertical space
predict_btn = st.button("✨ Get Price Estimate", type="primary") # Use a primary button for emphasis

if predict_btn:
    try:
        # Step 1: Feature Engineering (Internal detail, hidden from user)
        is_4_door = 1 if float(doors) == 4.0 else 0

        # Step 2: Create the DataFrame required by the model
        input_data = {
            'Make': [make],
            'Colour': [colour],
            'Odometer (KM)': [float(odometer_km)],
            'Is_4_Door': [is_4_door]
        }
        input_df = pd.DataFrame(input_data)

        # Step 3: Make the prediction
        pred = model.predict(input_df)[0]
        
        # Step 4: Display the result (The only visible output)
        st.success("✅ Prediction Successful!")
        st.metric(label="Estimated Market Price", value=f"**${pred:,.0f}**") # Use ,.0f for cleaner currency
        

    except Exception as e:
        # Display a generic error message for the user
        st.error("We could not process your prediction at this time. Please check your inputs.")
