# app.py
# Simple and Userfriendly website with University and Student Info

import os
import pickle
import pandas as pd
import streamlit as st

#  1. Configuration & Setup 
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)


#  2. Sidebar for Personal/University Info & Links 
# Content placed in st.sidebar will appear on the left side of the page.
st.sidebar.title("🚗 Carl for Car Industries")

# Displaying University Logo, By using URL
logo_url = "https://talloiresnetwork.tufts.edu/wp-content/uploads//Parami-University-1.png"

if logo_url and "example.com" not in logo_url:
    st.sidebar.image(logo_url, use_column_width=True)
else:
    st.sidebar.subheader("Parami University")

# Use st.sidebar.markdown() for dividers
st.sidebar.markdown("---")

# My Info
st.sidebar.subheader("Owner's Information")
st.sidebar.write(f"**Name:** Min Thant Hein")
st.sidebar.write(f"**Student ID:** PIUS20230001")
st.sidebar.write(f"**Contact Email:** minthanthein@parami.edu.mm ")


st.sidebar.markdown("---")


# Adding Reliable Links in the Sidebar
st.sidebar.subheader("🔗 Market Resources")
# Using st.sidebar.markdown to include the link
st.sidebar.markdown("Check **Price Valuations** at [Kelley Blue Book](https://www.kbb.com).")
st.sidebar.markdown("**Industry News** from [Automotive News](https://www.autonews.com/).")

st.sidebar.markdown("---")


#  3. Utility: Load Model (Hidden from User) (Back-End) 
@st.cache_resource
def load_model(model_path: str):
    """Loads the trained scikit-learn pipeline."""
    if not os.path.exists(model_path):
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

#  4. Main Application Layout (User-Focused) (Front-End) 

# Centering The Title Using Python Columns 
col_left, col_center, col_right = st.columns([1, 3, 1])

with col_center:
    # This places the title in the middle column, effectively centering it.
    st.title("🚗 Predict Your Car's Value")

# Subheader and inputs start below the centered title block
st.subheader("Enter your vehicle's details below to get an estimated price.")

# Define the columns for input fields
col1, col2, col3, col4 = st.columns(4)

# Get user input for car features
with col1:
    make = st.selectbox(
        "Manufacturer", 
        options=["Toyota", "Honda", "BMW", "Nissan"],
        index=0
    )
with col2:
    colour = st.selectbox(
        "Exterior Colour", 
        options=["Black", "White", "Blue", "Red"],
        index=0
    )
with col3:
    # Set a user-friendly range and step for the Odometer (KM)
    odometer_km = st.slider(
        "Mileage (KM)", 
        min_value=0,
        max_value=250000,
        value=60000,
        step=5000
    )
with col4:
    doors = st.selectbox(
        "Number of Doors", 
        options=[2, 3, 4, 5],
        index=2
    )


#  5. Prediction Logic 
st.write("") 
predict_btn = st.button("✨ Get Price Estimate", type="primary") 

if predict_btn:
    try:
        # Feature Engineering (Step 1)
        is_4_door = 1 if float(doors) == 4.0 else 0

        # Create the DataFrame required by the model (Step 2)
        input_data = {
            'Make': [make], 'Colour': [colour], 
            'Odometer (KM)': [float(odometer_km)], 'Is_4_Door': [is_4_door]
        }
        input_df = pd.DataFrame(input_data)

        # Make the prediction (Step 3)
        pred = model.predict(input_df)[0]
        
        # Display the result (Step 4)
        st.success("✅ Prediction Successful!")
        st.metric(label="Estimated Market Price", value=f"**${pred:,.0f}**")
        
        # Adding the Link in the Main Body (using st.markdown)
        st.markdown("---")
        st.subheader("Stay Updated on Market Trends")
        st.markdown("For market validation, check the latest valuations at [Kelley Blue Book](https://www.kbb.com).")
        st.markdown("---")
        

    except Exception as e:
        # Displaying an error message for the user
        st.error("We could not process your prediction at this time. Please check your inputs.")




