
# app.py
# Streamlit app to serve the car price prediction model trained in car_sales.py

import os
import io
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------
# Streamlit Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------
# Utility: Load model (cached)
# ---------------------------
@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Please run car_sales.py first to train and save the pipeline."
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# ---------------------------
# Helpers: Build input frames
# ---------------------------
REQUIRED_FEATURES = ['Make', 'Colour', 'Odometer (KM)', 'Is_4_Door']

def build_single_row(make: str, colour: str, odometer_km: float, doors: int) -> pd.DataFrame:
    """Create a single-row DataFrame with the exact feature schema expected by the pipeline."""
    is_4_door = 1 if float(doors) == 4.0 else 0
    row = {
        'Make': make,
        'Colour': colour,
        'Odometer (KM)': odometer_km,
        'Is_4_Door': is_4_door
    }
    return pd.DataFrame([row], columns=REQUIRED_FEATURES)

def prepare_batch_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a user-supplied DataFrame that may contain 'Doors' or 'Is_4_Door'.
    Produces the exact schema required: ['Make', 'Colour', 'Odometer (KM)', 'Is_4_Door'].
    """
    df = df.copy()

    # Derive Is_4_Door if 'Doors' exists and 'Is_4_Door' does not
    if 'Is_4_Door' not in df.columns and 'Doors' in df.columns:
        # Defensive conversion to numeric
        doors_num = pd.to_numeric(df['Doors'], errors='coerce')
        df['Is_4_Door'] = np.where(doors_num == 4.0, 1, 0)

    # If both exist, prefer explicit Is_4_Door and drop Doors
    if 'Doors' in df.columns:
        df = df.drop(columns=['Doors'])

    # Ensure all required columns exist
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns for prediction: "
            f"{missing}. Expected columns: {REQUIRED_FEATURES}.\n"
            "Tip: include either 'Doors' (will be converted) or 'Is_4_Door'."
        )

    # Coerce odometer to numeric (model’s numeric pipeline handles imputation/scaling)
    df['Odometer (KM)'] = pd.to_numeric(df['Odometer (KM)'], errors='coerce')
    # Coerce Is_4_Door to 0/1 integers if needed
    df['Is_4_Door'] = pd.to_numeric(df['Is_4_Door'], errors='coerce').fillna(0).astype(int)

    # Keep only the required features in correct order
    df = df[REQUIRED_FEATURES]
    return df

# ---------------------------
# Sidebar: Paths & Info
# ---------------------------
st.sidebar.title("⚙️ Settings")
default_model_path = "linear_regression_pipeline.joblib"
model_path = st.sidebar.text_input("Model file path", value=default_model_path)

st.sidebar.markdown(
    """
**How to get the model file:**
1. Run `python car_sales.py` (or `python3 car_sales.py`).
2. Confirm it saved **linear_regression_pipeline.joblib** in your project folder.
3. Come back and click **Reload model** below.
"""
)

reload_clicked = st.sidebar.button("🔄 Reload model")

# ---------------------------
# Main: App Header
# ---------------------------
st.title("🚗 Car Price Predictor")
st.caption("Uses the trained scikit-learn pipeline (Linear Regression) from `car_sales.py`.")

# Try loading the model (with optional reload)
load_error = None
model = None
try:
    if reload_clicked:
        # Clear cache and reload
        load_model.clear()
    model = load_model(model_path)
    st.success(f"Model loaded from: `{model_path}`")
except Exception as e:
    load_error = e
    st.error(str(e))
    st.stop()

# ---------------------------
# Tabs: Single & Batch Predictions + Model Info
# ---------------------------
tab_single, tab_batch, tab_info = st.tabs(["🎯 Single Prediction", "📦 Batch Predictions", "🔍 Model Info"])

# --- Single Prediction Tab ---
with tab_single:
    st.subheader("Enter car details to predict price")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        make = st.text_input("Make", value="Toyota", help="Free text; unseen makes are okay (ignored by OHE).")
    with col2:
        colour = st.text_input("Colour", value="Black", help="Free text; unseen colours are okay.")
    with col3:
        odometer_km = st.number_input("Odometer (KM)", min_value=0.0, value=60000.0, step=1000.0)
    with col4:
        doors = st.selectbox("Doors", options=[2, 3, 4, 5], index=2)

    predict_btn = st.button("🔮 Predict Price")
    if predict_btn:
        try:
            input_df = build_single_row(make, colour, odometer_km, doors)
            pred = model.predict(input_df)[0]
            st.metric(label="Predicted Price", value=f"${pred:,.2f}")
            with st.expander("View input as DataFrame"):
                st.dataframe(input_df)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --- Batch Predictions Tab ---
with tab_batch:
    st.subheader("Upload a CSV for batch price predictions")
    st.markdown(
        """
**Accepted columns:**
- `Make` (str)
- `Colour` (str)
- `Odometer (KM)` (numeric)
- `Doors` (int) **or** `Is_4_Door` (0/1)

You may include extra columns; they will be ignored for prediction.
"""
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            raw_df = pd.read_csv(uploaded)
            st.write("Preview of uploaded data:")
            st.dataframe(raw_df.head())
            prepared_df = prepare_batch_frame(raw_df)
            preds = model.predict(prepared_df)
            result_df = raw_df.copy()
            result_df['Predicted Price'] = preds

            st.success("Batch predictions completed.")
            st.dataframe(result_df.head())

            # Download results
            csv_buf = io.StringIO()
            result_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="⬇️ Download predictions as CSV",
                data=csv_buf.getvalue(),
                file_name="car_price_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

# --- Model Info Tab ---
with tab_info:
    st.subheader("Pipeline Overview")
    try:
        pipe = model  # This is the full Pipeline: preprocessor + regressor
        st.write("**Top-level steps:**")
        st.code(list(pipe.named_steps.keys()), language="python")

        pre = pipe.named_steps.get('preprocessor', None)
        reg = pipe.named_steps.get('regressor', None)

        if pre is not None:
            st.write("**Preprocessor (ColumnTransformer):**")
            for name, trans, cols in pre.transformers_:
                st.markdown(f"- **{name}** → columns: `{cols}`")
                if hasattr(trans, 'steps'):
                    st.markdown("  - Steps:")
                    for sname, _s in trans.steps:
                        st.markdown(f"    - {sname}")
        if reg is not None:
            st.write("**Regressor:**", type(reg).__name__)
    except Exception as e:
        st.warning(f"Could not extract pipeline details: {e}")

    st.divider()
    st.subheader("Re-evaluate on the dataset (optional)")
    st.caption("Uses the same dataset path as in your training script: `dataset/car-sales-extended-missing-data.csv`.")
    reevaluate = st.button("📊 Evaluate model on held-out test split")
    if reevaluate:
        try:
            # Recreate the same split and evaluation used in car_sales.py
            file_path = 'dataset/car-sales-extended-missing-data.csv'
            cs = pd.read_csv(file_path)

            X = cs[['Make', 'Colour', 'Odometer (KM)', 'Doors']].copy()
            y = cs['Price'].copy()

            # Feature engineering: Is_4_Door and drop Doors
            X['Is_4_Door'] = np.where(pd.to_numeric(X['Doors'], errors='coerce') == 4.0, 1, 0)
            X = X.drop(columns=['Doors'])

            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error, r2_score

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Impute y_test to avoid NaN in metrics
            y_test_clean = y_test.fillna(y_test.mean())

            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test_clean, preds)
            r2 = r2_score(y_test_clean, preds)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}")
            with c2:
                st.metric("R-squared (R²)", f"{r2:.4f}")
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

# ---------------------------
# Footer
# ---------------------------
st.divider()
