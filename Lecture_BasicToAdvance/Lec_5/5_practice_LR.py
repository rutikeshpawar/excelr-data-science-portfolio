import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Toyota Corolla Price Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- PATH RESOLUTION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "5_practice_LR_model.pkl")
DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "Data_Files", "Toyoto_Corrola.csv")

# --- DATA & MODEL CACHING ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_label_encoder():
    data = pd.read_csv(DATA_PATH, index_col=0)
    data["Model"] = data["Model"].astype(str)
    le = LabelEncoder()
    le.fit(data["Model"])
    return le

# Initialize instances
try:
    model = load_model()
    label_encoder = load_label_encoder()
    model_options = list(label_encoder.classes_)
except Exception as e:
    st.error(f"Initialization Error: Ensure model and data files are correctly placed. Details: {e}")
    st.stop()

# --- SIDEBAR: DOCUMENTATION & APP INFO ---
with st.sidebar:
    st.image("https://icons8.com", width=100)
    st.title("About the App")
    st.write(
        "This system utilizes a **Linear Regression Model** to estimate the market valuation "
        "of used Toyota Corolla vehicles based on physical attributes, mileage, and specification standards."
    )
    st.markdown("---")
    st.caption("Developed for Portfolio | Data Science Practice")

# --- MAIN PAGE DESIGN ---
st.title("🚗 Toyota Corolla Price Predictor")
st.markdown(
    "Fill in the vehicle specifications below to calculate an estimated market price instantly. "
    "The application handles target encoding automatically."
)
st.markdown("---")

# Section 1: Core Specifications
st.subheader("📋 Core Attributes")
col_model, col_age = st.columns([2, 1])

with col_model:
    selected_model = st.selectbox("Vehicle Model Variant", model_options, help="Select the exact variant layout code.")
    model_code = int(label_encoder.transform([selected_model])[0])

with col_age:
    age = st.number_input("Age (Months)", min_value=0, max_value=200, value=20, step=1, help="Age of the vehicle in months.")

# Section 2: Usage Metrics
st.subheader("📈 Performance & Usage")
col_km, col_hp, col_wt = st.columns(3)

with col_km:
    km = st.number_input("Odometer (KM)", min_value=0, value=50000, step=1000)
with col_hp:
    hp = st.number_input("Horsepower (HP)", min_value=0, value=100, step=1)
with col_wt:
    weight = st.number_input("Weight (kg)", min_value=500, max_value=2000, value=1100, step=5)

# Section 3: Engineering Layouts
st.subheader("⚙️ Technical Design")
col_dr, col_cyl, col_gr = st.columns(3)

with col_dr:
    doors = st.selectbox("Number of Doors", [3, 4, 5])
with col_cyl:
    cylinders = st.selectbox("Engine Cylinders", [3, 4, 5, 6], index=1) # Defaults to 4
with col_gr:
    gears = st.selectbox("Gearbox Count", [4, 5, 6], index=1) # Defaults to 5

st.markdown("---")

# --- PREDICTION EXECUTION ---
if st.button("🔥 Calculate Market Value", type="primary", use_container_width=True):
    with st.spinner("Processing architectural metrics against regression matrices..."):
        # Format feature array
        features = np.array([[model_code, age, km, hp, cylinders, doors, gears, weight]], dtype=float)
        prediction = model.predict(features)
        predicted_price = float(prediction[0])
        
        # Display Result Metric Card
        st.balloons()
        st.markdown("### 📊 Valuation Assessment")
        
        # Highlight values cleanly
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric(label="Estimated Valuation", value=f"₹{predicted_price:,.2f}")
        with metric_col2:
            st.metric(label="Calculated Model Code", value=f"ID #{model_code}")
            
        st.info(f"**Selected Spec Target:** {selected_model}")
