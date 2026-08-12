# Loan Approval Prediction - Streamlit Application

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# Load model and encoders from the same folder as this app
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
ENCODER_PATH = BASE_DIR / "encoders.pkl"


@st.cache_resource
def load_model():
    # Load the trained Random Forest model and categorical encoders
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return model, encoders


model, encoders = load_model()


# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)


# Title
st.title("🏦 Loan Approval Prediction System")

st.write(
    "Enter applicant and loan information to predict "
    "whether the loan is likely to be approved or rejected."
)

st.divider()


# ============================================================
# APPLICANT INFORMATION
# ============================================================

st.header("👤 Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    person_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

with col2:
    person_gender = st.selectbox(
        "Gender",
        encoders["person_gender"].classes_
    )

with col3:
    person_education = st.selectbox(
        "Education",
        encoders["person_education"].classes_
    )


col1, col2, col3 = st.columns(3)

with col1:
    person_income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

with col2:
    person_emp_exp = st.number_input(
        "Employment Experience (Years)",
        min_value=0,
        max_value=60,
        value=3
    )

with col3:
    person_home_ownership = st.selectbox(
        "Home Ownership",
        encoders["person_home_ownership"].classes_
    )


# ============================================================
# LOAN INFORMATION
# ============================================================

st.header("💰 Loan Information")

col1, col2, col3 = st.columns(3)

with col1:
    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=10000.0,
        step=500.0
    )

with col2:
    loan_intent = st.selectbox(
        "Loan Intent",
        encoders["loan_intent"].classes_
    )

with col3:
    loan_int_rate = st.number_input(
        "Loan Interest Rate (%)",
        min_value=0.0,
        max_value=30.0,
        value=10.0,
        step=0.01
    )


col1, col2 = st.columns(2)

with col1:
    loan_percent_income = st.number_input(
        "Loan Percent of Income",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01
    )

with col2:
    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (Years)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=1.0
    )


# ============================================================
# CREDIT INFORMATION
# ============================================================

st.header("🏦 Credit Information")

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=700
)

previous_loan_defaults_on_file = st.selectbox(
    "Previous Loan Default",
    encoders["previous_loan_defaults_on_file"].classes_
)


# ============================================================
# PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "🔮 Predict Loan Status",
    type="primary",
    use_container_width=True
)


if predict_button:

    # Convert categorical user inputs using the SAME encoders used during training
    gender_encoded = encoders["person_gender"].transform(
        [person_gender]
    )[0]

    education_encoded = encoders["person_education"].transform(
        [person_education]
    )[0]

    ownership_encoded = encoders["person_home_ownership"].transform(
        [person_home_ownership]
    )[0]

    intent_encoded = encoders["loan_intent"].transform(
        [loan_intent]
    )[0]

    default_encoded = encoders[
        "previous_loan_defaults_on_file"
    ].transform(
        [previous_loan_defaults_on_file]
    )[0]


    # Create input DataFrame in EXACTLY the same column order as training
    input_data = pd.DataFrame(
        [[
            person_age,
            gender_encoded,
            education_encoded,
            person_income,
            person_emp_exp,
            ownership_encoded,
            loan_amnt,
            intent_encoded,
            loan_int_rate,
            loan_percent_income,
            cb_person_cred_hist_length,
            credit_score,
            default_encoded
        ]],
        columns=[
            "person_age",
            "person_gender",
            "person_education",
            "person_income",
            "person_emp_exp",
            "person_home_ownership",
            "loan_amnt",
            "loan_intent",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
            "credit_score",
            "previous_loan_defaults_on_file"
        ]
    )


    # Make prediction
    prediction = model.predict(input_data)[0]


    # Get prediction probabilities
    probabilities = model.predict_proba(input_data)[0]

    probability_dict = dict(
        zip(model.classes_, probabilities)
    )

    approval_probability = probability_dict.get(1, 0)


    # Display result
    if prediction == 1:

        st.success(
            "## ✅ LOAN APPROVED\n\n"
            "The Random Forest model predicts that "
            "this application is likely to be approved."
        )

    else:

        st.error(
            "## ❌ LOAN REJECTED\n\n"
            "The Random Forest model predicts that "
            "this application is likely to be rejected."
        )


    # Display probability
    st.subheader("📊 Approval Probability")

    st.progress(float(approval_probability))

    st.write(
        f"Estimated probability of approval: "
        f"**{approval_probability * 100:.2f}%**"
    )


    # Show encoded data for verification
    with st.expander("🔍 View Encoded Data Sent to Model"):

        st.dataframe(
            input_data,
            use_container_width=True
        )


# Footer
st.divider()

st.caption(
    "Loan Approval Prediction | Machine Learning Project | "
    "Random Forest Classifier"
)