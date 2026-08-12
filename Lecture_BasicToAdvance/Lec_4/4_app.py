import streamlit as st
import joblib
import numpy as np

model = joblib.load("4_linearRegression.pkl")

st.title("Sales Prediction Application")

st.write("Please enter the following details to predict sales")

TV =st.number_input("TV Investment", min_value=0, max_value=1000, step=10)

RADIO = st.number_input("Radio Investment", min_value=0, max_value=1000, step=10)

NEWSPAPER = st.number_input("Newspaper Investment", min_value=0, max_value=1000, step=10)

if st.button("Predict Sales"):
    input_data = np.array([TV, RADIO, NEWSPAPER]).reshape(1, -1)
    prediction = model.predict(input_data)
    st.success(f"The predicted sales is: {prediction[0]:.2f}")