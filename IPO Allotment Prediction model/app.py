import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load Model
model = joblib.load('/Users/adi/Desktop/Mission/IPO Allotment Prediction model/ipo_model.pkl')

# Input Fields
oversubscription = st.number_input("Oversubscription Rate", min_value=1.0)
category = st.selectbox("Investor Category", ["Retail", "Institutional", "HNI"])
issue_size = st.number_input("Issue Size (Crore INR)", min_value=1)

# Preprocess Inputs
input_data = pd.DataFrame({
    'Log_Issue_Size': [np.log1p(issue_size)],
    'Total': [oversubscription],
    'Investor_Category': [category]
})

# Predict
if st.button("Predict"):
    probability = model.predict_proba(input_data)[0][1]
    st.write(f"Probability of Allotment: {probability:.2%}")