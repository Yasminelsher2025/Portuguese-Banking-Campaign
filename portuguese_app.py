

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================================
# 1. LOADING THE STANDARD PIPELINE
# =========================================================================
@st.cache_resource
def load_production_pipeline():
    return joblib.load('random_forest_pipeline.pkl')

Random_forest_pipeline = load_production_pipeline()
THRESHOLD = 0.47

# =========================================================================
# 2. STREAMLIT INTERFACE (UI) FOR ALL 13 REQUIRED COLUMNS
# =========================================================================
st.set_page_config(layout= 'wide')

st.title("Portuguese Banking Campaign Conversion Predictor")


col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.image("bank.jpg")


st.header("Lead Attributes")


col1, col2 = st.columns(2)

with col1:
    # Numerical fields
    customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
    balance = st.number_input("Annual Balance (in Euros)", value=1200)
    day_of_month = st.slider("Day of the Month (Last Contact)", min_value=1, max_value=31, value=15)
    num_contacts_in_campaign = st.number_input("Number of Contacts in this Campaign", min_value=1, value=1)
    num_contacts_prev_campaign = st.number_input("Number of Contacts in Previous Campaigns", min_value=0, value=0)
    
    # Categorical fields
    job_type = st.selectbox("Job Type", ['blue-collar', 'management', 'technician', 'admin.', 'services', 'retired', 'self-employed', 'entrepreneur', 'unemployed', 'housemaid', 'student', 'unknown'])
    marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])

with col2:
    education = st.selectbox("Education Level", ['secondary', 'tertiary', 'primary', 'unknown'])
    month = st.selectbox("Last Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    communication_type = st.selectbox("Communication Type", ['cellular', 'telephone', 'unknown'])
    
    # Binary fields
    default = st.selectbox("Has Credit in Default?", ['no', 'yes'])
    housing_loan = st.selectbox("Has Housing Loan?", ['no', 'yes'])
    personal_loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])

# =========================================================================
# 3. BUILDING THE MATCHING DATAFRAME ROW
# =========================================================================

new_data = pd.DataFrame([{
    'customer_age': customer_age,
    'job_type': job_type,
    'marital': marital,
    'education': education,
    'default': default,
    'balance': balance,
    'housing_loan': housing_loan,
    'personal_loan': personal_loan,
    'communication_type': communication_type,
    'day_of_month': day_of_month,
    'month': month,
    'num_contacts_in_campaign': num_contacts_in_campaign,
    'num_contacts_prev_campaign': num_contacts_prev_campaign
    
}])

# =========================================================================
# 4. PREDICTING USING YOUR THRESHOLD INLINE
# =========================================================================
st.divider()

if st.button("Evaluate Lead Strategy", type="primary"):
    
    probabilities = Random_forest_pipeline.predict_proba(new_data)[:, 1]
    lead_probability = probabilities[0]
    
    prediction = (lead_probability >= THRESHOLD).astype(int)
    
    st.subheader("Operational Verdict")
    st.metric(label="Subscription Probability Score", value=f"{round(lead_probability * 100, 2)}%")
    
    if prediction == 1:
        st.success("🎯 **High Conversion Potential!** Move this prospect to priority contact queue. They clear the 0.47 decision threshold.")
    else:
        st.warning("⚠️ **Low Conversion Potential.** Drop from aggressive phone campaign list to save outreach overhead costs.")
