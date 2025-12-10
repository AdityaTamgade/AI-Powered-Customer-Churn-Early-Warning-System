import streamlit as st
import pandas as pd
import joblib
import json

# ------------------------------
# LOAD MODELS + SCALER + FEATURES
# ------------------------------
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

with open("models/feature_names.json", "r") as f:
    feature_names = json.load(f)

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Early Warning System")
st.write("Fill in the customer details below to predict the probability of churn.")

# ------------------------------
# INPUT SECTIONS
# ------------------------------
with st.expander("👤 Customer Info", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        customer_id = st.number_input("Customer ID", min_value=1, value=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 10, 100, 30)
    with col2:
        tenure = st.slider("Tenure (months)", 0, 120, 12)
        plan_type = st.selectbox("Plan Type", ["Basic", "Standard", "Premium"])
        contract_type = st.selectbox("Contract Type", ["Monthly", "Yearly"])
    with col3:
        payment_method = st.selectbox("Payment Method", ["Credit Card", "UPI", "PayPal"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)

with st.expander("📈 Usage & Support", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        internet_usage_gb = st.slider("Internet Usage (GB/month)", 0, 500, 50)
    with col2:
        num_complaints = st.number_input("Number of Complaints", min_value=0, value=1)
    with col3:
        support_calls = st.number_input("Support Calls", min_value=0, value=1)

# ------------------------------
# FEATURE ENGINEERING
# ------------------------------
high_usage_flag = 1 if internet_usage_gb > 100 else 0
high_complaints_flag = 1 if num_complaints > 3 else 0
high_support_flag = 1 if support_calls > 5 else 0
tenure_months = tenure

# ------------------------------
# BUILD INPUT DF
# ------------------------------
input_dict = {
    "customer_id": customer_id,
    "gender": 1 if gender == "Male" else 0,
    "age": age,
    "tenure": tenure,
    "plan_type": ["Basic", "Standard", "Premium"].index(plan_type),
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "num_complaints": num_complaints,
    "support_calls": support_calls,
    "contract_type": ["Monthly", "Yearly"].index(contract_type),
    "payment_method": ["Credit Card", "UPI", "PayPal"].index(payment_method),
    "internet_usage_gb": internet_usage_gb,
    "high_usage_flag": high_usage_flag,
    "high_complaints_flag": high_complaints_flag,
    "high_support_flag": high_support_flag,
    "tenure_months": tenure_months
}

input_df = pd.DataFrame([input_dict])
input_df = input_df[feature_names]
scaled_input = scaler.transform(input_df)

# ------------------------------
# PREDICTION
# ------------------------------
if st.button("Predict Churn"):
    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1]

    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    
    col1, col2 = st.columns(2)
    if prediction == 1:
        col1.error(f"⚠ Customer likely to churn!")
    else:
        col1.success(f"😊 Customer is NOT likely to churn.")
    
    col2.metric(label="Churn Probability", value=f"{proba:.2%}")

    st.balloons()  # fun animation for positive result
