import streamlit as st
import pandas as pd
import joblib
import json
import os

# ------------------------------
# RAG IMPORTS
# ------------------------------
from rag.retrieve import retrieve
from rag.generate import generate_response

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
    page_title="AI Powered Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Churn Early Warning System")
st.caption("AI + ML powered churn prediction with explainability")

# ------------------------------
# CHECK GEMINI API KEY
# ------------------------------
if not os.getenv("GEMINI_API_KEY"):
    st.warning("⚠ GEMINI_API_KEY not found. AI features will not work.")

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
# PREDICTION BUTTON
# ------------------------------
if st.button("🚀 Predict Churn"):

    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0][1]

    st.markdown("---")
    st.subheader("🔍 Prediction Result")

    col1, col2 = st.columns(2)

    if prediction == 1:
        col1.error("⚠ Customer likely to churn!")
        pred_text = f"Customer likely to churn with probability {proba:.2%}"
    else:
        col1.success("😊 Customer is NOT likely to churn.")
        pred_text = f"Customer not likely to churn. Probability {proba:.2%}"

    col2.metric("Churn Probability", f"{proba:.2%}")

    # Animation only for non-churn
    if prediction == 0:
        st.balloons()

    # ------------------------------
    # 🤖 RAG AI EXPLANATION
    # ------------------------------
    st.markdown("---")
    st.subheader("🤖 AI-Powered Explanation & Strategy")

    with st.spinner("Analyzing customer behavior with AI..."):

        try:
            query = "Why will this customer churn and what actions should be taken?"
            retrieved_docs = retrieve(query)

            explanation = generate_response(pred_text, retrieved_docs)

            st.success(explanation)

        except Exception as e:
            st.error("❌ Failed to generate AI explanation")
            st.exception(e)

# ------------------------------
# 💬 CHATBOT SECTION
# ------------------------------
st.markdown("---")
st.subheader("💬 AI Churn Assistant")

st.write("Ask anything about churn, retention, or customer behavior")

user_query = st.text_input("Type your question...")

if user_query:

    with st.spinner("Thinking..."):

        try:
            retrieved_docs = retrieve(user_query)
            answer = generate_response("General Query", retrieved_docs)

            st.info(answer)

        except Exception as e:
            st.error("❌ Error generating response")
            st.exception(e)
