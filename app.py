
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #f1f5f9;
}
.block-container {
    padding-top: 2rem;
    max-width: 1150px;
}
h1, h2, h3 {
    color: #f8fafc;
}
p, label, .stMarkdown {
    color: #cbd5e1;
}
[data-testid="stMetric"] {
    background: #172338;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #263750;
}
.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    height: 3.2rem;
    font-weight: 600;
    width: 100%;
}
.stButton > button:hover {
    background: #1d4ed8;
    color: white;
}
[data-testid="stForm"] {
    background: #111c2e;
    padding: 20px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    saved = joblib.load("churn_model.pkl")
    return saved["model"], saved["features"]

model, features = load_model()

# ---------------- HEADER ----------------
st.markdown("# 📊 Customer Churn Analyzer")
st.markdown(
    "AI-powered telecom customer retention prediction"
)
st.caption("Machine Learning • Random Forest • Customer Analytics")

st.divider()

# ---------------- INPUT FORM ----------------
with st.form("customer_form"):

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, value=30)
        subscription = st.number_input(
            "Subscription Length", min_value=1, value=20
        )
        status = st.selectbox("Status", [1, 2])

    with col2:
        complains = st.selectbox(
            "Customer Complaints",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        charge = st.number_input(
            "Charge Amount", min_value=0, value=2
        )
        customer_value = st.number_input(
            "Customer Value", min_value=0, value=100
        )

    with col3:
        call_failure = st.number_input(
            "Call Failure", min_value=0, value=90
        )
        frequency = st.number_input(
            "Frequency of Use", min_value=0, value=20
        )
        distinct = st.number_input(
            "Distinct Called Numbers", min_value=0, value=10
        )

    st.divider()
    st.subheader("📱 Usage Information")

    col4, col5 = st.columns(2)

    with col4:
        seconds = st.number_input(
            "Seconds of Use", min_value=0, value=990
        )

    with col5:
        sms = st.number_input(
            "Frequency of SMS", min_value=0, value=10
        )

    predict = st.form_submit_button(
        "🔍 Analyze Customer Churn"
    )

# ---------------- PREDICTION ----------------
if predict:

    # Engineered features
    sms_ratio = sms / seconds if seconds != 0 else 0
    failure_rate = (
        call_failure / frequency if frequency != 0 else 0
    )
    usage_per_month = (
        seconds / subscription if subscription != 0 else 0
    )

    # Hidden model features:
    # These defaults must be checked against your training data.
    customer = {
        "Call  Failure": call_failure,
        "Complains": complains,
        "Subscription  Length": subscription,
        "Charge  Amount": charge,
        "Seconds of Use": seconds,
        "Frequency of use": frequency,
        "Frequency of SMS": sms,
        "Distinct Called Numbers": distinct,
        "Age Group": 5,
        "Tariff Plan": 2,
        "Status": status,
        "Age": age,
        "Customer Value": customer_value,
        "SMS_Call_Ratio": sms_ratio,
        "Call_Failure_Rate": failure_rate,
        "Usage_Per_Month": usage_per_month
    }

    input_df = pd.DataFrame([customer])[features]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # ---------------- RESULTS ----------------
    st.divider()
    st.subheader("📈 Prediction Analysis")

    result_col, probability_col = st.columns(2)

    with result_col:
        if prediction == 1:
            st.error("⚠️ Customer is predicted to CHURN")
        else:
            st.success("✅ Customer is predicted to STAY")

    with probability_col:
        st.metric(
            "Estimated Churn Probability",
            f"{probability:.1%}"
        )

    st.progress(float(probability))

    if prediction == 1:
        st.warning(
            "The model predicts that this customer may leave. "
            "Consider reviewing their service experience."
        )
    else:
        st.info(
            "The model predicts that this customer is likely to stay."
        )

    st.caption(
        "This is a machine-learning prediction, not a guarantee "
        "of actual customer behavior."
    )

