import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

MODEL_FILE = "churn_model.joblib"
if not os.path.exists(MODEL_FILE):
    st.error("churn_model.joblib not found. Run the notebook (Section 10) first so the model gets saved here.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

bundle = load_model()
model, model_name, columns = bundle["model"], bundle["model_name"], bundle["feature_columns"]

st.title("📉 Customer Churn Prediction")
st.caption(f"Model in use: **{model_name}** (tuned) | Dataset: IBM Telco Customer Churn")

tab_predict, tab_models = st.tabs(["🔮 Predict churn", "📊 Model comparison"])

with tab_predict:
    st.subheader("Enter customer details")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Has partner", ["No", "Yes"])
        dependents = st.selectbox("Has dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with c2:
        st.markdown("**Services**")
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multi = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        no_net = "No internet service"
        opts = ["No", "Yes"] if internet != "No" else [no_net]
        security = st.selectbox("Online security", opts)
        backup = st.selectbox("Online backup", opts)
        device = st.selectbox("Device protection", opts)
        tech = st.selectbox("Tech support", opts)
        tv = st.selectbox("Streaming TV", opts)
        movies = st.selectbox("Streaming movies", opts)

    with c3:
        st.markdown("**Contract & billing**")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                  "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5)
        total = st.number_input("Total charges ($)", 0.0, 9000.0, float(round(tenure * monthly, 2)), step=10.0,
                                help="Defaults to tenure x monthly charges; edit if you know the exact value.")

    if st.button("Predict churn risk", type="primary"):
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone, "MultipleLines": multi if phone == "Yes" else "No phone service",
            "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": device, "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }])[columns]

        prob = float(model.predict_proba(row)[0, 1])
        st.divider()
        m1, m2 = st.columns([1, 2])
        m1.metric("Churn probability", f"{prob:.1%}")
        if prob >= 0.65:
            m2.error("🔴 HIGH RISK – customer is likely to leave. Offer a retention plan immediately.")
        elif prob >= 0.40:
            m2.warning("🟠 MEDIUM RISK – monitor and consider a loyalty offer.")
        else:
            m2.success("🟢 LOW RISK – customer is likely to stay.")
        st.progress(min(max(prob, 0.0), 1.0))

        tips = []
        if contract == "Month-to-month":
            tips.append("Offer a discount for moving to a 1- or 2-year contract.")
        if tech == "No":
            tips.append("Offer free Tech Support for a few months.")
        if security == "No":
            tips.append("Bundle Online Security as a free add-on.")
        if payment == "Electronic check":
            tips.append("Encourage automatic payment (bank transfer / credit card).")
        if tips and prob >= 0.40:
            st.markdown("**Suggested retention actions:**")
            for t in tips:
                st.markdown(f"- {t}")

with tab_models:
    st.subheader("Results from the notebook")
    if os.path.exists("model_comparison.csv"):
        res = pd.read_csv("model_comparison.csv")
        st.dataframe(res, use_container_width=True, hide_index=True)
        st.bar_chart(res.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]])
    else:
        st.info("model_comparison.csv not found. Run the notebook first.")
