import streamlit as st
import requests

st.title("Employee Attrition Predictor")

st.write("Enter employee details:")

age = st.number_input("Age", min_value=18, max_value=60)
income = st.number_input("Monthly Income", min_value=0)
job_sat = st.slider("Job Satisfaction", 1, 4)
worklife = st.slider("Work Life Balance", 1, 4)
distance = st.number_input("Distance From Home", min_value=0)
overtime = st.selectbox("OverTime", ["No", "Yes"])
years_company = st.number_input("Years At Company", min_value=0)
num_companies = st.number_input("Num Companies Worked", min_value=0)
performance = st.slider("Performance Rating", 1, 4)
total_years = st.number_input("Total Working Years", min_value=0)

overtime_val = 1 if overtime == "Yes" else 0

if age < 18:
    st.error("Age must be at least 18")
    st.stop()
if st.button("Predict"):

    features = [
        age,
        income,
        job_sat,
        worklife,
        distance,
        overtime_val,
        years_company,
        num_companies,
        performance,
        total_years
    ]

    try:
        response = requests.post(
            "http://backend:8000/predict",
            json={"features": features}
        )

        result = response.json()

        st.subheader("Prediction Result:")
        st.success(result["prediction"])

        if "confidence" in result:
            st.write(f"Confidence: {result['confidence']}%")
            st.progress(result["confidence"] / 100)

    except Exception as e:
        st.error("Backend not running! Start FastAPI server.")
        st.text(str(e))