import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('salary_model.pkl')
le_gender = joblib.load('le_gender.pkl')
le_edu = joblib.load('le_edu.pkl')
le_job = joblib.load('le_job.pkl')

st.title("💼 Employee Salary Prediction")

# Input fields
age = st.slider("Age", 18, 65, 30)
gender = st.selectbox("Gender", le_gender.classes_)
education = st.selectbox("Education Level", le_edu.classes_)
job_title = st.selectbox("Job Title", le_job.classes_)
experience = st.slider("Years of Experience", 0, 40, 5)

if st.button("Predict Salary"):
    # Encode inputs
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': le_gender.transform([gender]),
        'Education Level': le_edu.transform([education]),
        'Job Title': le_job.transform([job_title]),
        'Years of Experience': [experience]
    })

    # Predict
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Salary: ₹{int(prediction):,}")
